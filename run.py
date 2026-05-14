"""
基于 ONNX Runtime 推理，无需 PyTorch。支持 DirectML GPU 加速与 CPU 自动降级。
快捷键: F1=暂停/恢复按键, F2=退出

用法:
    python run.py                  # 默认参数
    python run.py --threshold 0.6  # 自定义阈值
    python run.py --no-act --show  # 仅预览不按键
"""
import ctypes
import os
import sys
import time
import threading
import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np

try:
    import onnxruntime as ort
    _HAS_DML = 'DmlExecutionProvider' in ort.get_available_providers()
except ImportError:
    try:
        import onnxruntime_directml as ort
        _HAS_DML = True
    except ImportError:
        print('错误: 需要安装 onnxruntime 或 onnxruntime-directml')
        print('  pip install onnxruntime-directml   (GPU加速，推荐)')
        print('  pip install onnxruntime             (仅CPU)')
        sys.exit(1)

if sys.platform == 'win32':
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(-4)
    except AttributeError:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

if sys.platform == 'win32':
    # Bump the global timer to 1ms so time.sleep can hit ~120fps loop intervals;
    # default Windows resolution is ~15.6ms which clamps capture_fps to ~64.
    try:
        import atexit
        ctypes.windll.winmm.timeBeginPeriod(1)
        atexit.register(ctypes.windll.winmm.timeEndPeriod, 1)
    except Exception:
        pass

CONFIG = {
    'data': {
        'frame_stack': 4,
        'img_height': 144,
        'img_width': 240,
        'grayscale': False,
        'keys': ['key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7'],
        'crop_region_pc': [0.335, 0.237, 0.665, 0.915],
        'crop_region_mobile': [0.0, 0.292, 1.0, 0.838],
    },
    'inference': {
        'capture_fps': 240,
        'confidence_threshold': 0.5,
        'key_delay': 0.02,
        'key_map': {
            'key_1': 'd', 'key_2': 'f', 'key_3': 'j', 'key_4': 'k',
            'key_5': 's', 'key_6': 'l', 'key_7': 'space',
        },
    },
}

SCANCODE_MAP = {'d': 32, 'f': 33, 'j': 36, 'k': 37, 's': 31, 'l': 38, 'space': 57}

PUL = ctypes.POINTER(ctypes.c_ulong)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [
        ('wVk', ctypes.c_ushort),
        ('wScan', ctypes.c_ushort),
        ('dwFlags', ctypes.c_ulong),
        ('time', ctypes.c_ulong),
        ('dwExtraInfo', PUL),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ('uMsg', ctypes.c_ulong),
        ('wParamL', ctypes.c_short),
        ('wParamH', ctypes.c_ushort),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ('dx', ctypes.c_long),
        ('dy', ctypes.c_long),
        ('mouseData', ctypes.c_ulong),
        ('dwFlags', ctypes.c_ulong),
        ('time', ctypes.c_ulong),
        ('dwExtraInfo', PUL),
    ]


class _Input_I(ctypes.Union):
    _fields_ = [('ki', _KeyBdInput), ('mi', _MouseInput), ('hi', _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [('type', ctypes.c_ulong), ('ii', _Input_I)]


def press_key(scan_code):
    extra = ctypes.c_ulong(0)
    ii_ = _Input_I()
    ii_.ki = _KeyBdInput(0, scan_code, 8, 0, ctypes.pointer(extra))
    x = _Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(scan_code):
    extra = ctypes.c_ulong(0)
    ii_ = _Input_I()
    ii_.ki = _KeyBdInput(0, scan_code, 10, 0, ctypes.pointer(extra))
    x = _Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def _get_pids_by_exe(exe_name):
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 2
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ('dwSize', wintypes.DWORD),
            ('cntUsage', wintypes.DWORD),
            ('th32ProcessID', wintypes.DWORD),
            ('th32DefaultHeapID', ctypes.POINTER(ctypes.c_ulong)),
            ('th32ModuleID', wintypes.DWORD),
            ('cntThreads', wintypes.DWORD),
            ('th32ParentProcessID', wintypes.DWORD),
            ('pcPriClassBase', ctypes.c_long),
            ('dwFlags', wintypes.DWORD),
            ('szExeFile', ctypes.c_wchar * 260),
        ]

    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return set()
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    pids = set()
    if kernel32.Process32FirstW(snap, ctypes.byref(entry)):
        while True:
            if entry.szExeFile.lower() == exe_name.lower():
                pids.add(entry.th32ProcessID)
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            if not kernel32.Process32NextW(snap, ctypes.byref(entry)):
                break
    kernel32.CloseHandle(snap)
    return pids


def find_hwnd_by_exe(exe_name):
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    pids = _get_pids_by_exe(exe_name)
    if not pids:
        return 0
    result_hwnd = 0
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _):
        nonlocal result_hwnd
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            result_hwnd = hwnd
            return False
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return result_hwnd


def get_window_rect(hwnd):
    from ctypes import wintypes
    rect = wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)


class HotkeyController:
    def __init__(self):
        self._act_enabled = threading.Event()
        self._act_enabled.set()
        self._stop_requested = threading.Event()
        self._listener = None

    @property
    def act_enabled(self):
        return self._act_enabled.is_set()

    @property
    def stop_requested(self):
        return self._stop_requested.is_set()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        if self._listener:
            self._listener.stop()
        self._stop_requested.set()

    def _run(self):
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.f1:
                    if self._act_enabled.is_set():
                        self._act_enabled.clear()
                    else:
                        self._act_enabled.set()
                elif key == keyboard.Key.f2:
                    self._stop_requested.set()
                    return False
            except Exception:
                pass
            return None

        self._listener = keyboard.Listener(on_press=on_press, suppress=False)
        self._listener.start()
        self._listener.join()


class RhythmAI:
    """音游AI推理引擎 (ONNX Runtime)"""

    def __init__(self, model_path, threshold=0.5, no_act=False, show=False, key_delay=0.02):
        data_cfg = CONFIG['data']
        inf_cfg = CONFIG['inference']
        self.threshold = threshold
        self.no_act = no_act
        self.show = show
        self.key_delay = key_delay

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers_to_try = ['DmlExecutionProvider', 'CPUExecutionProvider'] if _HAS_DML else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers_to_try)
        except Exception as e:
            print(f'警告: 加载模型失败 (providers={providers_to_try}): {e}')
            print('尝试仅使用 CPU...')
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

        active_providers = self.session.get_providers()
        if 'DmlExecutionProvider' in active_providers:
            self.device_name = 'GPU (DirectML)'
        elif 'CUDAExecutionProvider' in active_providers:
            self.device_name = 'GPU (CUDA)'
        else:
            self.device_name = 'CPU'

        if _HAS_DML and 'DmlExecutionProvider' not in active_providers:
            print('  ⚠ GPU 降级到 CPU 推理 (DirectML 未激活)')
            print('    可能原因: GPU 驱动不兼容 / DirectX 12 不可用')

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.target_size = (data_cfg['img_width'], data_cfg['img_height'])
        self.frame_stack = data_cfg['frame_stack']
        self.grayscale = data_cfg.get('grayscale', True)
        self.key_names = data_cfg['keys']
        self.num_keys = len(self.key_names)
        self.crop_region_pc = data_cfg.get('crop_region_pc')
        self.crop_region_mobile = data_cfg.get('crop_region_mobile')
        self.frame_buffer = deque(maxlen=self.frame_stack)
        self.prev_key_states = np.zeros(self.num_keys, dtype=bool)
        self.key_action_map = self._build_key_action_map(inf_cfg.get('key_map', {}))
        self.hotkey = None
        if not no_act:
            self.hotkey = HotkeyController()
        self._hwnd = 0

    def _build_key_action_map(self, custom_map):
        default_layout = ['d', 'f', 'j', 'k', 's', 'l', 'space']
        result = {}
        for i, key_name in enumerate(self.key_names):
            if key_name in custom_map:
                result[key_name] = custom_map[key_name].lower()
            elif i < len(default_layout):
                result[key_name] = default_layout[i]
            else:
                result[key_name] = key_name.lower()
        return result

    def _find_window(self, exe_name='nikke.exe'):
        self._hwnd = find_hwnd_by_exe(exe_name)
        return self._hwnd

    def _bring_to_front(self):
        if self._hwnd:
            ctypes.windll.user32.SetForegroundWindow(self._hwnd)

    def _compute_capture_rect(self, x, y, w, h):
        # Apply the configured crop_region directly to the MSS monitor rect so
        # BitBlt only grabs the gameplay strip instead of the full window —
        # ~77% fewer pixels on PC, and cvtColor/resize downstream see less data.
        is_mobile = h > w
        crop_region = self.crop_region_mobile if is_mobile else self.crop_region_pc
        if not crop_region:
            return {'left': x, 'top': y, 'width': w, 'height': h}
        cx1 = int(crop_region[0] * w)
        cy1 = int(crop_region[1] * h)
        cx2 = int(crop_region[2] * w)
        cy2 = int(crop_region[3] * h)
        return {'left': x + cx1, 'top': y + cy1, 'width': cx2 - cx1, 'height': cy2 - cy1}

    def _preprocess_frame(self, frame_bgra):
        # Resize the contiguous BGRA buffer first, then drop the alpha channel
        # via cvtColor on the tiny 240x144 tensor — avoids a full-size BGRA→BGR
        # allocation per frame.
        resized = cv2.resize(frame_bgra, self.target_size, interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
            return frame[np.newaxis, :]
        frame = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        return frame.transpose(2, 0, 1)

    def _build_input_array(self):
        channels = 1 if self.grayscale else 3
        h, w = (self.target_size[1], self.target_size[0])
        while len(self.frame_buffer) < self.frame_stack:
            self.frame_buffer.appendleft(
                self.frame_buffer[0] if self.frame_buffer else np.zeros((channels, h, w), dtype=np.uint8)
            )
        stacked = np.concatenate(list(self.frame_buffer), axis=0)
        return (stacked.astype(np.float32) / 255.0)[np.newaxis, :]

    def _execute_keys(self, key_states):
        if self.no_act:
            return
        act_enabled = self.hotkey.act_enabled if self.hotkey else True
        if act_enabled:
            self._bring_to_front()
        has_press = False
        for i, (current, prev) in enumerate(zip(key_states, self.prev_key_states)):
            key_name = self.key_names[i]
            action_key = self.key_action_map.get(key_name, key_name)
            scan = SCANCODE_MAP.get(action_key)
            if scan is not None:
                if current and not prev:
                    if act_enabled:
                        press_key(scan)
                        has_press = True
                elif not current and prev:
                    release_key(scan)
        if has_press and self.key_delay > 0:
            time.sleep(self.key_delay)
        self.prev_key_states = key_states.copy()

    def _release_all_keys(self):
        for i in range(self.num_keys):
            if self.prev_key_states[i]:
                key_name = self.key_names[i]
                action_key = self.key_action_map.get(key_name, key_name)
                scan = SCANCODE_MAP.get(action_key)
                if scan is not None:
                    release_key(scan)
        self.prev_key_states[:] = False

    def run(self):
        if self.hotkey:
            self.hotkey.start()
        print('正在查找 NIKKE 窗口...')
        hwnd = self._find_window()
        if not hwnd:
            print('未找到 nikke.exe 进程，请先启动游戏！')
            return
        print('已锁定游戏窗口')
        print(f'推理设备: {self.device_name}')
        if not self.no_act:
            print('按键模式: 已启用 (F1=暂停/恢复, F2=退出)')
        else:
            print('按键模式: 已禁用 (仅预览)')

        from mss import MSS as mss_cls
        user32 = ctypes.windll.user32
        target_fps = CONFIG['inference']['capture_fps']
        interval = 1.0 / target_fps
        start_time = time.perf_counter()
        inference_count = 0

        try:
            with mss_cls() as sct:
                while not (self.hotkey and self.hotkey.stop_requested):
                    try:
                        if not user32.IsWindow(self._hwnd):
                            print('\n窗口已关闭，尝试重新查找...')
                            self._hwnd = 0
                            for _ in range(10):
                                hwnd = self._find_window()
                                if hwnd:
                                    break
                                time.sleep(2)
                            if not hwnd:
                                print('无法找到窗口，退出')
                                break
                            print(f'重新锁定窗口 (hwnd={hwnd})')

                        rect = get_window_rect(self._hwnd)
                        x, y, w, h = rect
                        if w <= 0 or h <= 0:
                            time.sleep(0.5)
                            continue

                        monitor = self._compute_capture_rect(x, y, w, h)
                        if monitor['width'] <= 0 or monitor['height'] <= 0:
                            time.sleep(0.5)
                            continue

                        t0 = time.perf_counter()
                        screenshot = np.array(sct.grab(monitor))
                        processed = self._preprocess_frame(screenshot)
                        self.frame_buffer.append(processed)

                        if len(self.frame_buffer) >= self.frame_stack:
                            input_array = self._build_input_array()
                            logits = self.session.run([self.output_name], {self.input_name: input_array})[0]
                            probs = 1.0 / (1.0 + np.exp(-logits[0]))
                            key_states = probs >= self.threshold
                            inference_count += 1
                            self._execute_keys(key_states)

                            elapsed = time.perf_counter() - start_time
                            if inference_count % max(1, target_fps) == 0:
                                fps = inference_count / elapsed if elapsed > 0 else 0
                                act_str = 'ON' if (not self.hotkey or self.hotkey.act_enabled) else 'OFF'
                                keys_str = ' '.join(
                                    f'[{self.key_action_map[self.key_names[i]].upper()}]' if key_states[i]
                                    else f' {self.key_action_map[self.key_names[i]].upper()} '
                                    for i in range(self.num_keys)
                                )
                                print(f'\r{fps:.0f}FPS | {keys_str} | ACT:{act_str}', end='', flush=True)

                            if self.show:
                                vis = np.ascontiguousarray(screenshot[..., :3])
                                bar_h = 30
                                overlay = vis.copy()
                                cv2.rectangle(overlay, (0, 0), (vis.shape[1], bar_h), (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
                                cell_w = vis.shape[1] // self.num_keys
                                for i in range(self.num_keys):
                                    label = self.key_action_map[self.key_names[i]].upper()
                                    color = (0, 220, 80) if key_states[i] else (120, 120, 120)
                                    prob = probs[i]
                                    txt = f'{label}:{prob:.2f}'
                                    cx = i * cell_w + cell_w // 2
                                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                                    cv2.putText(vis, txt, (cx - tw // 2, bar_h - 8),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                                cv2.imshow('RhythmAI', vis)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

                        elapsed_frame = time.perf_counter() - t0
                        sleep = interval - elapsed_frame
                        if sleep > 0:
                            time.sleep(sleep)
                    except Exception:
                        self._hwnd = 0
                        time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            if not self.no_act:
                self._release_all_keys()
            if self.hotkey:
                self.hotkey.stop()
            if self.show:
                cv2.destroyAllWindows()
            elapsed = time.perf_counter() - start_time
            avg_fps = inference_count / elapsed if elapsed > 0 else 0
            print(f'\n已停止 | {inference_count}帧 {avg_fps:.0f}FPS {elapsed:.1f}s')


def _print_startup_banner():
    """打印启动横幅"""
    frames = ['( >_< )', '( ≧▽≦ )', '( ★ω★ )', '( ♪♬♪ )', '( ᐛ )و']
    import random
    pick = random.choice(frames)
    print()
    print('  ╔══════════════════════════════════════╗')
    print('  ║       NIKKE Rhythm AI  v1.0         ║')
    print('  ╚══════════════════════════════════════╝')
    print(f'  {pick}')
    print()
    print('  提示: 如启动失败，请清空 C 盘 Temp 缓存后重试:')
    print('    Win+R → 输入 %temp% → 全选删除')
    print()


def main():
    parser = argparse.ArgumentParser(description='NIKKE 音游AI')
    parser.add_argument('--model', type=str, default=None, help='ONNX 模型路径 (默认: 同目录下 best_model.onnx)')
    parser.add_argument('--threshold', type=float, default=0.5, help='按键置信度阈值 (默认: 0.5)')
    parser.add_argument('--no-act', action='store_true', help='不执行按键，仅显示预测')
    parser.add_argument('--show', action='store_true', help='弹窗显示实时画面')
    parser.add_argument('--fps', type=int, default=60, help='捕获帧率 (默认: 60)')
    parser.add_argument('--key-delay', type=float, default=0.02, help='按键延时(秒) (默认: 0.02)')
    args = parser.parse_args()

    if args.model:
        model_path = args.model
    else:
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'best_model.onnx')

    if not os.path.exists(model_path):
        print(f'模型文件不存在: {model_path}')
        print('请将 best_model.onnx 放到脚本同目录下，或用 --model 指定路径')
        print('提示: 使用 export_onnx.py 可从 .pt 导出 .onnx')
        sys.exit(1)

    CONFIG['inference']['capture_fps'] = args.fps
    _print_startup_banner()
    ai = RhythmAI(
        model_path=model_path,
        threshold=args.threshold,
        no_act=args.no_act,
        show=args.show,
        key_delay=args.key_delay,
    )
    ai.run()


if __name__ == '__main__':
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetThreadExecutionState(2147483649)
    main()
