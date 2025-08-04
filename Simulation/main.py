import tkinter as tk
from tkinter import filedialog
import threading
import time
from pathlib import Path
import sys
import traceback
import logging
from logging.handlers import RotatingFileHandler
import queue

import torch
from torchvision import transforms
from PIL import Image
import mss
from pynput.keyboard import Controller, Key, KeyCode

# --- Logger setup ---
logger = logging.getLogger("jumpnet_infer")
logger.setLevel(logging.DEBUG)
file_handler = RotatingFileHandler("jumpnet_infer.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(threadName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
logger.addHandler(console_handler)

# --- Model import / fallback ---
JumpNet = None
try:
    from model import JumpNet
    logger.info("JumpNet imported: src.model.JumpNet")
except Exception:
    try:
        from model import JumpNet
        logger.info("JumpNet imported: model.JumpNet")
    except Exception as e:
        logger.warning("JumpNet could not be imported, fallback will be used: %s", e)
        traceback.print_exc()
        JumpNet = None

if JumpNet is None:
    import torch.nn as nn
    from torchvision import models
    from torchvision.models import MobileNet_V2_Weights

    class JumpNet(nn.Module):
        """
        MobileNetV2-based network for jump classification and hold duration regression.
        Used as a fallback if JumpNet cannot be imported.
        """
        def __init__(self):
            super(JumpNet, self).__init__()
            weights = MobileNet_V2_Weights.DEFAULT
            base_model = models.mobilenet_v2(weights=weights)
            self.backbone = base_model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(
                nn.Linear(1280, 512),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.jump_head = nn.Linear(512, 1)
            self.hold_head = nn.Linear(512, 1)

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            return torch.sigmoid(self.jump_head(x)), self.hold_head(x)

    logger.info("Fallback JumpNet class is being used.")

# === Capture Thread ===
class CaptureThread(threading.Thread):
    """
    Thread for capturing screen region, running inference and pressing key according to the model output.
    """
    def __init__(self, model, device, region, key_to_press="space", threshold=0.5,
                 interval_ms=30, hold_scale=1.0, debounce_ms=120, apply_extra_sigmoid=False,
                 resize_hw=(227, 227), status_callback=None, fps_callback=None):
        super().__init__(daemon=True)
        self.model = model.eval().to(device)
        self.device = device
        self.region = region
        self.key_to_press = key_to_press
        self.threshold = threshold
        self.interval = max(1, interval_ms) / 1000.0
        self.hold_scale = hold_scale
        self.debounce = debounce_ms / 1000.0
        self.apply_extra_sigmoid = apply_extra_sigmoid
        self.resize_hw = resize_hw
        self.status_callback = status_callback or (lambda s: None)
        self.fps_callback = fps_callback or (lambda f: None)

        self._stop_event = threading.Event()
        self.keyboard = Controller()
        self.last_trigger = 0.0
        self._pressing_lock = threading.Lock()

        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_hw),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.key_obj = self._map_key(self.key_to_press)

    def _map_key(self, name):
        """
        Maps key name string to pynput Key or KeyCode.
        """
        name = str(name).lower()
        special = {
            "space": Key.space,
            "enter": Key.enter,
            "shift": Key.shift,
            "ctrl": Key.ctrl,
            "alt": Key.alt,
            "tab": Key.tab,
            "esc": Key.esc,
            "up": Key.up,
            "down": Key.down,
            "left": Key.left,
            "right": Key.right,
            "backspace": Key.backspace,
            "delete": Key.delete,
            "home": Key.home,
            "end": Key.end,
            "pageup": Key.page_up,
            "pagedown": Key.page_down,
        }
        if name in special:
            return special[name]
        if len(name) == 1:
            return KeyCode.from_char(name)
        return KeyCode.from_char(name[0])

    def stop(self):
        self._stop_event.set()

    def _press_key(self, duration):
        """
        Presses the mapped key for given duration in seconds.
        """
        if not self._pressing_lock.acquire(blocking=False):
            logger.debug("Key press lock is busy, skipped.")
            return
        try:
            key = self.key_obj
            logger.info("Pressing key: %s, planned duration: %.3f s", key, duration)
            start = time.perf_counter()
            self.keyboard.press(key)
            time.sleep(max(0.0, duration))
            self.keyboard.release(key)
            end = time.perf_counter()
            actual = end - start
            logger.info("Key released: %s, planned duration=%.3f s, actual press duration=%.3f s", key, duration, actual)
            self.status_callback(f"Pressed: {key}, planned={duration:.3f}s, actual={actual:.3f}s")
        finally:
            self._pressing_lock.release()

    def run(self):
        logger.debug("Run started (thread alive).")
        monitor = {
            "left": int(self.region[0]),
            "top": int(self.region[1]),
            "width": int(self.region[2]),
            "height": int(self.region[3]),
        }

        if monitor["width"] <= 0 or monitor["height"] <= 0:
            self.status_callback("Invalid region size.")
            logger.error("Invalid region: %s", monitor)
            return

        frame_count = 0
        last_fps_time = time.perf_counter()
        self.status_callback("Worker started.")
        logger.info("Capture thread started. Region: %s, interval=%.1fms, hold_scale=%.2f, debounce=%.1fms",
                    monitor, self.interval * 1000, self.hold_scale, self.debounce * 1000)

        try:
            with mss.mss() as sct:
                while not self._stop_event.is_set():
                    loop_start = time.perf_counter()
                    try:
                        shot = sct.grab(monitor)
                    except Exception as e:
                        self.status_callback(f"Screen capture error: {e}")
                        logger.warning("Screen capture error: %s", e)
                        time.sleep(0.1)
                        continue

                    try:
                        img = Image.frombytes("RGB", shot.size, shot.rgb)
                        tensor = self.tf(img).unsqueeze(0).to(self.device)
                    except Exception as e:
                        self.status_callback(f"Preprocessing error: {e}")
                        logger.warning("Preprocessing error: %s", e)
                        time.sleep(0.05)
                        continue

                    try:
                        with torch.inference_mode():
                            out = self.model(tensor)
                        logger.debug("Model raw output: %s", out)
                    except Exception as e:
                        self.status_callback(f"Model inference error: {e}")
                        logger.error("Model inference error: %s", e)
                        time.sleep(0.1)
                        continue

                    # Model output: jump_pred, hold_pred
                    if isinstance(out, (list, tuple)) and len(out) >= 2:
                        jump_pred, hold_pred = out[0], out[1]
                    else:
                        self.status_callback("Model output format is different than expected.")
                        logger.error("Expected (jump, hold) output not found, out=%s", out)
                        break

                    try:
                        jump_prob = jump_pred.view(-1)[0].item()
                        hold_time = max(0.0, hold_pred.view(-1)[0].item()) * self.hold_scale
                    except Exception as e:
                        self.status_callback(f"Output processing error: {e}")
                        logger.warning("Output processing error: %s", e)
                        time.sleep(0.05)
                        continue

                    now = time.perf_counter()

                    # ---- DYNAMIC THRESHOLD USED HERE! ----
                    if jump_prob > self.threshold and (now - self.last_trigger) >= self.debounce:
                        self.last_trigger = now
                        threading.Thread(target=self._press_key, args=(hold_time,), daemon=True).start()
                        msg = f"Triggered: prob={jump_prob:.3f}, hold={hold_time:.3f}s, threshold={self.threshold:.2f}"
                        self.status_callback(msg)
                        logger.info(msg)

                    frame_count += 1
                    if now - last_fps_time >= 1.0:
                        fps = frame_count / (now - last_fps_time)
                        self.fps_callback(fps)
                        frame_count = 0
                        last_fps_time = now

                    elapsed = time.perf_counter() - loop_start
                    if elapsed < self.interval:
                        time.sleep(self.interval - elapsed)
        except Exception as e:
            self.status_callback(f"Unexpected error: {e}")
            logger.exception("Thread ended due to unexpected error.")
        finally:
            self.status_callback("Worker finished.")
            logger.info("Capture thread naturally ended.")

# === GUI ===
class App:
    """
    Main Tkinter GUI class for controlling model inference and screen region capture.
    """
    def __init__(self, root):
        self.root = root
        root.title("JumpNet Real-Time Tkinter")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.capture_thread = None
        self.region = None

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=f"Device: {self.device}").grid(row=0, column=0, columnspan=2, sticky="w")
        self.model_label = tk.Label(frame, text="Model: (not loaded)")
        self.model_label.grid(row=1, column=0, columnspan=2, sticky="w")
        self.region_label = tk.Label(frame, text="Region: (not selected)")
        self.region_label.grid(row=2, column=0, columnspan=2, sticky="w")

        self.load_btn = tk.Button(frame, text="Load Model (.pt/.pth)", command=self.load_model)
        self.load_btn.grid(row=3, column=0, pady=4, sticky="ew")
        self.snip_btn = tk.Button(frame, text="Select Region (Snip)", command=self.start_snip)
        self.snip_btn.grid(row=3, column=1, pady=4, sticky="ew")

        tk.Label(frame, text="Threshold:").grid(row=4, column=0, sticky="e")
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.threshold_entry = tk.Entry(frame, textvariable=self.threshold_var, state="normal")  # Enabled!
        self.threshold_entry.grid(row=4, column=1, sticky="w")

        tk.Label(frame, text="Interval (ms):").grid(row=5, column=0, sticky="e")
        self.interval_var = tk.IntVar(value=30)
        self.interval_entry = tk.Entry(frame, textvariable=self.interval_var)
        self.interval_entry.grid(row=5, column=1, sticky="w")

        tk.Label(frame, text="Hold multiplier:").grid(row=6, column=0, sticky="e")
        self.hold_scale_var = tk.DoubleVar(value=1.0)
        self.hold_scale_entry = tk.Entry(frame, textvariable=self.hold_scale_var)
        self.hold_scale_entry.grid(row=6, column=1, sticky="w")

        tk.Label(frame, text="Debounce (ms):").grid(row=7, column=0, sticky="e")
        self.debounce_var = tk.IntVar(value=120)
        self.debounce_entry = tk.Entry(frame, textvariable=self.debounce_var)
        self.debounce_entry.grid(row=7, column=1, sticky="w")

        tk.Label(frame, text="Key to press:").grid(row=8, column=0, sticky="e")
        self.key_entry = tk.Entry(frame)
        self.key_entry.insert(0, "w")
        self.key_entry.grid(row=8, column=1, sticky="w")

        self.extra_sigmoid_var = tk.BooleanVar(value=False)
        self.sigmoid_chk = tk.Checkbutton(frame, text="Apply extra sigmoid (usually not needed)", variable=self.extra_sigmoid_var)
        self.sigmoid_chk.grid(row=9, column=0, columnspan=2, sticky="w")

        self.start_btn = tk.Button(frame, text="Start", command=self.start_capture, state="disabled")
        self.start_btn.grid(row=10, column=0, pady=6, sticky="ew")
        self.stop_btn = tk.Button(frame, text="Stop", command=self.stop_capture, state="disabled")
        self.stop_btn.grid(row=10, column=1, pady=6, sticky="ew")

        self.status_label = tk.Label(frame, text="Status: -")
        self.status_label.grid(row=11, column=0, columnspan=2, sticky="w")
        self.fps_label = tk.Label(frame, text="FPS: -")
        self.fps_label.grid(row=12, column=0, columnspan=2, sticky="w")

        # Live log box and queue
        self.log_box = tk.Text(frame, height=8, state="disabled", wrap="none")
        self.log_box.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.log_queue = queue.Queue()
        self.root.after(100, self._drain_logs)

        gui_handler = GuiHandler(lambda m: self._enqueue_log(m))
        gui_handler.setFormatter(fmt)
        logger.addHandler(gui_handler)

        root.bind("<Control-q>", lambda e: root.quit())

    def _enqueue_log(self, msg):
        self.log_queue.put(msg)

    def _drain_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.config(state="normal")
                self.log_box.insert("end", msg + "\n")
                self.log_box.see("end")
                self.log_box.config(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_logs)

    def load_model(self):
        path = filedialog.askopenfilename(title="Select model (.pt/.pth)", filetypes=[("PyTorch", "*.pt *.pth"), ("All", "*.*")])
        if not path:
            return
        ok, msg = self._load_model(Path(path))
        self.model_label.config(text=f"Model: {msg}")
        self._refresh_start_enable()
        logger.info("Model loaded: %s (%s)", path, msg)

    def _load_model(self, path: Path):
        try:
            try:
                model = torch.jit.load(str(path), map_location=self.device)
                self.model = model.to(self.device)
                return True, f"TorchScript loaded ({path.name})"
            except Exception:
                model = JumpNet()
                state = torch.load(str(path), map_location="cpu")
                if isinstance(state, dict):
                    model.load_state_dict(state, strict=False)
                else:
                    model = state
                self.model = model.to(self.device)
                return True, f"State dict / model loaded ({path.name})"
        except Exception as e:
            logger.error("Model load error: %s", e)
            return False, f"Load error: {e}"

    def start_snip(self):
        """
        Starts a fullscreen snipping window to select a region for screen capture.
        """
        self.snip_win = tk.Toplevel(self.root)
        self.snip_win.attributes("-fullscreen", True)
        self.snip_win.attributes("-topmost", True)
        self.snip_win.config(bg="black")
        self.snip_win.wm_attributes("-alpha", 0.25)

        canvas = tk.Canvas(self.snip_win, cursor="cross", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        start = {"x": None, "y": None}
        rect_id = None

        def on_press(event):
            start["x"], start["y"] = event.x_root, event.y_root

        def on_move(event):
            nonlocal rect_id
            if start["x"] is None:
                return
            if rect_id:
                canvas.delete(rect_id)
            x0, y0 = start["x"], start["y"]
            x1, y1 = event.x_root, event.y_root
            canvas_x0 = x0 - self.snip_win.winfo_x()
            canvas_y0 = y0 - self.snip_win.winfo_y()
            canvas_x1 = x1 - self.snip_win.winfo_x()
            canvas_y1 = y1 - self.snip_win.winfo_y()
            rect_id = canvas.create_rectangle(canvas_x0, canvas_y0, canvas_x1, canvas_y1, outline="red", width=2)

        def on_release(event):
            if start["x"] is None:
                return
            x0, y0 = start["x"], start["y"]
            x1, y1 = event.x_root, event.y_root
            self.snip_win.destroy()
            left = min(x0, x1)
            top = min(y0, y1)
            width = abs(x1 - x0)
            height = abs(y1 - y0)
            self.region = (left, top, width, height)
            self.region_label.config(text=f"Region: left={left}, top={top}, w={width}, h={height}")
            self._refresh_start_enable()
            logger.info("Region selected: %s", self.region)

        canvas.bind("<Button-1>", on_press)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_release)
        self.snip_win.bind("<Escape>", lambda e: self.snip_win.destroy())

    def _refresh_start_enable(self):
        if getattr(self, "model", None) is not None and self.region is not None:
            self.start_btn.config(state="normal")
        else:
            self.start_btn.config(state="disabled")

    def start_capture(self):
        """
        Starts the capture thread if both model and region are set.
        """
        if getattr(self, "model", None) is None:
            self._set_status("No model.")
            logger.warning("start_capture: no model")
            return
        if self.region is None:
            self._set_status("No region selected.")
            logger.warning("start_capture: no region selected")
            return

        try:
            interval = int(self.interval_var.get())
            hold_scale = float(self.hold_scale_var.get())
            debounce = int(self.debounce_var.get())
            threshold = float(self.threshold_var.get())  # Threshold is taken here!
        except Exception:
            self._set_status("Parameter value error.")
            logger.warning("start_capture: parameter parse error")
            return
        key_to_press = self.key_entry.get().strip() or "space"

        self.capture_thread = CaptureThread(
            model=self.model,
            device=self.device,
            region=self.region,
            key_to_press=key_to_press,
            threshold=threshold,   # DYNAMIC threshold!
            interval_ms=interval,
            hold_scale=hold_scale,
            debounce_ms=debounce,
            apply_extra_sigmoid=self.extra_sigmoid_var.get(),
            resize_hw=(227, 227),
            status_callback=lambda s: self.root.after(0, lambda: self._set_status(s)),
            fps_callback=lambda f: self.root.after(0, lambda: self._set_fps(f)),
        )
        self.capture_thread.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self._set_status("Running...")
        logger.info("Capture started.")

    def stop_capture(self):
        if self.capture_thread:
            self.capture_thread.stop()
            self.capture_thread = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_status("Stopped.")
        logger.info("Capture stopped.")

    def _set_status(self, text):
        self.status_label.config(text=f"Status: {text}")
        logger.debug("GUI status: %s", text)

    def _set_fps(self, val):
        self.fps_label.config(text=f"FPS: {float(val):.1f}")
        logger.debug("FPS: %.1f", float(val))

# GUI log handler
class GuiHandler(logging.Handler):
    def __init__(self, write_func):
        super().__init__()
        self.write = write_func

    def emit(self, record):
        try:
            msg = self.format(record)
            self.write(msg)
        except Exception:
            pass

def main():
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == "__main__":
    main()
