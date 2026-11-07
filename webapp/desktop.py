"""Local, three-tab VisionGuard interface. Run on the Pi graphical desktop."""
import os
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import cv2
from PIL import Image, ImageTk
import config
from alerts import RecentDetections, build_alerts
from detector import SignDetector
from hybrid_detector import HybridDetector


def make_detector(model_id):
    option = config.MODEL_OPTIONS[model_id]
    for key in ("path", "pretrained_path"):
        if key in option and not os.path.isfile(option[key]):
            raise FileNotFoundError(option[key])
    if option.get("pipeline") == "hybrid":
        return HybridDetector(option["pretrained_path"], option["path"],
                              color_fallback=option["color_fallback"])
    return SignDetector(option["path"], scope=option["scope"])


def open_source(kind, value):
    if kind == "camera" and value == "picamera2":
        from pi_camera import PiCameraCapture
        return PiCameraCapture()
    source = int(value) if kind == "camera" and value.isdigit() else value
    capture = (cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
               if kind == "stream" else cv2.VideoCapture(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Cannot open the selected video or camera.")
    return capture


class DetectionWorker(threading.Thread):
    """One owner for model/capture; Tk is touched only by the main thread."""
    def __init__(self):
        super().__init__(daemon=True)
        self.commands = queue.Queue()
        self.events = queue.Queue()
        self.frames = queue.Queue(maxsize=1)
        self.quit = threading.Event()

    def publish(self, item):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(item)

    def run(self):
        detector = capture = None
        selected = []
        confidence, size = .4, 640
        paused = False
        next_frame = 0
        interval = 0
        recent = RecentDetections()
        try:
            while not self.quit.is_set():
                try:
                    command, data = self.commands.get(timeout=.02 if capture else .1)
                except queue.Empty:
                    command, data = None, None
                try:
                    if command in {"model", "stop", "start"}:
                        if capture:
                            capture.release()
                            capture = None
                        recent.clear()
                        self.publish((None, []))
                        self.events.put(("status", "Stopped"))
                    if command == "model":
                        detector = None
                        detector = make_detector(data)
                        selected = detector.class_names()
                        self.events.put(("classes", selected))
                        self.events.put(("status", "Ready"))
                    elif command == "settings":
                        selected, confidence, size, hold = data
                        recent.hold_seconds = hold
                    elif command == "start":
                        if detector is None:
                            raise RuntimeError("Select and load a model first.")
                        capture = open_source(*data)
                        fps = capture.get(cv2.CAP_PROP_FPS) if data[0] == "video" else 0
                        interval = 1 / fps if 0 < fps <= 120 else 0
                        next_frame = time.monotonic()
                        paused = False
                        self.events.put(("status", "Running"))
                    elif command == "pause" and capture:
                        paused = not paused
                        self.events.put(("status", "Paused" if paused else "Running"))
                    if capture and not paused and time.monotonic() >= next_frame:
                        ok, frame = capture.read()
                        if not ok:
                            capture.release()
                            capture = None
                            recent.clear()
                            self.publish((None, []))
                            self.events.put(("status", "Source ended or disconnected"))
                            continue
                        frame, detections = detector.detect_and_annotate(
                            frame, target_classes=selected, confidence=confidence, image_size=size)
                        recent.update(detections)
                        self.publish((frame, build_alerts(recent.snapshot())))
                        next_frame = max(next_frame + interval, time.monotonic())
                except Exception as error:
                    if capture:
                        capture.release()
                        capture = None
                    recent.clear()
                    self.publish((None, []))
                    self.events.put(("error", str(error)))
        finally:
            if capture:
                capture.release()


class VisionGuard:
    def __init__(self, root):
        self.root = root
        root.title("VisionGuard — Driver Assistance")
        root.geometry("1024x700")
        root.minsize(640, 480)
        self.worker = DetectionWorker()
        self.worker.start()
        self.status = tk.StringVar(value="Loading default two-stage model…")
        self.sound = tk.BooleanVar(value=True)
        self.fullscreen = tk.BooleanVar(value=False)
        self.alert_size = tk.IntVar(value=42)
        self.confidence = tk.DoubleVar(value=40)
        self.hold = tk.DoubleVar(value=3)
        self.resolution = tk.StringVar(value="640")
        self.source_kind = tk.StringVar(value="camera")
        self.source = tk.StringVar(value=config.CAMERA_SOURCE)
        self.model = tk.StringVar(value=config.MODEL_OPTIONS[config.DEFAULT_MODEL]["label"])
        self.class_vars = {}
        self.audio_process = None
        self.last_sound = {}
        self.current_alerts = []
        self.last_frame = None
        self.photo = None
        self.closed = False
        style = ttk.Style()
        style.configure("TNotebook.Tab", font=("DejaVu Sans", 15), padding=(18, 10))
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)
        self.alert_tab, self.video_tab, self.settings_tab = (ttk.Frame(self.tabs) for _ in range(3))
        for page, title in ((self.alert_tab, "Driver alerts"), (self.video_tab, "Video / Live"),
                            (self.settings_tab, "Settings")):
            self.tabs.add(page, text=title)
        self.alert_tab.rowconfigure(0, weight=1)
        self.alert_tab.columnconfigure(0, weight=1)
        self.alert_label = tk.Label(self.alert_tab, text="DETECTION STOPPED", bg="#101820",
                                    fg="white", font=("DejaVu Sans", 42, "bold"), justify="center")
        self.alert_label.grid(row=0, column=0, sticky="nsew")
        self.alert_label.bind("<Configure>", lambda e: self.alert_label.configure(wraplength=max(100,e.width-32)))
        self.video_label = tk.Label(self.video_tab, text="Select a source in Settings, then Start", bg="#101820", fg="white")
        self.video_label.pack(fill="both", expand=True)
        self.video_label.bind("<Configure>", lambda e: self.draw_frame())
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self.draw_frame())
        self.build_settings()
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Start", command=self.start, state="disabled")
        self.start_button.pack(side="left", padx=4)
        ttk.Button(controls, text="Pause / Resume", command=lambda: self.worker.commands.put(("pause", None))).pack(side="left", padx=4)
        ttk.Button(controls, text="Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Label(controls, textvariable=self.status, wraplength=420).pack(side="left", padx=12)
        root.bind("<Escape>", lambda e: self.set_fullscreen(False))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.change_model()
        root.after(50, self.poll)

    def build_settings(self):
        canvas = tk.Canvas(self.settings_tab, highlightthickness=0)
        scroll = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(canvas, padding=18)
        window = canvas.create_window((0,0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text="Detection method").grid(row=0, column=0, sticky="w")
        models = ttk.Combobox(panel, state="readonly", textvariable=self.model,
                             values=[o["label"] for o in config.MODEL_OPTIONS.values()])
        self.model_selector = models
        models.grid(row=0, column=1, sticky="ew", pady=8)
        models.bind("<<ComboboxSelected>>", lambda e: self.change_model())
        ttk.Label(panel, text="Confidence (%)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(panel, from_=1, to=100, textvariable=self.confidence).grid(row=1, column=1, sticky="ew")
        ttk.Label(panel, text="Inference size").grid(row=2, column=0, sticky="w")
        ttk.Combobox(panel, state="readonly", textvariable=self.resolution, values=["640","960","1280"]).grid(row=2, column=1, sticky="ew", pady=8)
        ttk.Label(panel, text="Input type").grid(row=3, column=0, sticky="w")
        ttk.Combobox(panel, state="readonly", textvariable=self.source_kind, values=["camera", "video", "stream"]).grid(row=3, column=1, sticky="ew")
        ttk.Label(panel, text="Camera / file / stream URL").grid(row=4, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.source).grid(row=4, column=1, sticky="ew", pady=8)
        ttk.Button(panel, text="Browse video…", command=self.browse).grid(row=5, column=1, sticky="w")
        ttk.Label(panel, text="Camera: 0, /dev/video2 or picamera2. Stream: RTSP / HTTP URL.", wraplength=540).grid(row=6,column=0,columnspan=2,sticky="w",pady=8)
        ttk.Label(panel, text="Alert text size").grid(row=7,column=0,sticky="w")
        ttk.Spinbox(panel, from_=24, to=100, textvariable=self.alert_size).grid(row=7,column=1,sticky="ew")
        ttk.Label(panel, text="Hold brief alerts (seconds)").grid(row=8,column=0,sticky="w",pady=8)
        ttk.Spinbox(panel, from_=1,to=10,increment=.5,textvariable=self.hold).grid(row=8,column=1,sticky="ew")
        ttk.Checkbutton(panel, text="Directional sound alerts", variable=self.sound, command=self.sound_changed).grid(row=9,column=0,columnspan=2,sticky="w")
        ttk.Checkbutton(panel, text="Fullscreen (Escape to exit)", variable=self.fullscreen, command=lambda:self.set_fullscreen(self.fullscreen.get())).grid(row=10,column=0,columnspan=2,sticky="w",pady=8)
        self.class_panel = ttk.LabelFrame(panel, text="Enabled classes — updates with selected model", padding=8)
        self.class_panel.grid(row=11,column=0,columnspan=2,sticky="ew")
        ttk.Button(panel,text="Apply settings",command=self.apply).grid(row=12,column=0,columnspan=2,pady=12)
        ttk.Label(panel,text="Changing model stops playback. Apply source changes by stopping and starting. Direction is image position; it does not estimate distance.",wraplength=560).grid(row=13,column=0,columnspan=2,sticky="w")

    def browse(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*")])
        if path:
            self.source_kind.set("video")
            self.source.set(path)

    def change_model(self):
        self.stop()
        self.start_button.configure(state="disabled")
        self.model_selector.configure(state="disabled")
        self.class_vars = {}
        for child in self.class_panel.winfo_children():
            child.destroy()
        self.status.set("Loading model…")
        key = next(k for k,v in config.MODEL_OPTIONS.items() if v["label"] == self.model.get())
        self.worker.commands.put(("model",key))

    def apply(self):
        try:
            confidence, size, hold = self.confidence.get()/100, int(self.resolution.get()), self.hold.get()
            font_size = self.alert_size.get()
            if not .01 <= confidence <= 1 or size not in {640,960,1280} or not 1 <= hold <= 10 or not 24 <= font_size <= 100:
                raise ValueError()
        except (ValueError, tk.TclError):
            messagebox.showerror("Settings", "Use confidence 1–100%, size 640/960/1280, hold 1–10 seconds and text size 24–100.")
            return False
        selected = [name for name,var in self.class_vars.items() if var.get()]
        self.worker.commands.put(("settings",(selected,confidence,size,hold)))
        self.alert_label.configure(font=("DejaVu Sans",font_size,"bold"))
        self.render_alerts()
        return True

    def start(self):
        if self.apply():
            if not self.source.get().strip():
                messagebox.showerror("Input", "Choose a camera, video or stream URL.")
                return
            self.last_sound.clear()
            self.worker.commands.put(("start",(self.source_kind.get(),self.source.get().strip())))

    def stop(self):
        self.worker.commands.put(("stop",None))
        self.current_alerts = []
        self.last_frame = None
        self.stop_audio()
        self.render_alerts()

    def set_fullscreen(self, enabled):
        self.fullscreen.set(enabled)
        self.root.attributes("-fullscreen",enabled)

    def stop_audio(self):
        if self.audio_process and self.audio_process.poll() is None:
            self.audio_process.terminate()
        self.audio_process = None

    def sound_changed(self):
        if not self.sound.get():
            self.stop_audio()

    def play_alert(self):
        if not self.sound.get() or (self.audio_process and self.audio_process.poll() is None):
            return
        now = time.monotonic()
        for alert in self.current_alerts:
            key = (alert["class_name"],alert["direction"])
            if alert["recent"] or not alert["audio_url"] or now-self.last_sound.get(key,-100) < 6:
                continue
            path = Path(config.BASE_DIR) / alert["audio_url"].lstrip("/")
            if path.is_file() and shutil.which("ffplay"):
                self.audio_process = subprocess.Popen(["ffplay","-nodisp","-autoexit","-loglevel","error",str(path)], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                self.last_sound[key] = now
            break

    def render_alerts(self):
        active = self.current_alerts
        text = "\n\n".join(a["text"].upper() + ("\nRecently detected" if a["recent"] else "") for a in active[:3])
        if len(active)>3:
            text += f"\n+ {len(active)-3} other alerts"
        lines = max(1, text.count("\n")+1)
        fitted_size = min(self.alert_size.get(), max(20, int(max(100,self.alert_label.winfo_height()-40)/(lines*1.7))))
        self.alert_label.configure(font=("DejaVu Sans",fitted_size,"bold"))
        danger = any(a["class_name"] in {"person","stop sign","red traffic light","no entry"} for a in active)
        self.alert_label.configure(text=text or ("NO CURRENT DETECTIONS" if self.status.get()=="Running" else self.status.get().upper()),
                                   bg="#941f28" if danger else "#101820",fg="white")

    def draw_frame(self):
        if self.last_frame is None:
            self.video_label.configure(image="",text="No active frames")
            self.photo = None
            return
        if self.tabs.index(self.tabs.select()) != 1:
            return
        image = Image.fromarray(cv2.cvtColor(self.last_frame,cv2.COLOR_BGR2RGB))
        image.thumbnail((max(1,self.video_label.winfo_width()),max(1,self.video_label.winfo_height())))
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo,text="")

    def poll(self):
        if self.closed:
            return
        while True:
            try:
                kind,data = self.worker.events.get_nowait()
            except queue.Empty:
                break
            if kind == "classes":
                self.model_selector.configure(state="readonly")
                for index,name in enumerate(data):
                    var = tk.BooleanVar(value=True)
                    self.class_vars[name] = var
                    ttk.Checkbutton(self.class_panel,text=name,variable=var).grid(row=index//2,column=index%2,sticky="w",padx=8)
                self.start_button.configure(state="normal")
                self.apply()
            else:
                self.status.set(("Error: " if kind=="error" else "")+data)
                if kind=="error":
                    self.model_selector.configure(state="readonly")
                if kind=="error" or data != "Running":
                    self.stop_audio()
                    self.current_alerts = []
        try:
            frame,alerts = self.worker.frames.get_nowait()
            self.last_frame,self.current_alerts = frame,alerts
            self.draw_frame()
        except queue.Empty:
            pass
        if self.status.get() != "Running":
            self.current_alerts = []
        self.render_alerts()
        self.play_alert()
        self.root.after(50,self.poll)

    def close(self):
        self.closed = True
        self.worker.quit.set()
        self.stop_audio()
        self.root.destroy()


if __name__ == "__main__":
    VisionGuard(tk.Tk()).root.mainloop()
