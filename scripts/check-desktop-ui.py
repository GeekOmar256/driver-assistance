"""Developer smoke check: real model, synthetic file, all tabs, no camera required."""
import sys, time, tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"webapp"))
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageGrab
from desktop import VisionGuard

root=tk.Tk()
ui=VisionGuard(root)
ui.sound.set(False)
def until(check, seconds=45):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        root.update()
        if check():
            return
        time.sleep(.03)
    raise RuntimeError("GUI timeout: "+ui.status.get())
try:
    until(lambda:ui.status.get()=="Ready")
    assert "person" in ui.class_vars
    assert "Two-stage" in ui.model.get()
    with tempfile.TemporaryDirectory() as folder:
        path=str(Path(folder)/"ui-test.avi")
        writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*"MJPG"),5,(320,240))
        for _ in range(30):
            writer.write(np.zeros((240,320,3),dtype=np.uint8))
        writer.release()
        ui.source_kind.set("video")
        ui.source.set(path)
        ui.start()
        until(lambda:ui.last_frame is not None)
        output=Path(tempfile.gettempdir())/"visionguard-tkinter-preview"
        output.mkdir(exist_ok=True)
        for index,name in enumerate(("alerts","video","settings")):
            ui.tabs.select(index)
            root.update()
            if index==0:
                # Presentation fixture only: actual detection tested separately.
                ui.current_alerts=[{"text":"Person ahead","recent":False,"class_name":"person"},
                                   {"text":"Red traffic light on your left","recent":False,"class_name":"red traffic light"}]
                ui.render_alerts()
            root.update()
            ImageGrab.grab(bbox=(root.winfo_rootx(),root.winfo_rooty(),root.winfo_rootx()+root.winfo_width(),root.winfo_rooty()+root.winfo_height())).save(output/(name+".png"))
        ui.stop()
        until(lambda:ui.status.get()=="Stopped")
        print("GUI default model, class list, real CPU inference, video frames and three tabs passed")
        print(output)
finally:
    ui.close()
    ui.worker.join(timeout=5)
