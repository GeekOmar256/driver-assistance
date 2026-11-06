"""Optional CSI camera adapter for the Flask frame generator."""
class PiCameraCapture:
    def __init__(self):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        try:
            configuration = self.camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"})
            self.camera.configure(configuration)
            self.camera.start()
        except Exception:
            self.camera.close()
            raise

    def read(self):
        # Picamera2 RGB888 arrays use BGR byte order for OpenCV.
        return True, self.camera.capture_array("main")

    def release(self):
        try:
            self.camera.stop()
        finally:
            self.camera.close()
