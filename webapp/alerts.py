import config
import time
from audio_assets import PHRASES, filename, DIRECTION_PHRASES

DIRECTION_PHRASE = DIRECTION_PHRASES

# Shown first, since they call for a driver response rather than just awareness.
_PRIORITY = {"person": 0, "stop sign": 0, "red traffic light": 0, "no entry": 0,
             "yellow traffic light": 1, "yield sign": 1, "traffic light": 1}


class RecentDetections:
    """Hold brief frame detections long enough for the driver/UI to see them.

    Call under the stream state lock. Store the latest frame's group, not a
    growing list of objects accumulated across frames.
    """
    def __init__(self, hold_seconds=3.0, clock=time.monotonic):
        self.hold_seconds = hold_seconds
        self.clock = clock
        self.clear()

    def clear(self):
        self.groups = {}
        self.current_keys = set()

    def update(self, detections):
        now = self.clock()
        groups = {}
        for item in detections:
            key = (item["class_name"], item["direction"])
            groups.setdefault(key, []).append(dict(item))
        self.current_keys = set(groups)
        colors = {"red traffic light", "yellow traffic light", "green traffic light"}
        for key in list(self.groups):
            name, direction = key
            # Replace an old color when the new frame shows a changed signal.
            if name in colors and key not in groups and any(
                c in colors and d == direction for c, d in groups
            ):
                del self.groups[key]
        for key, items in groups.items():
            self.groups[key] = (now, items)
        self.snapshot()

    def snapshot(self):
        now = self.clock()
        result = []
        for key, (seen, items) in list(self.groups.items()):
            if now - seen >= self.hold_seconds:
                del self.groups[key]
                continue
            result.extend(dict(item, recent=key not in self.current_keys) for item in items)
        return result

    def finish(self):
        self.current_keys.clear()


def build_alerts(detections):
    """Turn raw per-frame detections into grouped, friendly alert messages.

    Detections sharing a class and direction are collapsed into one entry
    (e.g. two cars ahead -> "2 Cars ahead") so the panel doesn't flood with
    near-duplicate lines frame to frame.
    """
    groups = {}
    for d in detections:
        key = (d["class_name"], d["direction"])
        groups.setdefault(key, []).append(d)

    alerts = []
    for (class_name, direction), items in groups.items():
        display = config.CLASS_DISPLAY.get(class_name, {
            "icon": "ℹ️",
            "label": class_name.title(),
            "plural": class_name.title() + "s",
            "kind": "vehicle",
        })
        count = len(items)
        label = display["plural"] if count > 1 else display["label"]
        prefix = f"{count} " if count > 1 else ""
        phrase = DIRECTION_PHRASE.get(direction, "")
        text = f"{prefix}{label} {phrase}".strip()
        alerts.append({
            "icon": display["icon"],
            "text": text,
            "kind": display["kind"],
            "class_name": class_name,
            "direction": direction,
            "audio_url": "/static/audio/" + filename(class_name, direction) if class_name in PHRASES else None,
            "recent": all(d.get("recent", False) for d in items),
        })

    alerts.sort(key=lambda a: (_PRIORITY.get(a["class_name"], 2), a["text"]))
    return alerts
