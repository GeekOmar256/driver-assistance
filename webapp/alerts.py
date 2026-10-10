import config

DIRECTION_PHRASE = {
    "left": "on your left",
    "right": "on your right",
    "ahead": "ahead",
}

# Shown first, since they call for a driver response rather than just awareness.
_PRIORITY = {"stop sign": 0, "traffic light": 1}


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
        })

    alerts.sort(key=lambda a: (_PRIORITY.get(a["class_name"], 2), a["text"]))
    return alerts
