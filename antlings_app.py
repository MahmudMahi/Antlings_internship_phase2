import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\Antlings_openCV\visdrone_human_car-4\weights\best.pt"
model = YOLO(MODEL_PATH)

def draw_only_human_car(frame, results):
    r = results[0]
    annotated = frame.copy()
    human_count = 0

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)

        for box, cls_id, track_id in zip(boxes, clss, ids):
            if cls_id not in [0, 1]:
                continue

            x1, y1, x2, y2 = box

            if cls_id == 0:
                label = "human"
                color = (0, 0, 255)
                human_count += 1
            else:
                label = "car"
                color = (255, 0, 0)

            text = f"{label} ID:{track_id}" if track_id is not None else label

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                text,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

    cv2.putText(
        annotated,
        f"Human Count: {human_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    return annotated

VIDEO_PATH = r"C:\Antlings_openCV\original_media_files\YTDown_YouTube_Busy-Road-in-Dhaka-City-l-300-Feet-Road-_Media_9IaU8FRafgI_001_1080p.mp4"
OUT_PATH = r"C:\Antlings_openCV\annotated_media_files\tracked_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        iou=0.5,
        verbose=False
    )

    annotated = draw_only_human_car(frame, results)
    out.write(annotated)

cap.release()
out.release()
print("Saved to:", OUT_PATH)