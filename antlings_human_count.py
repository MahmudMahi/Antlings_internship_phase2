from ultralytics import YOLO
from pathlib import Path
import cv2

MODEL_PATH = r"C:\Antlings_openCV\visdrone_human_car-4\weights\best.pt"
IMAGE_PATH = r"C:\Antlings_openCV\original_media_files\DSC04931_-_Race_-_Close-up.d1a869ec.fill-1370x800.jpg"
OUT_PATH = r"C:\Antlings_openCV\annotated_media_files\annotated_image.jpg"

model = YOLO(MODEL_PATH)

results = model.predict(source=IMAGE_PATH, conf=0.25, verbose=False)
r = results[0]

img = cv2.imread(IMAGE_PATH)

human_count = 0

if r.boxes is not None and len(r.boxes) > 0:
    boxes = r.boxes.xyxy.cpu().numpy().astype(int)
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for box, cls_id in zip(boxes, clss):
        x1, y1, x2, y2 = box

        if cls_id == 0:
            human_count += 1
            color = (0, 0, 255)
            label = "human"
        elif cls_id == 1:
            color = (255, 0, 0)
            label = "car"
        else:
            color = (0, 255, 255)
            label = f"class{cls_id}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

cv2.putText(
    img,
    f"Human Count: {human_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.imwrite(OUT_PATH, img)
print("Human count:", human_count)
print("Saved annotated image to:", OUT_PATH)