import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO("yolo11n.pt")

# # results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# results = model("3.mp4", save=True, show=True)

class_list = model.names
print(class_list)

cap = cv2.VideoCapture("highway.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

y_line_red = 430

class_counts_out = defaultdict(int)
crossed_ids_out = set()

class_counts_in = defaultdict(int)
crossed_ids_in = set()

frame_skip = 2  # Process every 3rd frame (skip 2 frames in between)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Skip frames for lower frame rate
    if frame_count % (frame_skip + 1) != 0:
        continue

    # YOLO Model Inference
    results = model.track(frame, persist=True)

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()
        cv2.line(frame, (670, y_line_red), (950, y_line_red), (0, 0, 255), 1)
        cv2.line(frame, (310, y_line_red), (590, y_line_red), (0, 0, 255), 1)

        # Draw bounding boxes and labels
    for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        class_name = class_list[class_idx]
        cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        if cy < y_line_red and cx > 670 and track_id not in crossed_ids_out:
            crossed_ids_out.add(track_id)
            class_counts_out[class_name] += 1

        if cy > y_line_red and cx < 590 and track_id not in crossed_ids_in:
            crossed_ids_in.add(track_id)
            class_counts_in[class_name] += 1
    y_offset_out = 30
    y_offset_in = 30

    for class_name, count in class_counts_out.items():
        cv2.putText(frame, f"Out - {class_name}: {count}", (900, y_offset_out),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 0, 0), 2)
        y_offset_out += 30
    
    for class_name, count in class_counts_in.items():
        cv2.putText(frame, f"In - {class_name}: {count}", (50, y_offset_in),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 0, 0), 2)
        y_offset_in += 30

    # Display the processed frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()