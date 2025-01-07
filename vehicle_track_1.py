import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

# Static Line Drawing

"""
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

"""

# With Dynamic Line drawing

model = YOLO("yolo11n.pt")
class_list = model.names
print(class_list)

cap = cv2.VideoCapture("highway.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

class_counts_out = defaultdict(int)
crossed_ids_out = set()

class_counts_in = defaultdict(int)
crossed_ids_in = set()

vehicle_paths = defaultdict(list)
allowed_classes = {1, 2, 3, 5, 7}

vehicle_last_position = {}
PIXELS_TO_KMH = 0.1


frame_skip = 1
frame_count = 0

def detect_lanes_and_lines(frame):
    """
    Detect lanes using edge detection and Hough transform.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    
    left_lane = []
    right_lane = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
            if slope < 0:  # Left lane
                left_lane.append((x1, y1, x2, y2))
            elif slope > 0:  # Right lane
                right_lane.append((x1, y1, x2, y2))

    return left_lane, right_lane


def calculate_lane_lines(left_lane, right_lane, frame_width, frame_height):
    """
    Calculate 'In' and 'Out' lines for each lane (left and right) based on detected lanes.
    """
    if not left_lane and not right_lane:
        # Default to approximate locations if no lanes are detected
        y_left_out = frame_height // 3  # Default 'Out' line closer to the middle
        y_left_in = frame_height // 2  # Default 'In' line closer to the bottom
        x_left = frame_width // 4

        y_right_out = frame_height // 3
        y_right_in = frame_height // 2
        x_right = 3 * frame_width // 4

        return (x_left, y_left_in, y_left_out), (x_right, y_right_in, y_right_out)

    # Extract min and max y-values for detected lines
    y_left = [y1 for x1, y1, x2, y2 in left_lane] + [y2 for x1, y1, x2, y2 in left_lane]
    y_right = [y1 for x1, y1, x2, y2 in right_lane] + [y2 for x1, y1, x2, y2 in right_lane]

    y_min_left = int(min(y_left, default=frame_height))
    y_max_left = int(max(y_left, default=frame_height))

    y_min_right = int(min(y_right, default=frame_height))
    y_max_right = int(max(y_right, default=frame_height))

    # Adjust placement of 'Out' and 'In' lines dynamically
    lane_height_left = y_max_left - y_min_left
    lane_height_right = y_max_right - y_min_right

    # Ensure lines are placed within a reasonable height range
    y_left_out = y_min_left + int(0.15 * lane_height_left)  # 'Out' line higher (near top)
    y_left_in = y_min_left + int(0.6 * lane_height_left)  # 'In' line lower (closer to bottom)

    y_right_out = y_min_right + int(0.47 * lane_height_right)  # 'Out' line higher
    y_right_in = y_min_right + int(0.6 * lane_height_right)  # 'In' line lower

    # Horizontal positions of lines (middle of respective lanes)
    x_left = frame_width // 4
    x_right = 3 * frame_width // 4

    return (x_left, y_left_in, y_left_out), (x_right, y_right_in, y_right_out)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Skip frames for lower processing load
    if frame_count % (frame_skip + 1) != 0:
        continue

    # Detect lanes
    left_lane, right_lane = detect_lanes_and_lines(frame)

    # Calculate dynamic 'In' and 'Out' lines
    (x_left, y_left_in, _), (_, _, y_right_out) = calculate_lane_lines(
        left_lane, right_lane, frame.shape[1], frame.shape[0])

    # Draw horizontal counting lines for left (In) and right (Out) lanes
    cv2.line(frame, (0, y_left_in), (frame.shape[1] // 2, y_left_in), (0, 0, 255), 2)  # Red for left lane 'In'
    cv2.line(frame, (frame.shape[1] // 2, y_right_out), (frame.shape[1], y_right_out), (255, 0, 255), 2)  # Pink for right lane 'Out'

    # YOLO model inference
    results = model.track(frame, persist=True)

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Check if `track_ids` is not None
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
        else:
            track_ids = []

        # Process detected objects
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            # Filter only specified classes
            if class_idx not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]
            
            # Save the current position of the vehicle to its path
            if track_id is not None:
                vehicle_paths[track_id].append((cx, cy))

                # Draw the trajectory
                for i in range(1, len(vehicle_paths[track_id])):
                    if vehicle_paths[track_id][i - 1] and vehicle_paths[track_id][i]:
                        cv2.line(frame, vehicle_paths[track_id][i - 1], vehicle_paths[track_id][i], (0, 255, 255), 2)


            # Draw bounding box, ID, and class
            cv2.circle(frame, (cx, cy), 1, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Count vehicles crossing the 'In' line in the left lane
            if cx < frame.shape[1] // 2 and cy > y_left_in and track_id not in crossed_ids_in:
                crossed_ids_in.add(track_id)
                class_counts_in[class_name] += 1

            # Count vehicles crossing the 'Out' line in the right lane
            if cx > frame.shape[1] // 2 and cy < y_right_out and track_id not in crossed_ids_out:
                crossed_ids_out.add(track_id)
                class_counts_out[class_name] += 1

    # Display counts
    y_offset_out = 30
    y_offset_in = 30

    # Display 'Out' counts on the frame
    for class_name, count in class_counts_out.items():
        cv2.putText(frame, f"Out - {class_name}: {count}", (800, y_offset_out),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 0, 0), 2) 
        y_offset_out += 30

    # Display 'In' counts on the frame
    for class_name, count in class_counts_in.items():
        cv2.putText(frame, f"In - {class_name}: {count}", (50, y_offset_in),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 0, 0), 2)
        y_offset_in += 30

    # Display the processed frame
    cv2.imshow("YOLO Object Tracking & Counting", frame)

    # Exit on 'q' key press
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()