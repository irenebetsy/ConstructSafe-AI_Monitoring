import cv2
import torch
import time
import numpy as np
import os

# Load the YOLOv5 model (trained for person and helmet detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Define safety classes for detection
SAFETY_CLASSES = ['person', 'helmet']  # Assuming your YOLO model detects these classes

# Folder path containing videos
video_folder = r'D:\svn\Construction\Data'  # Corrected path
# Path to your folder containing videos

# List all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Define restricted zones as top-left and bottom-right corners of rectangles (adjust these as per video)
restricted_zones = [((100, 100), (300, 300)), ((400, 100), (600, 300))]  # Example of restricted zones

# Function to check if a detected person is in a restricted zone
def is_in_restricted_zone(cord, frame):
    x1, y1, x2, y2 = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0])
    for (top_left, bottom_right) in restricted_zones:
        if top_left[0] < x1 < bottom_right[0] and top_left[1] < y1 < bottom_right[1]:
            return True
    return False

# Function to check if a person is wearing a helmet based on proximity of bounding boxes
def is_wearing_helmet(person_bbox, helmet_bboxes):
    px1, py1, px2, py2 = person_bbox
    
    for hx1, hy1, hx2, hy2 in helmet_bboxes:
        # Check if the helmet is within or above the person's bounding box (simple proximity check)
        if hx1 >= px1 and hy1 >= py1 and hx2 <= px2 and hy2 <= py2:
            return True  # A helmet is detected within the person's bounding box
    return False  # No helmet detected within proximity of the person

# Function to process each frame and detect objects
def process_frame(frame):
    # Perform object detection with YOLOv5
    results = model(frame)

    # Extract detected object names and bounding boxes
    labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()

    # Filter detections
    people_bboxes = []
    helmet_bboxes = []
    detections = []
    
    for label, cord in zip(labels, cords):
        name = model.names[int(label)]
        if name == 'person':
            people_bboxes.append((cord[0], cord[1], cord[2], cord[3]))  # Store person bounding boxes
        elif name == 'helmet':
            helmet_bboxes.append((cord[0], cord[1], cord[2], cord[3]))  # Store helmet bounding boxes
        detections.append((name, cord))

    return detections, people_bboxes, helmet_bboxes

# Loop through all the videos in the folder
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture has been initialized correctly
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    # Set video resolution (reduce resolution for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Height

    print(f"Processing video: {video_file}")

    # Main loop for processing the video
    while cap.isOpened():
        start_time = time.time()

        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if the video ends

        # Process the frame to detect objects
        detections, people_bboxes, helmet_bboxes = process_frame(frame)

        # Draw bounding boxes and labels on the frame
        for detection in detections:
            name, cord = detection

            # Coordinates and bounding box
            x1, y1, x2, y2 = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0])

            # Draw a rectangle around detected object
            if name == 'person':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for person
            if name == 'helmet':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for helmet

            # Put the label of the object detected
            cv2.putText(frame, f'{name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Check for violations (person without helmet or person in restricted zone)
        for person_bbox in people_bboxes:
            if not is_wearing_helmet(person_bbox, helmet_bboxes):
                print(f"Violation: Worker without helmet detected in video: {video_file}")
                cv2.putText(frame, 'No Helmet!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)  # Red alert text

            if is_in_restricted_zone(person_bbox, frame):
                print(f"Violation: Worker in restricted zone detected in video: {video_file}")
                cv2.putText(frame, 'Restricted Zone!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)  # Red alert text

        # Calculate and display FPS (frames per second)
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Show the processed frame
        cv2.imshow(f'Safety Monitoring - {video_file}', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture for the current video
    cap.release()

# Close all open windows after processing all videos
cv2.destroyAllWindows()
