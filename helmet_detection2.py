import cv2
import torch


# Load YOLOv5 model (nano version for speed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Function to detect yellow helmets
def detect_yellow_helmets(video_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model(frame)

        # Filter results for helmets (assuming helmet is class 0; adjust accordingly)
        for *box, conf, cls in results.xyxy[0]:
            if cls == 0:  # Assuming class 0 is for helmets
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle around helmet

        # Show the frame with detections
        cv2.imshow('Helmet Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Specify your video paths
video_paths = [
    'D:\svn\Construction\Data\bandicam 2024-10-05 12-13-50-312.mp4'
]

for video in video_paths:
    detect_yellow_helmets(video)
