import torch
import cv2
import datetime
import pandas as pd
import warnings
import os
warnings.filterwarnings("ignore")


# Path to the input video
video_path = r'D:\svn\Construction\Data\bandicam 2024-10-05 12-13-50-312.mp4'  # Replace with your actual video path

# Load YOLOv5 model (Replace 'yolov5n' with your custom model if needed)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')


# Create output directory for helmet detections
output_path = r'D:\svn\Construction\Output\helmet'
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Create a timestamp for the output video filename
current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_path, f'helmet_output_{current_timestamp}.mp4')  # Output video path
excel_path = os.path.join(output_path, f'helmet_detections_{current_timestamp}.xlsx')  # Excel path
# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video stream or file: {video_path}")

# Get video properties (width, height, FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# DataFrame to store detections
results_df = pd.DataFrame(columns=["Timestamp", "Label", "Confidence"])

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to capture a frame.")
        break

    # Perform inference on the frame
    results = model(frame)

    # Convert results to pandas DataFrame
    df = results.pandas().xyxy[0]

    # Helmet detection logic (COCO doesn't include helmets, you might need a custom model)
    helmet_detected = False
    frame_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Iterate over detections and add details to DataFrame
    for i, row in df.iterrows():
        label = row['name']  # Class label (e.g., 'person', 'helmet')
        confidence = row['confidence']  # Confidence score
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  # Bounding box

        # If 'helmet' is detected, set the flag to true
        if label == 'helmet' and confidence > 0.3:  # Filter for 'helmet' if available
            helmet_detected = True
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Append detection details to the DataFrame (for all detections)
        results_df = pd.concat([results_df, pd.DataFrame({
            "Timestamp": [frame_timestamp],
            "Label": [label],
            "Confidence": [confidence]
        })], ignore_index=True)

    # Display alert if no helmet detected
    if not helmet_detected:
        cv2.putText(frame, 'Violation: No Helmet Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write frame to the output video
    out.write(frame)

    # Display the frame with detections in real-time
    cv2.imshow('Helmet Detection', frame)

    # Add a small delay between frames for proper visualization
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the detections to an Excel file
results_df.to_excel(excel_path, index=False)
print(f"Excel sheet saved as {excel_path}")
