import torch
import cv2
import datetime
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

# Load YOLOv5 model (yolov5n in your case)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Path to the input video
video_path = 'D:\svn\Construction\Data\bandicam 2024-10-05 12-13-50-312.mp4'  # Replace with your actual video path

output_path = r'D:\svn\Construction\Output\helmet'
excel_path = os.path.join(output_path, f"helmet_detection_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

# Create a timestamp for the output video filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_video_path = os.path.join(output_path, f'output_video_with_detections_{timestamp}.mp4')  # Output video path with timestamp

# Load the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=["Timestamp", "Label", "Confidence"])

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop when the video ends
    
    # Perform inference on the current frame
    results = model(frame)

    # Convert the results to a pandas DataFrame
    df = results.pandas().xyxy[0]

    # Get current timestamp
    current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log the detected labels and their confidence scores
    helmet_detected = False  # Flag to check if helmet is detected

    for i, row in df.iterrows():
        label = row['name']  # Class label (e.g., 'person', 'helmet')
        confidence = row['confidence']  # Confidence score

        # Append to DataFrame if it's a helmet
        if label == 'helmet' and confidence > 0.3:
            results_df = results_df.append({"Timestamp": current_timestamp, "Label": label, "Confidence": confidence}, ignore_index=True)
            helmet_detected = True

    # Check if any helmets were detected and display violation alert
    if not helmet_detected:
        cv2.putText(frame, 'Violation: No Helmet Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the frame with the drawn detections to the output video
    out.write(frame)

    # Display the input video frame
    cv2.imshow('Input Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

# Release the video capture and writer
cap.release()
out.release()

# Save the results to Excel only if there are any detections
if not results_df.empty:
    results_df.to_excel(excel_path, index=False)
else:
    print("No helmet detections were recorded.")

# Destroy all OpenCV windows
cv2.destroyAllWindows()

print(f"Detection results saved to {excel_path}")
