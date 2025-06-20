import torch
import cv2
import pandas as pd
import datetime
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load YOLOv5 model (yolov5n in your case)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Path to the input video
video_path = r'D:\svn\Construction\Data\bandicam 2024-10-05 12-13-50-312.mp4'  # Replace with your actual video path
output_path = r'D:\svn\Construction\Output\person'

# Create a timestamp for the output video filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
excel_path = f'{output_path}\\person_detection_results_{timestamp}.xlsx'  # Output Excel path with timestamp

# Load the video using OpenCV
cap = cv2.VideoCapture(video_path)

# Initialize DataFrame for results
results_df = pd.DataFrame(columns=["Timestamp", "Number of Persons Detected", "Confidence Scores"])

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop when the video ends

    # Perform inference on the current frame
    results = model(frame)

    # Convert the results to a pandas DataFrame
    df = results.pandas().xyxy[0]

    # Count the number of persons detected and collect confidence scores
    num_persons = 0
    confidence_scores = []
    
    for i, row in df.iterrows():
        label = row['name']  # Class label (e.g., 'person')
        confidence = row['confidence']  # Confidence score
        
        if label == 'person' and confidence > 0.3:  # Filter for persons only
            num_persons += 1
            confidence_scores.append(confidence)  # Append confidence score to the list

            # Draw bounding box and label on the frame
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Get current timestamp
    current_timestamp = datetime.datetime.now()

    # Create a new DataFrame for the current frame's results
    current_result = pd.DataFrame({
        "Timestamp": [current_timestamp],
        "Number of Persons Detected": [num_persons],
        "Confidence Scores": [", ".join(map(str, confidence_scores))]  # Join confidence scores as a string
    })

    # Concatenate the current result with the main DataFrame
    results_df = pd.concat([results_df, current_result], ignore_index=True)

    # Display the frame with detections
    cv2.imshow('Video Frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save results to Excel
results_df.to_excel(excel_path, index=False)

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

print(f"Results saved to {excel_path}")
