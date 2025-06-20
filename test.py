import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
print("Available classes:", model.names)  # Print class names

# Path to the input video
video_path = r'D:\cvprojects\proposal\Construction\Data\bandicam 2024-10-05 12-13-50-312.mp4'
output_video_path = r'D:\cvprojects\proposal\Construction\Output\output_video_with_detections.mp4'

# Load the video using OpenCV
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open the input video file.")
    exit()

# Get video properties (width, height, FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Could not open the output video file for writing.")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop when the video ends
    
    # Perform inference on the current frame
    results = model(frame)

    # Convert the results to a pandas DataFrame
    df = results.pandas().xyxy[0]

    # Create a copy of the original frame for output
    output_frame = frame.copy()
    
    # Draw bounding boxes for detected objects on the output frame
    for i, row in df.iterrows():
        label = row['name']  # Class label
        confidence = row['confidence']  # Confidence score
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  # Bounding box
        
        # Print detected classes and confidence
        print(f'Detected: {label}, Confidence: {confidence}') 

        # Adjust the confidence threshold if needed
        if confidence > 0.1:  # Testing with a low threshold
            # Draw bounding box and label on the output frame
            color = (0, 255, 0) if label == 'helmet' else (0, 0, 255)  # Green for helmets, red for others
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write the frame with the drawn detections to the output video
    out.write(output_frame)

    # Display the original frame (input)
    cv2.imshow('Input Video', frame)
    
    # Display the frame with detections
    cv2.imshow('Detection Output', output_frame)
    
    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
