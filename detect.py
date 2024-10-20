import cv2

# Load the pre-trained SSD model and the configuration file
model = 'frozen_inference_graph.pb'
config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNetFromTensorflow(model, config)

# Load the video
video_path = 'epic_horses.mp4'
img = cv2.VideoCapture(video_path)

# Get the width and height of the video frames for output video
frame_width = int(img.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(img.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save output video
output_video = cv2.VideoWriter('output_horse_detection.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

# Set the scale factor for resizing
scale_factor = 1 # Adjust this value to scale down the frame size

while True:
    ret, frame = img.read()
    if not ret:
        break
    
    # Resize the frame
    scaled_frame = cv2.resize(frame, (int(frame_width * scale_factor), int(frame_height * scale_factor)))

    # Preprocess the scaled image
    height, width, _ = scaled_frame.shape
    blob = cv2.dnn.blobFromImage(scaled_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the model
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = labels[class_id]
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            # Scale the bounding box back to the original frame size
            x1 = int(x1 / scale_factor)
            y1 = int(y1 / scale_factor)
            x2 = int(x2 / scale_factor)
            y2 = int(y2 / scale_factor)

            # Draw the bounding box on the original frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label on the image
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Write the frame with detections to the output video
    output_video.write(frame)

    # Display the image
    cv2.imshow('image', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and VideoWriter, then close all windows
img.release()
output_video.release()  # Release the VideoWriter
cv2.destroyAllWindows()
