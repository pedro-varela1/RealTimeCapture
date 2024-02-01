import cv2
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path where the model is saved
# (contains: config.json, preprocessor_config.json, pytorch_model.bin)
# If you already have the model downloaded, you can use the path to the local directory
# The path bellows is the model from Hugging Face model hub
model_path = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(model_path, revision="no_timm", device=device)
model = DetrForObjectDetection.from_pretrained(model_path, revision="no_timm").to(device)

# Start video capture from camera (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to PIL format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    # Make predictions using the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

    # Convert the outputs to COCO API format and keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs,
                                                      target_sizes=target_sizes, threshold=0.9)[0]

    # Add rectangles for each detection
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

        # Get the class corresponding to the label
        class_name = model.config.id2label[label.item()]

        # Draw a rectangle using the bounding box coordinates
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

        # Add a label with the class and confidence
        label_text = f"{class_name}: {round(score.item(), 3)}"
        cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Show the frame with the detections
    cv2.imshow("Real-Time Object Detection", frame)

    # Check if the 'q' key was pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
