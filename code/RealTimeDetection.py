import sys, os, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import datasets, hopenet, utils
from mtcnn import MTCNN
from PIL import Image

cudnn.enabled = True
snapshot_path = "../hopenet_robust_alpha1.pkl"

# ResNet50 structure
model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

# Load model weights
saved_state_dict = torch.load(snapshot_path, map_location=torch.device('cpu'))
model.load_state_dict(saved_state_dict)

# Image transformations
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ensure the model runs on CPU
model.to(torch.device('cpu'))

# Model in evaluation mode
model.eval()

# Create tensor for continuous angle predictions
idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(torch.device('cpu'))

# Initialize face detector
detector = MTCNN()

# L1 loss for error calculation
l1loss = torch.nn.L1Loss(size_average=False)


# Start capturing from webcam
def webcam_real_time_detection():
    cap = cv2.VideoCapture(61)  # Open webcam (change the argument to a video file path for video input)

    # Set the camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        faces = detector.detect_faces(img_rgb)

        if len(faces) > 0:
            face = faces[0]['box']  # Assuming a single face
            x, y, w, h = map(int, face)  # Ensure coordinates are integers

            # Print the detected face coordinates for debugging
            print(f"Detected face at: x={x}, y={y}, w={w}, h={h}")

            # Crop and prepare the image for the model
            img = Image.fromarray(img_rgb)
            img = img.crop((int(x - 20), int(y - 20), int(x + w + 20), int(y + h + 20)))
            img = img.convert('RGB')
            img = transformations(img)
            img = img.unsqueeze(0)  # Add batch dimension

            # Prepare image for model
            images = Variable(img).to(torch.device('cpu'))

            # Perform head pose prediction
            yaw, pitch, roll = model(images)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            # Extract predictions
            pitch = pitch_predicted[0].item()
            yaw = yaw_predicted[0].item()  # Negate yaw for visualization
            roll = roll_predicted[0].item()

            # Calculate the center of the face for axis drawing
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Print the calculated center for debugging
            print(f"Drawing axis at: face_center_x={face_center_x}, face_center_y={face_center_y}")

            # Draw axis on the frame at the face's center
            utils.draw_axis(frame, yaw, pitch, roll, tdx=face_center_x, tdy=face_center_y)

            # Display yaw, pitch, and roll values on the image
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the result
        cv2.imshow('Head Pose Estimation', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Run the webcam real-time detection
webcam_real_time_detection()
