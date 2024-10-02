import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
from mtcnn import MTCNN
from PIL import Image
import numpy as np
from math import cos, sin

# 检测是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TorchScript 模型的路径
scripted_model_path = "../hopenet_model.pt"

# 加载 TorchScript 模型
model = torch.jit.load(scripted_model_path)

# 确保模型使用 GPU 或 CPU
model.to(device)

# 模型设置为评估模式
model.eval()

# Image transformations
transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create tensor for continuous angle predictions
idx_tensor = torch.FloatTensor([idx for idx in range(66)]).to(device)

# Initialize face detector
detector = MTCNN()

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


# Start capturing from webcam
def webcam_real_time_detection():
    cap = cv2.VideoCapture(0)  # Open webcam (change the argument to a video file path for video input)

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
            images = Variable(img).to(device)

            # Perform head pose prediction
            yaw, pitch, roll = model(images)

            # Continuous predictions
            yaw_predicted = softmax_temperature(yaw.data, 1)
            pitch_predicted = softmax_temperature(pitch.data, 1)
            roll_predicted = softmax_temperature(roll.data, 1)

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
            draw_axis(frame, yaw, pitch, roll, tdx=face_center_x, tdy=face_center_y)

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
