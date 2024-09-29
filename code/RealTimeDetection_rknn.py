import torch
import torchvision
import hopenet
# from rknn.api import RKNN
from rknnlite.api import RKNNLite
import cv2
import numpy as np
import utils
from mtcnn import MTCNN
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


def save_model_to_torchscript(model_class, model_path, save_path):
    """
    将现有的PyTorch模型转换为TorchScript格式并保存。

    参数:
    model_class: PyTorch 模型对象（类，而非实例）
    model_path: str, 预训练模型的路径 (.pth 或 .pkl)
    save_path: str, 保存 TorchScript 模型的路径 (.pt)

    返回:
    None
    """
    # 创建模型实例
    model = model_class(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # 加载预训练模型权重
    saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)

    # 将模型转换为TorchScript格式
    scripted_model = torch.jit.script(model)

    # 保存TorchScript模型
    scripted_model.save(save_path)

    print(f"模型已保存为TorchScript格式, 路径为: {save_path}")


def convert_pytorch_to_rknn(torchscript_model_path, rknn_model_path):
    # 创建RKNNLite对象
    rknn = RKNN()

    # 配置模型，使用与PyTorch图像预处理相匹配的mean和std
    print('--> Configuring model')
    rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[0.229, 0.224, 0.225]], target_platform='rk3588')
    print('done')

    # 加载PyTorch TorchScript模型
    print('--> Loading TorchScript model')
    ret = rknn.load_pytorch(model=torchscript_model_path, input_size_list=[[1, 3, 224, 224]])
    if ret != 0:
        print('Load PyTorch model failed!')
        return

    print('done')

    # 构建RKNN模型
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build RKNN model failed!')
        return

    print('done')

    # 导出RKNN模型
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export RKNN model failed!')
        return

    print(f'RKNN model saved to {rknn_model_path}')

    # 释放RKNN对象
    rknn.release()


def save_model_to_onnx(model_class, model_path, save_path):
    """
    将现有的PyTorch模型转换为ONNX格式并保存。

    参数:
    model_class: PyTorch 模型对象（类，而非实例）
    model_path: str, 预训练模型的路径 (.pth 或 .pkl)
    save_path: str, 保存 ONNX 模型的路径 (.onnx)

    返回:
    None
    """
    # 创建模型实例
    model = model_class(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # 加载预训练模型权重
    saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)

    # 设置模型为评估模式
    model.eval()

    # 创建一个虚拟输入，大小与模型输入相匹配
    dummy_input = torch.randn(1, 3, 224, 224)

    # 导出模型为 ONNX 格式
    torch.onnx.export(model, dummy_input, save_path, export_params=True, opset_version=11)

    print(f"模型已保存为ONNX格式, 路径为: {save_path}")


def convert_onnx_to_rknn(onnx_model_path, rknn_model_path):
    """
    将ONNX模型转换为RKNN模型并保存。

    参数:
    onnx_model_path: str, ONNX 模型的路径 (.onnx)
    rknn_model_path: str, 保存 RKNN 模型的路径 (.rknn)

    返回:
    None
    """
    # 创建RKNN对象
    rknn = RKNN()

    # 配置模型参数，确保与ONNX模型的输入图像预处理保持一致
    print('--> Configuring model')
    rknn.config(mean_values=[[0.485 * 255, 0.456 * 255, 0.406 * 255]],
                std_values=[[0.229 * 255, 0.224 * 255, 0.225 * 255]],
                target_platform='rk3588')
    print('done')

    # 加载ONNX模型
    print('--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_model_path)
    if ret != 0:
        print('Load ONNX model failed!')
        return
    print('done')

    # 构建RKNN模型
    print('--> Building RKNN model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build RKNN model failed!')
        return
    print('done')

    # 导出RKNN模型
    print('--> Exporting RKNN model')
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export RKNN model failed!')
        return

    print(f'RKNN model saved to {rknn_model_path}')

    # 释放RKNN对象
    rknn.release()


def run_rknn_realtime_video(rknn_model_path):
    """
    使用导出的RKNN模型对实时视频进行处理，并在视频帧中绘制yaw、pitch、roll方向的坐标轴

    参数:
    rknn_model_path: str, 已导出的RKNN模型的路径 (.rknn)

    返回:
    None
    """

    # 创建 RKNNLite 对象
    rknn_lite = RKNNLite()

    # 加载 RKNN 模型
    print('--> Loading RKNN model')
    ret = rknn_lite.load_rknn(rknn_model_path)
    if ret != 0:
        print('Load RKNN model failed!')
        return
    print('done')

    # 初始化运行时环境
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime failed!')
        return
    print('done')

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

    # 打开摄像头
    cap = cv2.VideoCapture(61)  # 参数为摄像头索引或视频文件路径

    # Set the camera resolution to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 实时视频处理循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
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

            # Perform head pose prediction
            yaw, pitch, roll = rknn_lite.inference(inputs=[img])

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

    # 释放 RKNN 资源
    rknn_lite.release()


# 使用示例
if __name__ == "__main__":
    # 保存模型为TorchScript格式
    # save_model_to_torchscript(hopenet.Hopenet, "../hopenet_robust_alpha1.pkl", "../hopenet_scripted.pt")
    # convert_pytorch_to_rknn("../hopenet_scripted.pt", "../hopenet_scripted.rknn")
    # 运行实时视频处理
    run_rknn_realtime_video("../hopenet_model.rknn")
    # save_model_to_onnx(hopenet.Hopenet, "../hopenet_robust_alpha1.pkl", "../hopenet_model.onnx")
    # 将ONNX模型转换为RKNN模型
    # convert_onnx_to_rknn("../hopenet_model.onnx", "../hopenet_model.rknn")

