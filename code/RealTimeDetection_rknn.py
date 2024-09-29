import torch
import torchvision
import hopenet
# from rknn.api import RKNN
from rknnlite.api import RKNNLite


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
    使用导出的RKNN模型对实时视频进行处理

    参数:
    rknn_model_path: str, 已导出的RKNN模型的路径 (.rknn)

    返回:
    None
    """
    # 创建 RKNN 对象
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
    ret = rknn_lite.init_runtime(target='rk3588')  # 根据目标设备修改 'rk3566', 'rk3588', 'rk1808', 等
    if ret != 0:
        print('Init runtime failed!')
        return
    print('done')

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 参数 0 表示打开本地摄像头

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 实时视频处理循环
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 调整图像大小到模型输入大小
        img = cv2.resize(frame, (224, 224))
        img = img.astype('float32')

        # 将图像转换为输入数据
        img_input = np.expand_dims(img, axis=0)  # 添加 batch 维度

        # 推理
        outputs = rknn.inference(inputs=[img_input])

        # 显示推理结果 (这里的输出内容需根据你的模型进行解析和展示)
        print(f'Inference results: {outputs}')

        # 在摄像头图像上展示推理结果
        cv2.putText(frame, f'Yaw: {outputs[0][0]:.2f}, Pitch: {outputs[1][0]:.2f}, Roll: {outputs[2][0]:.2f}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示图像
        cv2.imshow('Real-Time Video Processing', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()

    # 释放 RKNN 资源
    rknn.release()


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

