import torch.onnx
import torchvision
import torch.backends.cudnn as cudnn
import hopenet
import torch


def export_hopenet_to_onnx(snapshot_path, onnx_model_path, opset_version=11, use_cuda=True):
    """
    将 Hopenet 模型导出为 ONNX 格式的函数

    Args:
    - snapshot_path (str): PyTorch 模型权重文件路径。
    - onnx_model_path (str): 导出的 ONNX 模型保存路径。
    - opset_version (int): ONNX 的 opset 版本，默认 11。
    - use_cuda (bool): 是否使用 GPU，如果为 True 则使用 GPU。

    Returns:
    - None
    """
    # 检测是否有可用的 GPU
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    cudnn.enabled = True

    # 构建模型结构
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # 加载模型权重
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)

    # 将模型移到指定设备
    model.to(device)

    # 模型设置为评估模式
    model.eval()

    # 创建一个虚拟输入（与实际输入的形状一致），用于导出 ONNX 模型
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 导出模型为 ONNX 格式
    torch.onnx.export(
        model,  # 你的 PyTorch 模型
        dummy_input,  # 模拟输入
        onnx_model_path,  # 保存 ONNX 模型的路径
        export_params=True,  # 保存模型参数
        opset_version=opset_version,  # ONNX opset 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=['input'],  # 输入张量名称
        output_names=['yaw', 'pitch', 'roll'],  # 输出张量名称
        dynamic_axes={'input': {0: 'batch_size'},  # 允许动态批次大小
                      'yaw': {0: 'batch_size'},
                      'pitch': {0: 'batch_size'},
                      'roll': {0: 'batch_size'}}
    )

    print(f"ONNX 模型已导出至: {onnx_model_path}")


def export_hopenet_to_torchscript(snapshot_path, torchscript_model_path, use_cuda=True):
    """
    将 Hopenet 模型导出为 TorchScript 格式的函数

    Args:
    - snapshot_path (str): PyTorch 模型权重文件路径。
    - torchscript_model_path (str): 导出的 TorchScript 模型保存路径。
    - use_cuda (bool): 是否使用 GPU，如果为 True 则使用 GPU。

    Returns:
    - None
    """
    # 检测是否有可用的 GPU
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    cudnn.enabled = True

    # 构建模型结构
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # 加载模型权重
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)

    # 将模型移到指定设备
    model.to(device)

    # 模型设置为评估模式
    model.eval()

    # 创建一个虚拟输入（与实际输入的形状一致）
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 使用 torch.jit.trace 将模型转换为 TorchScript 格式
    traced_script_module = torch.jit.trace(model, dummy_input)

    # 保存为 TorchScript 文件
    traced_script_module.save(torchscript_model_path)

    print(f"TorchScript 模型已导出至: {torchscript_model_path}")


# 调用函数进行 ONNX 和 TorchScript 模型导出
if __name__ == "__main__":
    snapshot_path = "../hopenet_robust_alpha1.pkl"  # 模型权重文件路径
    onnx_model_path = "../hopenet_model.onnx"  # 导出的 ONNX 模型保存路径
    torchscript_model_path = "../hopenet_model.pt"  # 导出的 TorchScript 模型保存路径

    # 导出 ONNX 模型
    # export_hopenet_to_onnx(snapshot_path, onnx_model_path)

    # 导出 TorchScript 模型
    export_hopenet_to_torchscript(snapshot_path, torchscript_model_path)
