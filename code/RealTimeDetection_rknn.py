import torch
import torchvision
import hopenet
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
    rknn = RKNNLite()

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

# 使用示例
if __name__ == "__main__":
    # 保存模型为TorchScript格式
    # save_model_to_torchscript(hopenet.Hopenet, "../hopenet_robust_alpha1.pkl", "../hopenet_scripted.pt")
    convert_pytorch_to_rknn("../hopenet_scripted.pt", "../hopenet_scripted.rknn")
