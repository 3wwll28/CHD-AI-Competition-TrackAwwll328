# 建议保留文档的版本
import torch

def inverse_sigmoid(x, eps=1e-5):
    """sigmoid反函数"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def get_activation_fn(activation):
    """获取激活函数"""
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu
    elif activation == "glu":
        return torch.nn.functional.glu
    else:
        raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def nested_tensor_from_tensor_list(tensor_list):
    """从张量列表创建嵌套张量"""
    if len(tensor_list) == 0:
        return None
    return torch.stack(tensor_list)

__all__ = ['inverse_sigmoid', 'get_activation_fn', 'nested_tensor_from_tensor_list']