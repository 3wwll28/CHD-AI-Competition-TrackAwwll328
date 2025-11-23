# text_adapter.py - 完全修复版本（修复StopIteration错误）
import torch
import torch.nn as nn

class TextFeatureAdapter(nn.Module):
    """文本特征适配器 - 修复StopIteration版本"""
    
    def __init__(self, input_dim=256, output_dim=256, num_layers=1, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # 简化的适配器：如果输入输出维度相同，使用恒等映射
        if input_dim == output_dim:
            self.adapter = nn.Identity()
            self.has_parameters = False  # 标记没有参数
        else:
            # 简单的线性投影
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            self.has_parameters = True  # 标记有参数
        
        # 立即移动到设备
        if device:
            self.to(device)
        
        print(f"      📝 文本适配器: {input_dim} -> {output_dim}, 设备: {device}")
    
    def forward(self, text_features):
        """适配文本特征维度 - 修复StopIteration版本"""
        # 处理不同类型的输入
        if isinstance(text_features, dict) and 'features' in text_features:
            text_features_tensor = text_features['features']
        else:
            text_features_tensor = text_features
        
        # 修复：只在有参数时检查设备
        if self.has_parameters and self.device:
            # 确保适配器在正确的设备上
            if next(self.parameters()).device != text_features_tensor.device:
                self.to(text_features_tensor.device)
        
        original_shape = text_features_tensor.shape
        
        # 应用适配器
        adapted = self.adapter(text_features_tensor)
        
        print(f"      🔧 文本特征适配: {original_shape} -> {adapted.shape}")
        return adapted

class CompleteTextProcessor:
    """完整的文本处理器 - 修复版本"""
    
    def __init__(self, original_processor, target_dim=256, device=None):
        self.original_processor = original_processor
        self.device = device
        
        # 文本处理器已经输出256维，所以适配器可以是恒等映射
        self.adapter = TextFeatureAdapter(256, target_dim, device=device)
        
    def encode_text(self, text_query):
        """编码文本 - 修复版本"""
        # 使用原始处理器
        result = self.original_processor.encode_text(text_query)
        
        # 提取特征和掩码
        if isinstance(result, dict):
            text_features = result['features']
            text_mask = result['mask']
        else:
            text_features, text_mask = result
        
        # 确保在正确的设备上
        if self.device and text_features.device != self.device:
            text_features = text_features.to(self.device)
            text_mask = text_mask.to(self.device)
        
        print(f"      📊 原始文本特征: {text_features.shape}")
        
        # 适配维度（可能只是恒等映射）
        adapted_features = self.adapter(text_features)
        
        return {
            'features': adapted_features,
            'mask': text_mask
        }
    
    def __call__(self, text_query):
        return self.encode_text(text_query)

# 备用方案：完全跳过适配器
class DirectTextProcessor:
    """直接文本处理器，完全跳过适配器"""
    
    def __init__(self, original_processor, device=None):
        self.original_processor = original_processor
        self.device = device
        print("      📝 使用直接文本处理器（跳过适配器）")
    
    def encode_text(self, text_query):
        """直接使用原始文本特征"""
        result = self.original_processor.encode_text(text_query)
        
        if isinstance(result, dict):
            return result
        else:
            return {'features': result[0], 'mask': result[1]}
    
    def __call__(self, text_query):
        return self.encode_text(text_query)