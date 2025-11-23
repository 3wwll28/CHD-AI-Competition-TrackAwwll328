# dimension_adapter.py - 修复版本
import torch
import torch.nn as nn
import torch.nn.functional as F

class CompetitionDimensionAdapter:
    """比赛专用维度适配器 - 确保所有模块兼容"""
    
    def __init__(self, hidden_dim=256):
        self.hidden_dim = hidden_dim
        self.adapters = {}
        print(f"🔧 初始化比赛维度适配器: 目标维度={hidden_dim}")
    
    def adapt_visual_features(self, visual_features):
        """适配视觉特征维度"""
        adapted_features = []
        for i, feat in enumerate(visual_features):
            print(f"      🔧 适配视觉特征尺度 {i}: {feat.shape}")
            
            # 确保通道数为256
            if feat.shape[1] != self.hidden_dim:
                adapter = nn.Conv2d(feat.shape[1], self.hidden_dim, 1)
                adapter = adapter.to(feat.device)
                adapted_feat = adapter(feat)
                print(f"         ✅ 通道数适配: {feat.shape[1]} -> {self.hidden_dim}")
            else:
                adapted_feat = feat
            
            adapted_features.append(adapted_feat)
        
        return adapted_features
    
    def adapt_text_features(self, text_features):
        """适配文本特征维度"""
        if isinstance(text_features, dict):
            features = text_features['features']
            mask = text_features['mask']
        else:
            features, mask = text_features
            
        print(f"      🔧 适配文本特征: {features.shape}")
        
        # 确保文本特征维度为256
        if features.shape[-1] != self.hidden_dim:
            adapter = nn.Linear(features.shape[-1], self.hidden_dim)
            adapter = adapter.to(features.device)
            adapted_features = adapter(features)
            print(f"         ✅ 文本维度适配: {features.shape[-1]} -> {self.hidden_dim}")
        else:
            adapted_features = features
            
        return {'features': adapted_features, 'mask': mask}
    
    def adapt_depth_features(self, depth_features):
        """适配深度特征维度"""
        print(f"      🔧 适配深度特征: {depth_features.shape}")
        
        if depth_features.shape[1] != self.hidden_dim:
            # 深度特征通常是 [batch, frames, channels, H, W]
            batch, frames, channels, H, W = depth_features.shape
            depth_flat = depth_features.view(batch * frames, channels, H, W)
            
            adapter = nn.Conv2d(channels, self.hidden_dim, 1)
            adapter = adapter.to(depth_features.device)
            adapted_flat = adapter(depth_flat)
            
            adapted_features = adapted_flat.view(batch, frames, self.hidden_dim, H, W)
            print(f"         ✅ 深度维度适配: {channels} -> {self.hidden_dim}")
        else:
            adapted_features = depth_features
            
        return adapted_features

    def adapt_lmttm_input(self, input_tensor):
        """适配LMTTM输入"""
        print(f"      🔧 适配LMTTM输入: {input_tensor.shape}")
        
        if input_tensor.dim() == 3:
            # [batch, tokens, features] -> [batch, 1, tokens, features]
            input_tensor = input_tensor.unsqueeze(1)
            print(f"         ✅ 3维转4维: {input_tensor.shape}")
        elif input_tensor.dim() == 4:
            # 检查形状 [batch, sequence, tokens, features]
            batch, seq, tokens, features = input_tensor.shape
            if features != self.hidden_dim:
                # 需要调整特征维度
                input_tensor = input_tensor.reshape(batch * seq * tokens, features)
                adapter = nn.Linear(features, self.hidden_dim)
                adapter = adapter.to(input_tensor.device)
                adapted = adapter(input_tensor)
                input_tensor = adapted.reshape(batch, seq, tokens, self.hidden_dim)
                print(f"         ✅ 特征维度适配: {features} -> {self.hidden_dim}")
        else:
            raise ValueError(f"不支持的LMTTM输入维度: {input_tensor.dim()}")
            
        return input_tensor

# 全局适配器实例
competition_adapter = CompetitionDimensionAdapter(256)