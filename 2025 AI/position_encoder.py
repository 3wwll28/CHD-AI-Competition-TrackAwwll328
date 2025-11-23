# position_encoder.py - 创建新文件专门处理位置编码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SafePositionEncoder:
    """安全位置编码器，避免维度不匹配"""
    
    @staticmethod
    def create_2d_positional_encoding(height, width, channels, device):
        """创建2D位置编码 - 安全版本"""
        try:
            print(f"      📐 创建位置编码: H={height}, W={width}, C={channels}")
            
            # 方法1: 使用简单的网格位置编码
            if channels == 2:
                # 对于2通道，直接创建xy网格
                y_embed = torch.arange(height, dtype=torch.float32, device=device).view(-1, 1).repeat(1, width)
                x_embed = torch.arange(width, dtype=torch.float32, device=device).view(1, -1).repeat(height, 1)
                
                # 归一化
                y_embed = y_embed / (height - 1) if height > 1 else y_embed
                x_embed = x_embed / (width - 1) if width > 1 else x_embed
                
                pos_encoding = torch.stack([x_embed, y_embed], dim=0)  # [2, H, W]
                return pos_encoding
                
            else:
                # 对于更多通道，使用正弦编码
                return SafePositionEncoder._create_sine_position_encoding(height, width, channels, device)
                
        except Exception as e:
            print(f"      🔴 位置编码创建失败: {e}")
            # 返回随机编码作为备用
            return torch.randn(channels, height, width, device=device)
    
    @staticmethod
    def _create_sine_position_encoding(height, width, channels, device):
        """创建正弦位置编码"""
        # 确保通道数是偶数
        if channels % 2 != 0:
            channels += 1
            
        # 创建位置网格
        y_pos = torch.arange(height, dtype=torch.float32, device=device)
        x_pos = torch.arange(width, dtype=torch.float32, device=device)
        
        # 归一化
        y_pos = y_pos / height * 2 * math.pi
        x_pos = x_pos / width * 2 * math.pi
        
        # 创建正弦编码
        pos_encoding = []
        for i in range(channels // 2):
            freq = 2 ** i
            y_sin = torch.sin(y_pos * freq).unsqueeze(1).repeat(1, width)
            y_cos = torch.cos(y_pos * freq).unsqueeze(1).repeat(1, width)
            x_sin = torch.sin(x_pos * freq).unsqueeze(0).repeat(height, 1)
            x_cos = torch.cos(x_pos * freq).unsqueeze(0).repeat(height, 1)
            
            pos_encoding.extend([y_sin, y_cos, x_sin, x_cos])
        
        # 堆叠并截断到目标通道数
        pos_encoding = torch.stack(pos_encoding[:channels], dim=0)
        return pos_encoding

class SimplePositionalEncoding(nn.Module):
    """简单的位置编码模块"""
    
    def __init__(self, max_size=100, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.max_size = max_size
        
        # 预计算位置编码
        self.register_buffer('position_table', self._get_sine_encoding_table(max_size, d_model))
    
    def _get_sine_encoding_table(self, max_size, d_model):
        """正弦位置编码表"""
        position = torch.arange(max_size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_size, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        """添加位置编码"""
        seq_len = x.size(1)
        if seq_len <= self.max_size:
            pos_encoding = self.position_table[:seq_len].unsqueeze(0)
            return x + pos_encoding
        else:
            # 动态计算
            position = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float().to(x.device) * (-math.log(10000.0) / self.d_model))
            
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return x + pe.unsqueeze(0)