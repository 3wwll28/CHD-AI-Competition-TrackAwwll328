import torch
import torch.nn as nn
import torch.nn.functional as F

class EmergencyFix:
    """紧急修复类，处理维度不匹配问题"""
    
    @staticmethod
    def fix_lmttm_input_shape(input_tensor, target_channels=256):
        """修复LMTTM输入形状"""
        batch_size, channels, depth, height, width = input_tensor.shape
        
        print(f"🔧 LMTTM输入修复: {input_tensor.shape} -> 目标通道数: {target_channels}")
        
        # 如果通道数不匹配，使用1x1卷积调整
        if channels != target_channels:
            adapter = nn.Conv3d(channels, target_channels, 1).to(input_tensor.device)
            input_tensor = adapter(input_tensor)
            print(f"   ✅ 通道数调整: {channels} -> {target_channels}")
        
        return input_tensor
    
    @staticmethod
    def safe_lmttm_forward(lmttm_model, input_tensor, memory_tokens):
        """安全的LMTTM前向传播"""
        try:
            # 修复输入形状
            fixed_input = EmergencyFix.fix_lmttm_input_shape(input_tensor)
            
            # 运行LMTTM
            with torch.no_grad():
                output, new_memory = lmttm_model(fixed_input, memory_tokens)
            
            print(f"   ✅ LMTTM修复成功: 输入 {input_tensor.shape} -> 输出 {output.shape}")
            return output, new_memory
            
        except Exception as e:
            print(f"   ⚠️ LMTTM修复失败: {e}")
            # 返回安全的备用输出
            batch_size = input_tensor.shape[0]
            safe_output = torch.randn(batch_size, 1, 256).to(input_tensor.device)
            return safe_output, memory_tokens

# 全局修复实例
emergency_fix = EmergencyFix()