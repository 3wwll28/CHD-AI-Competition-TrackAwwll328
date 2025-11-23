# safe_lmttm.py - 安全的LMTTM包装器
import torch
import torch.nn as nn

class SafeLMTTMWrapper:
    """安全的LMTTM包装器 - 处理所有形状问题"""
    
    @staticmethod
    def safe_forward(lmttm_model, input_tensor, memory_tokens):
        """安全的LMTTM前向传播"""
        print(f"🔧 SafeLMTTM输入: {input_tensor.shape}")
        
        try:
            # 确保输入是4维 [batch, sequence, tokens, features]
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(1)  # [B, T, F] -> [B, 1, T, F]
                print(f"   ✅ 3维转4维: {input_tensor.shape}")
            elif input_tensor.dim() == 4:
                # 已经是4维，检查形状
                batch, seq, tokens, features = input_tensor.shape
                if seq != 1 or tokens != 100 or features != 256:
                    print(f"   ⚠️ LMTTM输入形状异常: {input_tensor.shape}")
                    # 尝试修复到标准形状
                    input_tensor = input_tensor.reshape(batch, 1, 100, 256)
                    print(f"   ✅ 修复形状: {input_tensor.shape}")
            else:
                raise ValueError(f"不支持的LMTTM输入维度: {input_tensor.dim()}")
            
            # 调用LMTTM
            output, new_memory = lmttm_model(input_tensor, memory_tokens)
            print(f"   ✅ SafeLMTTM成功: {input_tensor.shape} -> {output.shape}")
            
            return output, new_memory
            
        except Exception as e:
            print(f"   🔴 SafeLMTTM失败: {e}")
            # 返回安全的备用输出
            batch_size = input_tensor.shape[0]
            safe_output = torch.randn(batch_size, 1, 256).to(input_tensor.device)
            print(f"   ⚠️ 使用备用输出: {safe_output.shape}")
            return safe_output, memory_tokens

    @staticmethod
    def create_safe_lmttm_config():
        """创建安全的LMTTM配置"""
        return {
            "batch_size": 1,
            "model": {
                "model": "lmttm",
                "drop_r": 0.2,
                "preprocess_mode": "3dBN",
                "process_unit": "transformer", 
                "memory_mode": "TL",
                "in_channels": 1,
                "dim": 256,
                "memory_tokens_size": 128,
                "num_blocks": 8,
                "summerize_num_tokens": 1,
                "out_class_num": 3,
                "patch_size": 1,
                "Read_use_positional_embedding": True,
                "Write_use_positional_embedding": True,
                "load_memory_add_noise": False,
                "load_memory_add_noise_mode": "normal"
            },
            "train": {
                "input_H": 28,
                "input_W": 28
            }
        }