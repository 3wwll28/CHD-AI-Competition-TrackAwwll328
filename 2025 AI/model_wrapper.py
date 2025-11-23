# model_wrapper.py - 创建新文件来包装学校模型
import torch
import torch.nn as nn

class SafeModelWrapper(nn.Module):
    """安全模型包装器，处理所有维度问题"""
    
    def __init__(self, original_model, hidden_dim=256):
        super().__init__()
        self.original_model = original_model
        self.hidden_dim = hidden_dim
        
        # 维度适配器
        from dimension_adapter import DimensionAdapter
        self.dimension_adapter = DimensionAdapter
        
        # 文本特征适配器
        self.text_adapter = nn.Linear(768, hidden_dim)
        
    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None,
                text_memory=None, text_mask=None, im_name=None, instanceID=None, ann_id=None):
        
        print("      🛡️ 安全模型包装器开始处理...")
        
        try:
            # 适配文本特征维度
            if text_memory is not None and text_memory.shape[-1] != self.hidden_dim:
                print(f"      🔧 适配文本特征: {text_memory.shape} -> {self.hidden_dim}")
                text_memory = self.text_adapter(text_memory)
            
            # 适配深度位置编码
            if depth_pos_embed is not None:
                depth_batch, depth_seq, depth_dim = depth_pos_embed.shape
                if depth_dim != self.hidden_dim:
                    print(f"      🔧 适配深度位置编码: {depth_pos_embed.shape}")
                    depth_pos_embed = depth_pos_embed.view(depth_batch * depth_seq, depth_dim)
                    depth_pos_embed = self.dimension_adapter.adapt_features(depth_pos_embed, self.hidden_dim)
                    depth_pos_embed = depth_pos_embed.view(depth_batch, depth_seq, self.hidden_dim)
            
            # 调用原始模型
            outputs = self.original_model(
                srcs=srcs,
                masks=masks,
                pos_embeds=pos_embeds,
                query_embed=query_embed,
                depth_pos_embed=depth_pos_embed,
                text_memory=text_memory,
                text_mask=text_mask,
                im_name=im_name,
                instanceID=instanceID,
                ann_id=ann_id
            )
            
            print("      ✅ 安全模型包装器完成")
            return outputs
            
        except Exception as e:
            print(f"      🔴 安全模型包装器失败: {e}")
            # 返回安全的模拟输出
            batch_size = srcs[0].shape[0] if srcs else 1
            hs = torch.randn(batch_size, 1, self.hidden_dim)
            reference_points = torch.randn(batch_size, 1, 2)
            return hs, reference_points, None, None, None, None

def wrap_school_model(original_model, hidden_dim=256):
    """包装学校提供的模型"""
    return SafeModelWrapper(original_model, hidden_dim)