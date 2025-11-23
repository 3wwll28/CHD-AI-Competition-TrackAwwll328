# text_processor.py - 纯扩大词典版本（无BERT）
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import math

class EnhancedTokenizer:
    """增强版词汇表，专门为3D视觉语言跟踪比赛优化"""
    
    def __init__(self):
        self.vocab = self._build_enhanced_vocab()
        self.unk_token_id = 1
        self.pad_token_id = 0
        self.cls_token_id = 2
        self.sep_token_id = 3
        
    def _build_enhanced_vocab(self):
        """构建专门为比赛优化的词汇表"""
        base_words = [
            # 特殊标记
            '[PAD]', '[UNK]', '[CLS]', '[SEP]',
            
            # 核心跟踪动词（比赛关键）
            'track', 'follow', 'find', 'locate', 'detect', 'identify', 'monitor',
            'trace', 'watch', 'observe', 'search', 'seek',
            
            # 颜色描述（从示例数据中提取）
            'white', 'black', 'red', 'blue', 'green', 'yellow', 'silver', 'grey',
            'gray', 'orange', 'purple', 'brown', 'pink', 'dark', 'light',
            
            # 车辆类型（从示例数据中提取）
            'car', 'vehicle', 'van', 'truck', 'bus', 'motorcycle', 'bicycle',
            'suv', 'sedan', 'minivan', 'pickup', 'ambulance', 'police', 'taxi',
            'jeep', 'limousine', 'tractor', 'trailer',
            
            # 车辆状态（从示例数据中提取）
            'moving', 'parking', 'stopped', 'running', 'driving', 'waiting',
            'accelerating', 'braking', 'turning', 'reversing', 'passing',
            'standing', 'stationary',
            
            # 空间位置和方向
            'left', 'right', 'front', 'back', 'rear', 'middle', 'side', 
            'center', 'top', 'bottom', 'upper', 'lower', 'corner', 'edge',
            'near', 'far', 'close', 'distant', 'adjacent', 'beside',
            'above', 'below', 'under', 'over', 'behind', 'ahead',
            
            # 相对位置描述
            'relative', 'position', 'location', 'orientation', 'direction',
            'facing', 'pointing', 'heading',
            
            # 道路和场景元素
            'road', 'street', 'highway', 'intersection', 'crosswalk', 'sidewalk',
            'parking', 'lot', 'garage', 'bridge', 'tunnel', 'roundabout',
            'lane', 'curb', 'shoulder',
            
            # 人物类型
            'person', 'pedestrian', 'cyclist', 'driver', 'rider', 'passenger',
            'walker', 'jogger', 'runner',
            
            # 时间和顺序
            'first', 'second', 'third', 'last', 'next', 'previous', 'current',
            'initial', 'final', 'beginning', 'end', 'start', 'finish',
            'before', 'after', 'during', 'while',
            
            # 尺寸和形状
            'small', 'large', 'big', 'tiny', 'huge', 'long', 'short', 'tall',
            'wide', 'narrow', 'heavy', 'light', 'size', 'dimension',
            'height', 'width', 'length', 'depth',
            
            # 运动描述
            'speed', 'velocity', 'fast', 'slow', 'quickly', 'slowly',
            'suddenly', 'gradually', 'steady', 'constant',
            'straight', 'curved', 'zigzag', 'circular',
            
            # 视觉特征
            'visible', 'invisible', 'occluded', 'truncated', 'clear', 'obscured',
            'bright', 'dark', 'shadow', 'reflection',
            
            # 常见介词和连接词
            'the', 'a', 'an', 'in', 'on', 'at', 'by', 'with', 'from', 'to',
            'and', 'or', 'but', 'while', 'when', 'where', 'which', 'that',
            'of', 'for', 'as', 'like',
            
            # 数字（0-99）
            *[str(i) for i in range(100)],
            
            # 常见量词
            'meter', 'meters', 'foot', 'feet', 'degree', 'degrees',
            'pixel', 'pixels', 'frame', 'frames',
            
            # 从示例描述中提取的关键词
            'distinctively', 'measuring', 'observed', 'starting', 'viewer',
            'position', 'azimuth', 'ranks', 'initially', 'unobstructed',
            'facing', 'positioned', 'similarly', 'continues', 'slightly',
            'closer', 'partially', 'truncated', 'towards', 'maintaining',
            'orientation', 'remains', 'nearest', 'direction',
        ]
        
        # 创建词汇表
        vocab = {word: idx for idx, word in enumerate(base_words)}
        
        print(f"📚 比赛专用词汇表构建完成，共 {len(vocab)} 个词条")
        return vocab
    
    def tokenize(self, text: str) -> List[str]:
        """改进的分词方法，支持复合词识别"""
        text = text.lower().strip()
        
        # 特殊处理：替换常见变体
        text = text.replace("'s", " 's")
        
        tokens = []
        words = text.split()
        
        i = 0
        while i < len(words):
            matched = False
            
            # 优先检查3词组合（如 "lower right corner"）
            if i + 2 < len(words):
                compound = words[i] + ' ' + words[i+1] + ' ' + words[i+2]
                if compound in self.vocab:
                    tokens.append(compound)
                    i += 3
                    matched = True
                    continue
            
            # 检查2词组合（如 "white car", "moving from"）
            if i + 1 < len(words) and not matched:
                compound = words[i] + ' ' + words[i+1]
                if compound in self.vocab:
                    tokens.append(compound)
                    i += 2
                    matched = True
                    continue
            
            # 单字词
            if not matched:
                # 清理单词（移除标点）
                word = words[i].strip('.,!?;:"\'()[]{}')
                if word:  # 确保不是空字符串
                    tokens.append(word)
                i += 1
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def __call__(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        tokens = self.tokenize(text)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # 填充或截断
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length-1] + [self.sep_token_id]
        else:
            input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))
        
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]
        
        return {
            'input_ids': torch.tensor([input_ids], dtype=torch.long),
            'attention_mask': torch.tensor([attention_mask], dtype=torch.long)
        }

class SimpleTextEncoder(nn.Module):
    """简单的文本编码器 - 仅使用嵌入层和池化"""
    
    def __init__(self, vocab_size, hidden_dim=256, max_seq_length=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # 词嵌入层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        
        # 位置编码
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 简单的投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # 词嵌入
        word_embeddings = self.word_embeddings(input_ids)
        
        # 位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 组合嵌入
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1)
            embeddings = embeddings * attention_mask
        
        # 平均池化
        if attention_mask is not None:
            sum_embeddings = torch.sum(embeddings * attention_mask, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            pooled_output = sum_embeddings / (sum_mask + 1e-9)
        else:
            pooled_output = torch.mean(embeddings, dim=1)
        
        # 投影到目标维度
        text_features = self.projection(pooled_output)
        
        return text_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]

class TextProcessor:
    """
    文本处理器 - 纯扩大词典版本
    输出格式: text_features [1, 1, hidden_dim], text_mask [1, seq_len]
    """
    
    def __init__(self, hidden_dim=256, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        
        print(f"🎯 初始化文本处理器（纯词典版本），使用设备: {self.device}")
        
        # 初始化增强tokenizer
        self.tokenizer = EnhancedTokenizer()
        
        # 构建简单的文本编码器
        vocab_size = len(self.tokenizer.vocab)
        self.model = SimpleTextEncoder(vocab_size, hidden_dim).to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"✅ 文本处理器初始化完成! 词汇表大小: {vocab_size}, 隐藏维度: {hidden_dim}")
    
    def encode_text(self, text: str):
        """
        编码文本
        返回: Dict包含 'features' 和 'mask'
        """
        with torch.no_grad():
            # 使用增强tokenizer
            inputs = self.tokenizer(text)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 前向传播
            text_features = self.model(input_ids, attention_mask)
            
            print(f"📝 文本特征提取完成:")
            print(f"   输入: '{text}'")
            print(f"   Token数量: {attention_mask.sum().item()}")
            print(f"   特征形状: {text_features.shape}")
            print(f"   掩码形状: {attention_mask.shape}")
            
            return {
                'features': text_features,  # [1, 1, hidden_dim]
                'mask': attention_mask      # [1, seq_len]
            }
    
    def __call__(self, text_query):
        """保持与原代码相同的调用方式"""
        return self.encode_text(text_query)

# 测试函数
def test_enhanced_processor():
    """测试增强文本处理器"""
    print("🧪 测试增强文本处理器（纯词典版本）")
    print("=" * 50)
    
    # 创建处理器
    processor = TextProcessor(hidden_dim=256)
    
    # 测试比赛相关的查询
    test_queries = [
        "track the white bus moving from lower right corner",
        "find the red car parking in the middle",
        "follow the black van turning left at intersection",
        "locate the silver vehicle facing front",
        "track white bus 2.5 meters height 7.3 meters length"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: '{query}'")
        result = processor.encode_text(query)
        
        print(f"✅ 输出格式:")
        print(f"   features 形状: {result['features'].shape}")
        print(f"   mask 形状: {result['mask'].shape}")
        
        # 统计未知词
        input_ids = processor.tokenizer(query)['input_ids'][0]
        unk_count = (input_ids == processor.tokenizer.unk_token_id).sum().item()
        print(f"   未知词数量: {unk_count}")
        
        # 验证维度
        assert result['features'].shape[2] == 256, f"特征维度应该是256，但得到{result['features'].shape[2]}"
        assert result['features'].shape[0] == 1, "batch维度应该是1"

if __name__ == "__main__":
    test_enhanced_processor()