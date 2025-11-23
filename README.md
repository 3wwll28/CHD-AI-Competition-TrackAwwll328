# CHD-AI-Competition-TrackAwwll328
长安大学2025 AI大赛A赛道参赛代码
2025 VLP 挑战赛参赛作品
项目名称
基于单目视频的3D视觉语言跟踪系统

项目描述
本项目针对长安大学2025人工智能创新大赛赛道A（VLSOT）任务，实现了基于单目RGB视频的3D视觉语言跟踪系统。系统能够根据自然语言指令（如"跟踪画面中穿蓝色外套的行人"）在3D空间中实时定位目标的位置、姿态及运动轨迹。

核心特性
跨模态关联: 实现视觉与语言信息的有效融合

实时3D跟踪: 在单目视频中实现目标的3D空间定位

深度估计: 基于单目图像的深度信息恢复

时序一致性: 优化的逐帧处理保证跟踪稳定性

容错机制: 多重安全机制确保系统鲁棒性

纯词典文本处理: 不依赖外部BERT模型，使用优化的比赛专用词汇表

环境要求
硬件要求
GPU: NVIDIA GPU with at least 8GB VRAM (推荐)

RAM: 至少 16GB

存储空间: 至少 50GB 可用空间

软件要求
Python 3.8+

CUDA 11.3+ (如使用GPU)

快速安装
1. 克隆仓库
bash
git clone https://github.com/你的用户名/CHD-AI-Competition-TrackA.git
cd CHD-AI-Competition-TrackA
2. 安装依赖
bash
pip install -r requirements.txt
模型权重下载
请从以下链接下载预训练模型权重：

百度网盘

链接: [在此处插入你的百度网盘链接]

提取码: [在此处插入提取码]

下载后，请将权重文件放置在项目根目录下的 weights/ 文件夹中，或根据实际路径修改代码中的权重路径。

项目结构
text
CHD-AI-Competition-TrackA/
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
├── LICENSE                   # MIT开源许可证
├── demo.py                   # 🚀 主入口文件 - 推理演示脚本
├── 属性文件夹/               # 核心算法模块
│   ├── dimension_adapter.py  # 维度适配器（处理特征维度匹配）
│   ├── emergency_fix.py      # 紧急修复模块（处理运行时异常）
│   ├── frame_by_frame_inference.py # 逐帧推理处理器
│   ├── LMTTM.py              # 核心的Linked Memory Token Turing Machine模型
│   ├── model_wrapper.py      # 安全模型包装器
│   ├── mono3dvg_transformer.py # 主要的3D视觉语言Transformer
│   ├── Mono3DVGInference.py  # 🎯 端到端推理管道
│   ├── ops.py                # 多尺度可变形注意力机制
│   ├── output_formatter.py   # 比赛结果格式化器
│   ├── position_encoder.py   # 位置编码模块
│   ├── safe_lmttm.py         # 安全的LMTTM包装器
│   ├── text_adapter.py       # 📝 文本特征适配器（修复StopIteration错误）
│   ├── text_processor.py     # 📝 纯词典文本处理器（不依赖BERT）
│   └── ... (其他辅助文件)
└── weights/                  # 模型权重目录（需自行创建）
使用方法
🚀 快速开始 - 使用demo.py
运行以下命令进行单视频推理：

bash
python demo.py
程序将引导你：

输入视频文件路径

输入文本查询（如"跟踪画面中穿蓝色外套的行人"）

自动完成推理并生成结果文件 submission.json

📊 批量测试
如需进行批量测试，可以直接调用推理API：

python
from Mono3DVGInference import Mono3DVGInference

# 初始化推理管道
pipeline = Mono3DVGInference(device="auto")

# 运行推理
result = pipeline.predict(
    video_path="your_video.mp4",
    text_query="跟踪红色汽车",
    output_path="submission.json"
)
🎯 高级用法
python
# 自定义配置
from Mono3DVGInference import Mono3DVGInference

# 使用自定义配置
custom_config = {
    'hidden_dim': 256,
    'nheads': 8,
    'enc_layers': 6,
    'dec_layers': 6,
    # ... 其他配置
}

pipeline = Mono3DVGInference(
    model_config=custom_config,
    checkpoint_path="path/to/your/checkpoint.pth",
    device="cuda"  # 或 "cpu"
)
技术架构
核心模块说明
Mono3DVGInference - 主推理管道

集成视频处理、文本处理、3D推理全流程

提供逐帧处理和时序一致性优化

LMTTM (Linked Memory Token Turing Machine) - 记忆增强模型

处理时序信息的记忆机制

增强长序列跟踪的稳定性

Mono3DVGTransformer - 3D视觉语言Transformer

多尺度可变形注意力机制

跨模态特征融合

文本处理系统 - 优化的语言理解

text_processor.py: 纯词典文本编码器，不依赖外部BERT

text_adapter.py: 文本特征适配器，修复设备兼容性问题

专门优化的比赛词汇表，覆盖跟踪任务关键词

安全机制 - 系统鲁棒性保障

维度适配器：自动处理特征维度不匹配

紧急修复：处理运行时异常

备用方案：确保系统在各种情况下都能输出合理结果

文本处理系统特色
🆕 纯词典文本处理器 (text_processor.py)
不依赖BERT: 无需下载大型预训练模型

比赛专用词汇表: 专门为3D视觉跟踪任务优化的词汇

复合词识别: 支持"white car"、"moving from"等多词组合

设备兼容: 完全在PyTorch框架内运行

词汇表覆盖范围
跟踪动词: track, follow, find, locate, detect, monitor等

颜色描述: white, black, red, blue, green, yellow, silver等

车辆类型: car, van, truck, bus, motorcycle, bicycle等

空间位置: left, right, front, back, middle, corner等

运动状态: moving, parking, stopped, turning, accelerating等

🛠️ 文本适配器 (text_adapter.py)
设备兼容性: 自动处理CPU/GPU设备切换

维度适配: 确保文本特征与视觉特征维度匹配

容错机制: 处理StopIteration等异常情况


创新点
改进的跨模态注意力机制

更有效的视觉-语言特征对齐

自适应特征权重调整

优化的3D边界框回归策略

基于单目图像的精确3D定位

时序连续性约束

增强的记忆机制

LMTTM模块的长序列处理能力

动态记忆读写机制

独立的文本处理系统

纯词典方法，不依赖外部NLP模型

专门优化的比赛词汇表

更好的领域适应性

鲁棒的系统架构

多重容错机制

自动维度适配

优雅降级处理

注意事项
首次运行

确保已下载模型权重文件

检查CUDA环境配置（如使用GPU）

无需下载BERT等外部模型

输入要求

视频格式：支持MP4、AVI等常见格式

文本描述：应简洁明确，如"跟踪红色汽车"、"定位穿白色衣服的人"

支持中英文混合输入

输出格式

结果保存为JSON格式，符合比赛提交要求

包含完整的3D位置、边界框、车辆属性等信息

故障排除
常见问题
模块导入失败

bash
# 确保在项目根目录运行
python demo.py
CUDA内存不足

python
# 使用CPU模式
pipeline = Mono3DVGInference(device="cpu")
权重文件加载失败

检查权重文件路径是否正确

确认文件完整性

文本处理问题

系统使用纯词典方法，无需网络连接

如遇未知词汇，会自动处理为[UNK]标记

获取帮助
如遇问题，请：

检查控制台输出的错误信息

确认所有依赖包已正确安装

参考代码中的详细日志输出

联系方式
团队名称: 人工不智能

联系人: 吴书楠

邮箱: [2024904539@chd.edu.cn]

许可证
本项目采用 MIT License - 详见 LICENSE 文件
