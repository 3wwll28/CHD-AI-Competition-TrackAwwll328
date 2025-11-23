import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from PIL import Image
import imageio
import os
from datetime import datetime
import subprocess
import sys

class VideoFeatureExtractor:
    """
    视频特征提取管道（无OpenCV依赖）
    整合：视频抽帧 + 视觉特征提取 + 深度特征提取
    """
    
    def __init__(self, target_size=(224, 224), num_frames=30, hidden_dim=256, device=None):
        # 设备设置
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_size = target_size
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        print(f"初始化视频特征提取器，使用设备: {self.device}")
        
        # 初始化所有模块
        self._init_components()
        
        # 设置为评估模式
        self.set_eval_mode()
    
    def _init_components(self):
        """初始化所有组件"""
        # 1. 视频抽帧模块
        self.frame_extractor = VideoFrameExtractor(
            target_size=self.target_size, 
            num_frames=self.num_frames
        )
        
        # 2. 视觉骨干网络
        self.visual_backbone = VisualBackbone(
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # 3. 深度预测器
        self.depth_predictor = DepthPredictor(
            hidden_dim=self.hidden_dim
        ).to(self.device)
    
    def set_eval_mode(self):
        """设置为评估模式"""
        self.visual_backbone.eval()
        self.depth_predictor.eval()
        print("✅ 特征提取模块设置为评估模式")
    
    def extract_features(self, video_path):
        """
        从视频中提取特征
        返回: (视觉特征, 深度特征, 原始帧)
        """
        print("=" * 50)
        print("🎬 开始提取视频特征...")
        print(f"📁 视频路径: {video_path}")
        
        with torch.no_grad():
            try:
                # 🎯 步骤1: 视频抽帧
                print("\n1️⃣  视频抽帧...")
                frames = self.frame_extractor.extract_frames(video_path)
                frames = frames.to(self.device)
                print(f"   📊 帧数据形状: {frames.shape}")
                
                # 🎯 步骤2: 视觉特征提取
                print("\n2️⃣  视觉特征提取...")
                visual_features = self.visual_backbone(frames)
                print(f"   📊 多尺度特征数量: {len(visual_features)}")
                for i, feature in enumerate(visual_features):
                    print(f"     尺度 {i+1}: {feature.shape}")
                
                # 🎯 步骤3: 深度特征提取
                print("\n3️⃣  深度特征提取...")
                depth_features = self.depth_predictor(frames)
                print(f"   📊 深度特征形状: {depth_features.shape}")
                
                print("✅ 特征提取完成!")
                
                # 返回所有特征和原始帧
                return {
                    'visual_features': visual_features,  # 多尺度视觉特征列表
                    'depth_features': depth_features,    # 深度特征
                    'original_frames': frames,           # 原始帧数据
                    'num_frames': frames.shape[1]        # 帧数
                }
                
            except Exception as e:
                print(f"❌ 特征提取失败: {e}")
                raise


class VideoFrameExtractor:
    def __init__(self, target_size=(224, 224), num_frames=30):
        self.target_size = target_size
        self.num_frames = num_frames
    
    def extract_frames(self, video_path):
        """从视频中提取帧序列（使用imageio而不是OpenCV）"""
        try:
            print(f"🔍 检查视频文件: {video_path}")
            
            # 详细检查文件是否存在和权限
            if not os.path.exists(video_path):
                # 尝试自动修复双扩展名问题
                fixed_path = self._fix_double_extension(video_path)
                if fixed_path and os.path.exists(fixed_path):
                    print(f"🔄 自动修复路径: {fixed_path}")
                    video_path = fixed_path
                else:
                    raise ValueError(f"❌ 视频文件不存在: {video_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            print(f"   📏 文件大小: {file_size / (1024*1024):.2f} MB")
            
            if file_size == 0:
                raise ValueError("❌ 视频文件为空")
            
            # 检查文件权限
            if not os.access(video_path, os.R_OK):
                raise ValueError("❌ 没有读取视频文件的权限")
            
            print("   ✅ 视频文件检查通过")
            
            # 尝试使用imageio读取视频
            print("   🔄 尝试使用imageio读取视频...")
            
            try:
                reader = imageio.get_reader(video_path)
                metadata = reader.get_meta_data()
                
                total_frames = reader.count_frames()
                fps = metadata.get('fps', 30)
                duration = metadata.get('duration', total_frames / fps if fps > 0 else 0)
                
                print(f"   📹 视频信息:")
                print(f"      总帧数: {total_frames}")
                print(f"      FPS: {fps:.2f}")
                print(f"      时长: {duration:.2f}秒")
                print(f"      尺寸: {metadata.get('source_size', '未知')}")
                
                # 均匀采样帧
                frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
                frames = []
                
                successful_frames = 0
                for idx in frame_indices:
                    try:
                        # 读取帧
                        frame = reader.get_data(idx)
                        
                        # 转换为PIL图像进行处理
                        pil_image = Image.fromarray(frame)
                        
                        # 调整尺寸 - 兼容不同Pillow版本
                        try:
                            # 尝试使用新版本的Resampling
                            pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
                        except AttributeError:
                            # 回退到旧版本常量
                            pil_image = pil_image.resize(self.target_size, Image.LANCZOS)
                        
                        # 转换为numpy数组并归一化
                        frame_array = np.array(pil_image, dtype=np.float32) / 255.0
                        
                        # 转换为CHW格式 [3, H, W]
                        frame_array = np.transpose(frame_array, (2, 0, 1))
                        
                        frames.append(frame_array)
                        successful_frames += 1
                        
                    except (IndexError, Exception) as e:
                        print(f"   ⚠️  读取帧 {idx} 失败: {e}")
                        # 用黑帧填充
                        black_frame = np.zeros((3, *self.target_size), dtype=np.float32)
                        frames.append(black_frame)
                
                reader.close()
                
                print(f"   ✅ 成功读取 {successful_frames}/{self.num_frames} 帧")
                
                # 转换为 [1, num_frames, 3, H, W]
                frames_tensor = torch.tensor(np.array(frames)).unsqueeze(0)
                return frames_tensor
                
            except Exception as e:
                print(f"   ❌ imageio读取失败: {e}")
                # 尝试备选方法
                return self._extract_frames_alternative(video_path)
            
        except Exception as e:
            raise ValueError(f"视频读取失败: {e}")
    
    def _fix_double_extension(self, video_path):
        """尝试修复双扩展名问题"""
        # 检查是否是双扩展名问题
        basename = os.path.basename(video_path)
        if basename.count('.') > 1:
            # 尝试移除重复的扩展名
            name_parts = basename.split('.')
            # 保留文件名和最后一个扩展名
            fixed_name = '.'.join(name_parts[:-2]) + '.' + name_parts[-1]
            fixed_path = os.path.join(os.path.dirname(video_path), fixed_name)
            
            # 同时尝试其他可能的修复
            possible_fixes = [
                fixed_path,
                video_path.replace('.mp4.mp4', '.mp4'),  # 直接替换双扩展名
                os.path.join(os.path.dirname(video_path), 'test_video.mp4')  # 尝试简单名称
            ]
            
            for fix in possible_fixes:
                if os.path.exists(fix):
                    return fix
        
        return None
    
    def _extract_frames_alternative(self, video_path):
        """备选视频读取方法"""
        print("   🔄 尝试备选视频读取方法...")
        
        try:
            # 方法1: 使用imageio v3 API (如果可用)
            try:
                import imageio.v3 as iio
                frames = iio.imread(video_path, index=None)  # 读取所有帧
                print(f"   ✅ 使用imageio v3 API成功读取 {len(frames)} 帧")
                
                # 均匀采样
                frame_indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
                sampled_frames = []
                
                for idx in frame_indices:
                    frame = frames[idx]
                    pil_image = Image.fromarray(frame)
                    
                    # 调整尺寸
                    try:
                        pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
                    except AttributeError:
                        pil_image = pil_image.resize(self.target_size, Image.LANCZOS)
                    
                    frame_array = np.array(pil_image, dtype=np.float32) / 255.0
                    frame_array = np.transpose(frame_array, (2, 0, 1))
                    sampled_frames.append(frame_array)
                
                frames_tensor = torch.tensor(np.array(sampled_frames)).unsqueeze(0)
                return frames_tensor
                
            except Exception as e:
                print(f"   ❌ imageio v3 API也失败: {e}")
                
            # 方法2: 检查视频编码并尝试转换
            print("   🔄 检查视频编码信息...")
            self._check_video_codec(video_path)
            
            raise ValueError("所有视频读取方法都失败了")
            
        except Exception as e:
            raise ValueError(f"备选视频读取方法失败: {e}")
    
    def _check_video_codec(self, video_path):
        """检查视频编码信息"""
        try:
            # 使用FFmpeg检查视频信息（如果可用）
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name,width,height,r_frame_rate,duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                info = result.stdout.strip().split('\n')
                print(f"   📊 FFprobe视频信息:")
                print(f"      编码器: {info[0] if len(info) > 0 else '未知'}")
                print(f"      宽度: {info[1] if len(info) > 1 else '未知'}")
                print(f"      高度: {info[2] if len(info) > 2 else '未知'}")
                print(f"      帧率: {info[3] if len(info) > 3 else '未知'}")
                print(f"      时长: {info[4] if len(info) > 4 else '未知'}")
            else:
                print("   ℹ️  FFprobe不可用或视频格式不支持")
                
        except Exception as e:
            print(f"   ℹ️  无法获取视频编码信息: {e}")


class VisualBackbone(nn.Module):
    def __init__(self, backbone_type="resnet50", hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 使用ResNet50 - 兼容不同torchvision版本
        try:
            # 优先尝试新版本API
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            resnet = models.resnet50(weights=weights)
            print("   🔧 使用新版本torchvision API")
        except AttributeError:
            # 回退到旧版本API
            try:
                resnet = models.resnet50(pretrained=True)
                print("   🔧 使用旧版本torchvision API")
            except:
                # 最终备用方案
                resnet = models.resnet50()
                print("   ⚠️  使用无预训练权重的ResNet50")
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8  
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32
        
        # 特征金字塔网络
        self.fpn = self._build_fpn([256, 512, 1024, 2048], hidden_dim)
    
    def _build_fpn(self, in_channels_list, out_channels):
        layers = nn.ModuleDict()
        for i, in_channels in enumerate(in_channels_list):
            layers[f'c{i+1}'] = nn.Conv2d(in_channels, out_channels, 1)
        return layers
    
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        all_frame_features = []
        
        for t in range(num_frames):
            frame = x[:, t]  # [batch_size, 3, H, W]
            
            # 特征提取
            c1 = self.relu(self.bn1(self.conv1(frame)))
            c1 = self.maxpool(c1)
            
            c2 = self.layer1(c1)  # [batch, 256, H/4, W/4]
            c3 = self.layer2(c2)  # [batch, 512, H/8, W/8]
            c4 = self.layer3(c3)  # [batch, 1024, H/16, W/16]
            c5 = self.layer4(c4)  # [batch, 2048, H/32, W/32]
            
            # FPN统一维度
            p2 = self.fpn['c1'](c2)  # [batch, hidden_dim, H/4, W/4]
            p3 = self.fpn['c2'](c3)  # [batch, hidden_dim, H/8, W/8]
            p4 = self.fpn['c3'](c4)  # [batch, hidden_dim, H/16, W/16]
            p5 = self.fpn['c4'](c5)  # [batch, hidden_dim, H/32, W/32]
            
            frame_features = [p2, p3, p4, p5]
            all_frame_features.append(frame_features)
        
        # 重组为多尺度特征
        multi_scale_features = []
        for scale_idx in range(4):
            scale_features = []
            for t in range(num_frames):
                scale_features.append(all_frame_features[t][scale_idx])
            scale_tensor = torch.stack(scale_features, dim=1)
            multi_scale_features.append(scale_tensor)
        
        return multi_scale_features


class DepthPredictor(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, hidden_dim, 3, padding=1),
            nn.ReLU()
        )
        
        self.adapter = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    def forward(self, x):
        batch_size, num_frames = x.shape[0], x.shape[1]
        depth_features = []
        
        for t in range(num_frames):
            frame = x[:, t]  # [batch_size, 3, H, W]
            depth_feat = self.encoder(frame)  # [batch_size, hidden_dim, H/8, W/8]
            depth_feat = self.adapter(depth_feat)
            depth_features.append(depth_feat)
        
        depth_tensor = torch.stack(depth_features, dim=1)
        return depth_tensor


# ============================================================================
# 主程序 - 针对您的具体文件路径
# ============================================================================

def main():
    # 使用您提供的准确文件路径
    video_path = r"C:\Users\lenovo\Desktop\人工智能\test_video.mp4.mp4"
    
    print("🚀 视频特征提取器 - 针对双扩展名文件")
    print("=" * 60)
    print(f"🎯 目标视频: {video_path}")
    print("=" * 60)
    
    # 详细的环境检查
    print("\n🔍 环境检查:")
    print(f"  Python版本: {sys.version}")
    print(f"  工作目录: {os.getcwd()}")
    
    # 检查依赖库版本
    try:
        import imageio
        print(f"  imageio版本: {imageio.__version__}")
    except:
        print("  ❌ imageio未正确安装")
    
    try:
        from PIL import Image
        print(f"  Pillow版本: {Image.__version__}")
    except:
        print("  ❌ Pillow未正确安装")
    
    # 详细检查视频文件
    print(f"\n📁 视频文件检查:")
    if os.path.exists(video_path):
        print(f"  ✅ 文件存在: {video_path}")
        
        # 获取文件信息
        file_stats = os.stat(video_path)
        file_size = file_stats.st_size
        print(f"  📏 文件大小: {file_size} 字节 ({file_size / (1024*1024):.2f} MB)")
        
        # 检查文件权限
        if os.access(video_path, os.R_OK):
            print("  ✅ 有读取权限")
        else:
            print("  ❌ 没有读取权限")
            
        # 检查文件扩展名
        _, ext = os.path.splitext(video_path)
        print(f"  📄 文件扩展名: {ext}")
        
        # 检查是否是双扩展名
        basename = os.path.basename(video_path)
        if basename.count('.') > 1:
            print(f"  ⚠️  检测到双扩展名: {basename}")
            print(f"  💡 建议重命名为: {basename.replace('.mp4.mp4', '.mp4')}")
        
        # 支持的视频格式
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        if ext.lower() in supported_formats:
            print(f"  ✅ 文件格式支持")
        else:
            print(f"  ⚠️  文件格式可能不支持，支持的格式: {', '.join(supported_formats)}")
        
    else:
        print(f"  ❌ 文件不存在: {video_path}")
        print("  💡 尝试自动查找可能的文件...")
        
        # 尝试查找可能的文件
        video_dir = r"C:\Users\lenovo\Desktop\人工智能"
        if os.path.exists(video_dir):
            print(f"  📂 扫描目录: {video_dir}")
            for file in os.listdir(video_dir):
                if 'test_video' in file.lower() and any(file.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov']):
                    print(f"  🔍 找到可能的目标文件: {file}")
                    full_path = os.path.join(video_dir, file)
                    print(f"  💡 尝试使用: {full_path}")
                    video_path = full_path
                    break
        
        if not os.path.exists(video_path):
            print("  ❌ 未找到合适的视频文件")
            return
    
    # 尝试初始化特征提取器
    try:
        print(f"\n🎯 初始化视频特征提取器...")
        feature_extractor = VideoFeatureExtractor(
            target_size=(224, 224),
            num_frames=30,
            hidden_dim=256,
            device='cpu'
        )
        
        # 尝试提取特征
        print(f"\n🎬 开始提取视频特征...")
        features = feature_extractor.extract_features(video_path)
        
        # 打印提取结果摘要
        print("\n📊 特征提取摘要:")
        print(f"   视频文件: {os.path.basename(video_path)}")
        print(f"   视频帧数: {features['num_frames']}")
        print(f"   视觉特征尺度数: {len(features['visual_features'])}")
        print(f"   深度特征形状: {features['depth_features'].shape}")
        
        # 自动保存特征
        save_features(features, video_path)
        
        print("\n✅ 特征提取完成并已保存!")
        
    except Exception as e:
        print(f"\n❌ 特征提取失败: {e}")
        print(f"\n💡 解决方案建议:")
        print("1. 重命名文件，移除重复的扩展名")
        print("2. 检查视频文件是否损坏")
        print("3. 安装FFmpeg: pip install imageio-ffmpeg")
        print("4. 尝试使用其他视频文件")


def save_features(features, video_path):
    """保存提取的特征到文件"""
    try:
        # 创建保存目录
        output_dir = "extracted_features"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 如果有多重扩展名，只取第一部分作为名称
        if '.' in video_name:
            video_name = video_name.split('.')[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{video_name}_features_{timestamp}.pt"
        filepath = os.path.join(output_dir, filename)
        
        # 保存特征
        torch.save(features, filepath)
        print(f"💾 特征已保存到: {filepath}")
        
    except Exception as e:
        print(f"❌ 特征保存失败: {e}")


if __name__ == "__main__":
    main()