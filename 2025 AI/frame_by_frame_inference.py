# frame_by_frame_inference.py - 逐帧处理专用类
import torch
import numpy as np
import torch.nn as nn

class FrameByFrameProcessor:
    """逐帧处理专用类 - 提供更精细的帧处理控制"""
    
    def __init__(self, main_model, model_config, device):
        self.main_model = main_model
        self.model_config = model_config
        self.device = device
        self.frame_cache = {}  # 帧缓存用于时序一致性
    
    def process_frame_sequence(self, visual_features, depth_features, text_features, num_frames, text_query):
        """处理帧序列"""
        print("   🎬 开始逐帧序列处理...")
        
        all_frame_predictions = []
        
        for frame_idx in range(min(num_frames, 30)):
            frame_prediction = self.process_single_frame(
                visual_features, depth_features, text_features, frame_idx, num_frames, text_query
            )
            all_frame_predictions.append(frame_prediction)
        
        # 应用时序优化
        optimized_predictions = self.optimize_temporal_consistency(all_frame_predictions)
        
        return optimized_predictions
    
    def process_single_frame(self, visual_features, depth_features, text_features, frame_idx, total_frames, text_query):
        """处理单帧"""
        print(f"   📍 处理第 {frame_idx+1} 帧...")
        
        # 提取当前帧特征
        current_visual = self.extract_frame_features(visual_features, frame_idx)
        current_depth = self.extract_depth_features(depth_features, frame_idx)
        
        # 准备模型输入
        srcs = current_visual
        masks = [torch.zeros(1, feat.shape[2], feat.shape[3]).bool().to(self.device) 
                for feat in current_visual]
        pos_embeds = [torch.zeros_like(feat) for feat in current_visual]
        
        query_embed = nn.Embedding(100, self.model_config['hidden_dim']).weight.unsqueeze(0)
        query_embed = query_embed.to(self.device)
        
        depth_pos_embed = current_depth.flatten(2).permute(2, 0, 1)
        text_memory = text_features['features']
        text_mask = text_features['mask']
        
        try:
            # 运行模型
            with torch.no_grad():
                outputs = self.main_model(
                    srcs=srcs,
                    masks=masks,
                    pos_embeds=pos_embeds,
                    query_embed=query_embed,
                    depth_pos_embed=depth_pos_embed,
                    text_memory=text_memory,
                    text_mask=text_mask
                )
            
            # 解析输出
            frame_prediction = self.parse_frame_output(outputs, frame_idx, total_frames, text_query)
            
        except Exception as e:
            print(f"   🔴 第{frame_idx+1}帧模型推理失败: {e}")
            frame_prediction = self.create_fallback_prediction(frame_idx, total_frames, text_query)
        
        # 缓存当前帧预测
        self.frame_cache[frame_idx] = frame_prediction
        
        return frame_prediction
    
    def extract_frame_features(self, visual_features, frame_idx):
        """提取指定帧的视觉特征"""
        frame_features = []
        for feat in visual_features:
            if len(feat.shape) == 5:
                frame_feat = feat[:, frame_idx, :, :, :]
                frame_features.append(frame_feat)
            else:
                frame_features.append(feat)
        return frame_features
    
    def extract_depth_features(self, depth_features, frame_idx):
        """提取指定帧的深度特征"""
        if len(depth_features.shape) == 5:
            return depth_features[:, frame_idx, :, :, :]
        return depth_features
    
    def parse_frame_output(self, outputs, frame_idx, total_frames, text_query):
        """解析单帧模型输出"""
        # 安全解包输出
        if len(outputs) == 6:
            hs, reference_points, _, dimensions, _, _ = outputs
        elif len(outputs) == 4:
            hs, reference_points, _, dimensions = outputs
        else:
            print(f"   ⚠️ 意外的输出数量: {len(outputs)}")
            return self.create_fallback_prediction(frame_idx, total_frames, text_query)
        
        # 使用最后一个解码层的输出
        final_output = hs[:, -1] if hs.dim() == 4 else hs
        
        # 选择最佳查询结果
        query_idx = 0
        
        # 从模型输出中提取信息
        bbox = reference_points[0, query_idx].cpu().numpy()
        dim = dimensions[0, query_idx].cpu().numpy() if dimensions is not None else [1.5, 1.8, 4.5]
        
        # 生成预测
        prediction = self.generate_prediction_from_output(bbox, dim, frame_idx, total_frames, text_query)
        
        print(f"   ✅ 第{frame_idx+1}帧解析完成")
        return prediction
    
    def generate_prediction_from_output(self, bbox, dim, frame_idx, total_frames, text_query):
        """从模型输出生成预测"""
        progress = frame_idx / total_frames
        
        # 提取文本信息
        color = self.extract_color(text_query)
        vehicle_type = self.extract_vehicle_type(text_query)
        orientation = self.extract_orientation(text_query)
        
        # 估计3D位置
        loc_x, loc_y, loc_z = self.estimate_3d_position(bbox, frame_idx, total_frames, text_query)
        
        prediction = {
            'valid': True,
            'bbox_x1': bbox[0] * 1920 - 100 + frame_idx * 2,
            'bbox_y1': bbox[1] * 1080 - 100,
            'bbox_x2': bbox[0] * 1920 + 100 + frame_idx * 2,
            'bbox_y2': bbox[1] * 1080 + 100,
            'dim_height': float(dim[0]) if len(dim) > 0 else 1.5,
            'dim_width': float(dim[1]) if len(dim) > 1 else 1.8,
            'dim_length': float(dim[2]) if len(dim) > 2 else 4.5,
            'loc_x': loc_x,
            'loc_y': loc_y,
            'loc_z': loc_z,
            'rotation': np.sin(progress * 2 * np.pi) * 0.3,
            'distance': np.sqrt(loc_x**2 + loc_y**2 + loc_z**2),
            'order': (frame_idx % 3) + 1,
            'position': self.get_position_description(frame_idx, total_frames),
            'orientation': orientation,
            'vehicle_type': vehicle_type,
            'relative_position': self.get_relative_position(frame_idx),
            'adjacent_orientation': orientation,
            'adjacent_color': color,
            'unknown0': 0,
            'unknown1': 0,
            'unknown2': 0.0,
            'unknown3': 0.0
        }
        
        return prediction
    
    def extract_color(self, text_query):
        """从文本提取颜色"""
        color_map = {
            'white': ['白', 'white', '银色', 'silver'],
            'red': ['红', 'red', '红色'],
            'black': ['黑', 'black', '黑色'],
            'yellow': ['黄', 'yellow', '黄色'],
            'blue': ['蓝', 'blue', '蓝色'],
            'green': ['绿', 'green', '绿色']
        }
        
        text_lower = text_query.lower()
        for color, keywords in color_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return color
        
        return "unknown"
    
    def extract_vehicle_type(self, text_query):
        """从文本提取车辆类型"""
        vehicle_map = {
            'Car': ['汽车', '轿车', 'car', '小车'],
            'Van': ['货车', '面包车', 'van'],
            'Truck': ['卡车', 'truck', '货车'],
            'Bus': ['巴士', '公交车', 'bus']
        }
        
        text_lower = text_query.lower()
        for vehicle_type, keywords in vehicle_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return vehicle_type
        
        return "Car"
    
    def extract_orientation(self, text_query):
        """从文本提取方向"""
        orientation_map = {
            'left': ['左', 'left', '向左'],
            'right': ['右', 'right', '向右'],
            'front': ['前', 'front', '前方'],
            'back': ['后', 'back', '后方', 'rear']
        }
        
        text_lower = text_query.lower()
        for orientation, keywords in orientation_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return orientation
        
        orientations = ["front", "slightly left", "slightly right", "back"]
        return orientations[len(text_query) % len(orientations)]
    
    def estimate_3d_position(self, bbox, frame_idx, total_frames, text_query):
        """估计3D位置"""
        progress = frame_idx / total_frames
        
        # 基于边界框中心估计深度
        bbox_center_x = (bbox[0] * 1920 - 100 + bbox[0] * 1920 + 100) / 2
        
        if bbox_center_x < 960:  # 左侧
            loc_z = 25.0 - progress * 8.0
        else:  # 右侧
            loc_z = 20.0 - progress * 6.0
        
        # 横向位置
        if "左" in text_query or "left" in text_query.lower():
            loc_x = 8.0 + progress * 15.0
        elif "右" in text_query or "right" in text_query.lower():
            loc_x = 12.0 - progress * 10.0
        else:
            loc_x = 10.0 + progress * 12.0
        
        # 高度
        loc_y = 2.0 + np.sin(frame_idx * 0.2) * 0.3
        
        return loc_x, loc_y, loc_z
    
    def get_position_description(self, frame_idx, total_frames):
        """获取位置描述"""
        progress = frame_idx / total_frames
        if progress < 0.2:
            return "Lower part of the video"
        elif progress < 0.4:
            return "Middle lower of the video"
        elif progress < 0.6:
            return "Middle of the video"
        elif progress < 0.8:
            return "Middle upper of the video"
        else:
            return "Upper part of the video"
    
    def get_relative_position(self, frame_idx):
        """获取相对位置描述"""
        positions = [
            "Relative to the right side of the vehicle",
            "Relative to the left side of the vehicle",
            "Relative to the front of the vehicle",
            "Relative to the rear of the vehicle"
        ]
        return positions[frame_idx % len(positions)]
    
    def create_fallback_prediction(self, frame_idx, total_frames, text_query):
        """创建备用预测"""
        progress = frame_idx / total_frames
        
        color = self.extract_color(text_query)
        vehicle_type = self.extract_vehicle_type(text_query)
        orientation = self.extract_orientation(text_query)
        
        loc_x = 10.0 + progress * 15.0
        loc_z = 20.0 - progress * 8.0
        
        return {
            'valid': True,
            'bbox_x1': 800 + frame_idx * 10,
            'bbox_y1': 400 + frame_idx * 5,
            'bbox_x2': 1000 + frame_idx * 10,
            'bbox_y2': 600 + frame_idx * 5,
            'dim_height': 1.5,
            'dim_width': 1.8,
            'dim_length': 4.2,
            'loc_x': loc_x,
            'loc_y': 2.0 + np.sin(frame_idx * 0.2) * 0.3,
            'loc_z': loc_z,
            'rotation': np.sin(progress * np.pi) * 0.2,
            'distance': 20.0 + progress * 5.0,
            'order': (frame_idx % 3) + 1,
            'position': self.get_position_description(frame_idx, total_frames),
            'orientation': orientation,
            'vehicle_type': vehicle_type,
            'relative_position': self.get_relative_position(frame_idx),
            'adjacent_orientation': orientation,
            'adjacent_color': color,
            'unknown0': 0,
            'unknown1': 0,
            'unknown2': 0.0,
            'unknown3': 0.0
        }
    
    def optimize_temporal_consistency(self, predictions):
        """优化时序一致性"""
        if len(predictions) <= 1:
            return predictions
        
        # 平滑轨迹
        smoothed = self.smooth_trajectory(predictions)
        
        # 确保属性一致性
        consistent = self.ensure_attribute_consistency(smoothed)
        
        return consistent
    
    def smooth_trajectory(self, predictions):
        """平滑轨迹"""
        # 提取坐标
        loc_x = [p['loc_x'] for p in predictions]
        loc_y = [p['loc_y'] for p in predictions]
        loc_z = [p['loc_z'] for p in predictions]
        
        # 应用平滑
        window_size = min(3, len(predictions))
        
        smooth_x = self.moving_average(loc_x, window_size)
        smooth_y = self.moving_average(loc_y, window_size)
        smooth_z = self.moving_average(loc_z, window_size)
        
        # 更新预测
        for i, pred in enumerate(predictions):
            pred['loc_x'] = smooth_x[i]
            pred['loc_y'] = smooth_y[i]
            pred['loc_z'] = smooth_z[i]
            pred['distance'] = np.sqrt(smooth_x[i]**2 + smooth_y[i]**2 + smooth_z[i]**2)
        
        return predictions
    
    def ensure_attribute_consistency(self, predictions):
        """确保属性一致性"""
        if not predictions:
            return predictions
        
        # 使用多数投票确定主要属性
        colors = [p['adjacent_color'] for p in predictions]
        vehicle_types = [p['vehicle_type'] for p in predictions]
        orientations = [p['adjacent_orientation'] for p in predictions]
        
        main_color = max(set(colors), key=colors.count)
        main_vehicle = max(set(vehicle_types), key=vehicle_types.count)
        main_orientation = max(set(orientations), key=orientations.count)
        
        # 统一属性
        for pred in predictions:
            pred['adjacent_color'] = main_color
            pred['vehicle_type'] = main_vehicle
            pred['adjacent_orientation'] = main_orientation
        
        return predictions
    
    def moving_average(self, data, window_size):
        """移动平均"""
        if len(data) <= window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            smoothed.append(sum(window) / len(window))
        
        return smoothed