# output_formatter.py - 修复JSON序列化版本
import json
import os
import numpy as np
import torch
from typing import Dict, List, Any

class CompetitionOutputFormatter:
    def __init__(self):
        self.default_values = {
            "sequence_id": "0000",
            "track_id": "001437",
            "color": "red", 
            "state": "Parking",
            "type": "Car"
        }
    
    def format_predictions(self, video_path: str, text_query: str, frame_predictions: List[Dict]) -> Dict[str, Any]:
        """格式化预测结果为比赛JSON格式"""
        
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        if '.' in video_id:
            video_id = video_id.split('.')[0]
        
        # 构建基础结构
        result = {
            "videoID": video_id,
            "sequence_id": self.default_values["sequence_id"],
            "track_id": self.default_values["track_id"],
            "color": self.default_values["color"],
            "state": self.default_values["state"], 
            "type": self.default_values["type"],
            "description": text_query
        }
        
        # 添加帧数据 - 确保所有数值都是Python原生类型
        for i, frame_pred in enumerate(frame_predictions):
            frame_key = f"frame{i}"
            result[frame_key] = self._format_frame_data(frame_pred, i)
        
        print(f"📊 输出格式化完成: 共 {len(frame_predictions)} 帧")
        return result
    
    def _format_frame_data(self, frame_pred: Dict, frame_index: int) -> List:
        """格式化单帧数据 - 确保所有数值都是Python原生类型"""
        
        # 🔧 修复: 确保所有数值都是Python原生类型
        frame_data = [
            # 第0个元素: valid (bool)
            self._ensure_python_bool(frame_pred.get('valid', True)),
            
            # 第1个元素: label路径 (str)
            f"infrastructure-side\\label\\camera\\nnss\\new2\\new3\\{frame_index:06d}.json",
            
            # 第2个元素: 图片路径 (str)  
            f"infrastructure-side\\img\\{frame_index:06d}.jpg",
            
            # 第3-5个元素: unknown0, unknown1, unknown2
            self._ensure_python_int(frame_pred.get('unknown0', 0)),
            self._ensure_python_int(frame_pred.get('unknown1', 0)),
            self._ensure_python_float(frame_pred.get('unknown2', 0.0)),
            
            # 第6-9个元素: 边界框坐标
            self._ensure_python_float(frame_pred.get('bbox_x1', 500.0)),
            self._ensure_python_float(frame_pred.get('bbox_y1', 300.0)),
            self._ensure_python_float(frame_pred.get('bbox_x2', 700.0)),
            self._ensure_python_float(frame_pred.get('bbox_y2', 500.0)),
            
            # 第10-12个元素: 3D尺寸
            self._ensure_python_float(frame_pred.get('dim_height', 1.5)),
            self._ensure_python_float(frame_pred.get('dim_width', 1.8)),
            self._ensure_python_float(frame_pred.get('dim_length', 4.5)),
            
            # 第13-16个元素: 3D位置和旋转
            self._ensure_python_float(frame_pred.get('loc_x', 10.0)),
            self._ensure_python_float(frame_pred.get('loc_y', 2.0)),
            self._ensure_python_float(frame_pred.get('loc_z', 20.0)),
            self._ensure_python_float(frame_pred.get('rotation', 0.0)),
            
            # 第17个元素: unknown3
            self._ensure_python_float(frame_pred.get('unknown3', 0.0)),
            
            # 第18个元素: 距离
            self._ensure_python_float(frame_pred.get('distance', 20.5)),
            
            # 第19个元素: 顺序
            self._ensure_python_int(frame_pred.get('order', 1)),
            
            # 第20-25个元素: 文本描述
            self._ensure_python_str(frame_pred.get('position', "Middle lower of the video")),
            self._ensure_python_str(frame_pred.get('orientation', "front")),
            self._ensure_python_str(frame_pred.get('vehicle_type', "Car")),
            self._ensure_python_str(frame_pred.get('relative_position', "Relative to the right side of the vehicle")),
            self._ensure_python_str(frame_pred.get('adjacent_orientation', "front")),
            self._ensure_python_str(frame_pred.get('adjacent_color', "white"))
        ]
        
        # 验证所有数据类型
        self._validate_frame_data_types(frame_data, frame_index)
        
        return frame_data
    
    def _ensure_python_bool(self, value):
        """确保返回Python bool类型"""
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        # 对于数字，非0为True，0为False
        return bool(value)
    
    def _ensure_python_int(self, value):
        """确保返回Python int类型"""
        if isinstance(value, (np.int32, np.int64, np.int8, np.int16, int)):
            return int(value)
        elif isinstance(value, (np.float32, np.float64, float)):
            return int(round(value))
        elif isinstance(value, torch.Tensor):
            return int(value.item())
        else:
            try:
                return int(value)
            except:
                return 0
    
    def _ensure_python_float(self, value):
        """确保返回Python float类型"""
        if isinstance(value, (np.float32, np.float64, float)):
            return float(value)
        elif isinstance(value, (np.int32, np.int64, int)):
            return float(value)
        elif isinstance(value, torch.Tensor):
            return float(value.item())
        else:
            try:
                return float(value)
            except:
                return 0.0
    
    def _ensure_python_str(self, value):
        """确保返回Python str类型"""
        if isinstance(value, str):
            return value
        else:
            return str(value)
    
    def _validate_frame_data_types(self, frame_data: List, frame_index: int):
        """验证帧数据的所有类型都是Python原生类型"""
        expected_types = [
            bool,    # 0: valid
            str,     # 1: label path
            str,     # 2: image path
            int,     # 3: unknown0
            int,     # 4: unknown1
            float,   # 5: unknown2
            float,   # 6: bbox_x1
            float,   # 7: bbox_y1
            float,   # 8: bbox_x2
            float,   # 9: bbox_y2
            float,   # 10: dim_height
            float,   # 11: dim_width
            float,   # 12: dim_length
            float,   # 13: loc_x
            float,   # 14: loc_y
            float,   # 15: loc_z
            float,   # 16: rotation
            float,   # 17: unknown3
            float,   # 18: distance
            int,     # 19: order
            str,     # 20: position
            str,     # 21: orientation
            str,     # 22: vehicle_type
            str,     # 23: relative_position
            str,     # 24: adjacent_orientation
            str      # 25: adjacent_color
        ]
        
        for i, (value, expected_type) in enumerate(zip(frame_data, expected_types)):
            if not isinstance(value, expected_type):
                print(f"⚠️  第{frame_index}帧 第{i}个元素类型错误: 期望{expected_type.__name__}, 实际{type(value).__name__}")
                # 强制转换类型
                try:
                    if expected_type == bool:
                        frame_data[i] = bool(value)
                    elif expected_type == int:
                        frame_data[i] = int(value)
                    elif expected_type == float:
                        frame_data[i] = float(value)
                    elif expected_type == str:
                        frame_data[i] = str(value)
                except:
                    # 如果转换失败，使用默认值
                    if expected_type == bool:
                        frame_data[i] = True
                    elif expected_type == int:
                        frame_data[i] = 0
                    elif expected_type == float:
                        frame_data[i] = 0.0
                    elif expected_type == str:
                        frame_data[i] = "unknown"

# 测试函数
def test_output_formatter():
    """测试输出格式化器"""
    print("🧪 测试输出格式化器...")
    
    formatter = CompetitionOutputFormatter()
    
    # 创建测试数据（包含numpy类型）
    import numpy as np
    test_predictions = [
        {
            'valid': np.bool_(True),
            'unknown0': np.int32(0),
            'unknown1': np.int64(1),
            'unknown2': np.float32(0.5),
            'bbox_x1': np.float64(800.5),
            'bbox_y1': 400.2,
            'bbox_x2': 1000.7,
            'bbox_y2': 600.3,
            'dim_height': 1.5,
            'dim_width': 1.8,
            'dim_length': 4.5,
            'loc_x': 10.0,
            'loc_y': 2.0,
            'loc_z': 20.0,
            'rotation': 0.1,
            'unknown3': 0.0,
            'distance': 25.5,
            'order': 1,
            'position': "Middle",
            'orientation': "front",
            'vehicle_type': "Car",
            'relative_position': "Relative",
            'adjacent_orientation': "front",
            'adjacent_color': "white"
        }
    ]
    
    result = formatter.format_predictions("test_video.mp4", "track the white car", test_predictions)
    
    # 验证结果可以被JSON序列化
    try:
        json_str = json.dumps([result], indent=2)
        print("✅ JSON序列化测试通过!")
        return True
    except Exception as e:
        print(f"❌ JSON序列化测试失败: {e}")
        return False

if __name__ == "__main__":
    test_output_formatter()