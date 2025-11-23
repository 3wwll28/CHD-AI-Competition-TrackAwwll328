# demo.py - 修改后的版本，集成了路径设置
import os
import sys
import torch

# ==================================================
# 路径设置 - 直接集成在demo.py中
# ==================================================
def setup_import_paths():
    """设置Python导入路径，确保所有模块都能正确导入"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 添加必要的路径
    paths_to_add = [
        current_dir,  # 当前目录
        os.path.join(current_dir, 'utils'),  # utils目录
        os.path.join(current_dir, 'ops'),    # ops目录
    ]
    
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ 添加路径: {path}")
    
    print("🎯 导入路径设置完成!")

# 执行路径设置
setup_import_paths()

# ==================================================
# 导入依赖模块
# ==================================================
try:
    from Mono3DVGInference import Mono3DVGInference
    print("✅ Mono3DVGInference 导入成功")
except ImportError as e:
    print(f"❌ Mono3DVGInference 导入失败: {e}")
    sys.exit(1)

# ==================================================
# 主函数
# ==================================================
def main():
    # 🚀 初始化推理管道
    print("初始化推理管道...")
    
    try:
        pipeline = Mono3DVGInference(
            checkpoint_path=None,  # 可选，如果没有就使用None
            device="auto"  # 自动选择GPU/CPU
        )
        print("✅ 推理管道初始化成功")
    except Exception as e:
        print(f"❌ 推理管道初始化失败: {e}")
        print("💡 可能的原因:")
        print("   - 缺少依赖库，请检查 requirements.txt")
        print("   - 模型配置有问题")
        print("   - 硬件兼容性问题")
        return
    
    # 🎬 输入视频和文本
    print("\n📥 请输入推理参数:")
    video_path = input("视频文件路径: ").strip()
    text_query = input("文本查询: ").strip()
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        print("💡 请检查文件路径是否正确")
        return
    
    # 📊 运行推理
    print("\n🎬 开始推理...")
    try:
        result = pipeline.predict(
            video_path=video_path,
            text_query=text_query,
            output_path="submission.json"  # 输出文件
        )
        
        print(f"\n✅ 推理完成! 结果已保存至 submission.json")
        
        # 显示简要结果
        if 'videoID' in result:
            print(f"📊 推理结果摘要:")
            print(f"   - 视频ID: {result['videoID']}")
            print(f"   - 跟踪ID: {result['track_id']}")
            print(f"   - 描述: {result['description']}")
            
    except Exception as e:
        print(f"❌ 推理过程中出错: {e}")
        import traceback
        traceback.print_exc()

# ==================================================
# 程序入口
# ==================================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 长安大学AI大赛 - 3D视觉语言跟踪系统")
    print("=" * 50)
    
    # 环境检查
    print("\n🔍 环境检查:")
    print(f"   Python版本: {sys.version.split()[0]}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 运行主程序
    main()
    
    print("\n🎉 程序执行完毕!")