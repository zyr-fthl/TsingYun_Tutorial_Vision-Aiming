import sys
import os

print("=" * 50)
print("【PyTorch 运行环境深度诊断工具】")
print(f"当前 Python 解释器位置: {sys.executable}")
print(f"当前工作目录: {os.getcwd()}")
print("=" * 50)

try:
    print("1. 正在尝试导入 torch 基础库...")
    import torch
    print("   ✅ 成功导入 torch!")
    
    print(f"   - Torch 版本: {torch.__version__}")
    print(f"   - CUDA 是否可用 (GPU加速): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - 当前 GPU 设备: {torch.cuda.get_device_name(0)}")
        
    print("\n2. 正在尝试创建内存张量 (Tensor)...")
    x = torch.rand(2, 3)
    print("   ✅ 成功创建张量:")
    print(x)
    
    print("\n3. 正在模拟执行一个小型的神经网络前向传播...")
    # 模拟一个最基础的矩阵相乘，测试 c10.dll 和底层 C++ 算子是否正常
    linear = torch.nn.Linear(3, 2)
    out = linear(x)
    print("   ✅ 成功执行 C++ 算子前向传播:")
    print(out)
    
    print("\n" + "=" * 50)
    print("🎉 恭喜！你的 PyTorch 环境一切正常，无任何 DLL 冲突！")
    print("=" * 50)

except Exception as e:
    print("\n❌ 糟糕，捕获到运行时错误！")
    print("-" * 50)
    import traceback
    traceback.print_exc()
    print("-" * 50)
    print("\n💡 诊断建议：")
    if "1114" in str(e) or "c10.dll" in str(e).lower():
        print("这依然是典型的 Windows 动态链接库初始化失败。")
        print("请尝试在终端运行以下命令切换为【纯 CPU 兼容版 Torch】再试：")
        print("uv pip uninstall torch")
        print("uv pip install torch --index-url https://download.pytorch.org/whl/cpu")
    else:
        print("请检查虚拟环境是否完整。")
    print("=" * 50)