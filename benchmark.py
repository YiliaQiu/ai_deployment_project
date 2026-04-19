import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
try:
    import tensorrt as trt
except Exception as e:
    print("tensorrt Error: ", e)

try:
    from openvino import Core
except Exception as e:
    print("openvino Error: " ,e)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.flatten(1)
        x = self.fc1(x)
        return x

cnnmodel = CNN()
torch.save(cnnmodel, "cnn_mnist_nodel.pt") # 用于对比效率
print("✅ 已保存：cnn_mnist_model.pt")

def load_pt_model():
    model = CNN()
    model = torch.load("cnn_mnist_nodel.pt", map_location="cpu", weights_only=False)
    model.eval()
    return model

# macos不支持 PyTorch 原生 INT8 量化（会报错 NoQEngine）
def quantize_model(model):
    model_quant = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d}, # 量化卷积层和全连接层
        dtype = torch.qint8
    )
    return model_quant

def run_benchmark(infer_func, warm_up = True, loops = 1000):
    if warm_up:
        infer_func()

    start = time.time()
    for _ in range(loops):
        infer_func()
    total_time = time.time() - start
    avg_time = total_time / 1000
    return total_time, avg_time

if __name__ == "__main__":
    dummy_input_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
    torch_input_torch = torch.from_numpy(dummy_input_np)
    print("=" * 50)
    print("模型量化 + 推理性能基础测试")
    print("\n" + "=" * 50)

    model = load_pt_model()
    model_quant = quantize_model(model)
    ort_session = ort.InferenceSession("cnn_mnist.onnx")
    # model_trt = TRTModel()

    core = Core()
    model_ov = core.compile_model("cnn_mnist.onnx", "CPU")


    t1, a1 = run_benchmark(lambda: model(torch_input_torch), warm_up = True)
    t2, a2 = run_benchmark(lambda: model_quant(torch_input_torch), warm_up = True)
    t3, a3 = run_benchmark(lambda: ort_session.run(None, {"image_input": dummy_input_np}), warm_up = True)
    # t4, a3 = run_benchmark(lambda: model_trt(), warm_up = True)
    t5, a5 = run_benchmark(lambda: model_ov({"image_input": dummy_input_np}), warm_up = True)

    print("\n📊 最终性能对比（1000次推理）")
    print(f"原生PyTorch：总耗时 {t1:.3f}s  平均 {a1:.4f}s")
    print(f"量化PyTorch：总耗时 {t2:.3f}s  平均 {a2:.4f}s")
    print(f"ONNX Runtime：总耗时 {t3:.3f}s  平均 {a3:.4f}s")
    print(f"OPENVINO Runtime：总耗时 {t5:.3f}s  平均 {a5:.4f}s")

    speed_up = (a1 - a3) / a1 * 100
    print(f"\n✅ ONNX 平均耗时降低：{speed_up:.1f}%")

    # 结果如下：
    '''
    ==================================================
    模型量化 + 推理性能基础测试
    
    ==================================================
    
    📊 最终性能对比（1000次推理）
    原生PyTorch：总耗时 0.170s  平均 0.0002s
    量化PyTorch：总耗时 0.200s  平均 0.0002s
    ONNX Runtime：总耗时 0.041s  平均 0.0000s
    OPENVINO Runtime：总耗时 0.108s  平均 0.0001s

    ✅ ONNX 平均耗时降低：78.6%
    '''