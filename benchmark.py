import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

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

model = CNN()
torch.save(model, "cnn_mnist_nodel.pt") # 用于对比效率
print("✅ 已保存：cnn_mnist_model.pt")

def run_benchmark():
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    torch_input = torch.from_numpy(dummy_input)

    print("=" * 50)
    print("开始测试：原始Pytorch推理1000次(start inference: using Pytorch for 1000 times)")
    model = torch.load("cnn_mnist_nodel.pt", map_location = "cpu", weights_only=False)
    model.eval()

    # 预热，避免第一个加载耗时干扰
    with torch.no_grad():
        model(torch_input)

    start = time.time()
    for _ in range(1000):
        with torch.no_grad():
              model(torch_input)
    torch_total_time = time.time() - start
    torch_avg_time = torch_total_time / 1000

    print(f"PyTorch总耗时: {torch_total_time:.4f}秒")
    print(f"PyTorch平均每次耗时: {torch_avg_time:.5f}秒")

    #--------------------#
    ort_session = ort.InferenceSession("cnn_mnist.onnx")
    # 预热
    ort_session.run(["pred_output"], {"image_input": dummy_input})

    start = time.time()
    for _ in range(1000):
        ort_session.run(["pred_output"], {"image_input": dummy_input})
    ort_total_time = time.time() - start
    ort_avg_time = ort_total_time / 1000

    print(f"ort总耗时: {ort_total_time:.4f}秒")
    print(f"ort平均每次耗时: {ort_avg_time:.4f}秒")

    print("\n" + "=" * 50)
    print("🔥 最终对比 🔥")
    print(f"PyTorch  平均：{torch_avg_time:.4f} s/次")
    print(f"ONNX Runtime 平均：{ort_avg_time:.5f} s/次")

    if ort_avg_time < torch_avg_time:
        faster = (torch_avg_time - ort_avg_time) / torch_avg_time * 100
        print(f"✅ ONNX Runtime 更快，加速比：{faster:.1f}%")
    else:
        faster = (ort_avg_time - torch_avg_time) / ort_avg_time * 100
        print(f"✅ PyTorch 更快，加速比：{faster:.1f}%")



if __name__ == "__main__":
    run_benchmark()
    # 结果如下：
    '''
    开始测试：原始Pytorch推理1000次(start inference: using Pytorch for 1000 times)
    PyTorch总耗时: 0.0869秒
    PyTorch平均每次耗时: 0.00009秒
    ort总耗时: 0.0205秒
    ort平均每次耗时: 0.0000秒
    
    ==================================================
    🔥 最终对比 🔥
    PyTorch  平均：0.0001 s/次
    ONNX Runtime 平均：0.00002 s/次
    ✅ ONNX Runtime 更快，加速比：76.4%
    '''