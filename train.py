import torch
import torch.nn as nn
from torchvision import datasets, transforms

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
model.eval()

dummy_input = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy_input,
    "cnn_mnist.onnx",
    input_names=["image_input"],
    output_names = ["pred_output"],
    opset_version = 12,
    do_constant_folding = True
)
print("✅ CNN模型导出成功: cnn_mnist.onnx")