import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("cnn_mnist.onnx")

test_input = np.random.randn(1, 1, 28, 28).astype(np.float32)

output = session.run(
    ["pred_output"],
    {"image_input": test_input}
)
print("✅ ONNX 推理结果：")
print(output)