from fastapi import FastAPI
import onnxruntime as ort
import numpy as np

app = FastAPI(title="CNN模型部署服务")
session = ort.InferenceSession("cnn_mnist.onnx")

@app.get("/predict")
def predict():
    img = np.random.randn(1, 1, 28, 28).astype(np.float32)
    pred = session.run(["pred_output"], {"image_input": img})

    return {
        "status": "success",
        "predictions": pred[0].tolist()
    }

@app.get("/")
def root():
    return {"message": "✅ CNN 模型服务运行中"}