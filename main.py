from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI(title="MNIST 手写数字识别服务(MNIST Digit Recognition Service)")
session = ort.InferenceSession("cnn_mnist.onnx")

def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L') # to gray
    img = img.resize((28, 28))
    img_np = np.array(img)

    img_np = np.where(img_np > 127, 255, 0)
    img_np = 255 - img_np

    img_np = img_np / 255.0
    img_np = img_np.astype(np.float32)
    img_np = img_np.reshape((1, 1, 28, 28))
    return img_np

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = preprocess(img_bytes)
    output = session.run(["pred_output"], {"image_input": img})
    pred_class = np.argmax(output[0])
    return {
        "status": "success",
        "上传文件名(UpLoadFile)": file.filename,
        "模型预测数字(predNum)": int(pred_class),
        "原始输出(predictions)": output[0].tolist()
    }

@app.get("/")
def root():
    return {"message": "✅ MNIST 手写数字识别服务已启动！"}