This is a project for **practicng AI deployment** which contains 
- a) PyTorch Model Training & Inference
- b) ONNX Model Export & optimization
- c) Usage of ONNX Runtime Inference Engine
- d) FastAPI Model Serving
- e) Docker Containerized Deployment
- f) Linux Environment Deployment & Debugging
- g) End-to-End AI Engineering Implementation

Steps are as follows:
- 1. add requirements -> requirements.txt
- 2. Train a CNN and export it to ONNX, run train.py as follows:
```bash
python train.py
  ```
- 3. ONNX Runtime inference, run infer_onnx.py as follows:
```bash
python infer_onnx.py 
```

- 4. FastAPI online service, run main.py as follows:
    - mainPage: http://localhost:8000
    - predict: http://localhost:8000/predict
```bash
python main.py
```
- 5. Containerized deployment, run dockerfile as follows:
 ```bash
docker build -t cnn_deploy .
docker run -p 8000:8000 cnn_deploy
 ```

Using C++ to infer:
```bash
g++ -o infer infer.cpp \
-I./onnxruntime-osx-arm64-1.19.2/include \
-L./onnxruntime-osx-arm64-1.19.2/lib \
-lonnxruntime

export DYLD_LIBRARY_PATH=./onnxruntime-osx-arm64-1.19.2/lib
./infer

```

## other tools
- TensorRT: only useful for linux+nvidia GPU
- OpenVINO: useful for intel CPU/CPU