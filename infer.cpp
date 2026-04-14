#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
using namespace std;

// 轻量级图片读取哭库(无需安装OpenCV)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

vector<float> preprocess_image(const char* image_path) {
    int w, h, c;
    unsigned char* img = stbi_load(image_path, &w, &h, &c, 1); // 强制加载为单通道
    if (!img) {
        cerr << "图片加载失败！" << endl;
        exit(1);
    }

    vector<float> input(1 * 1 * 28 * 28, 0.0f);
    int idx = 0;

    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int px = x * w / 28;
            int py = y * h / 28;
            float val = img[py * w + px];

            val = (val > 127) ? 255: 0;
            val = 255.0f - val;
            input[idx++] = val / 255.0f; // 归一化到[0,1]
        }
    }
    stbi_image_free(img);
    return input;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist_infer");
    Ort::Session session(env, "cnn_mnist.onnx", Ort::SessionOptions{}); // macos/linux
    // Ort::Session session(env, L"cnn_mnist.onnx", Ort::SessionOptions{}); // windows

    auto input = preprocess_image("test.png");
    const int64_t input_shape[] = {1, 1, 28, 28};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input.data(), input.size(),
        input_shape, 4
    );

    const char* input_names[] = {"image_input"};
    const char* output_names[] = {"pred_output"};

    auto output_tensor = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* output = output_tensor[0].GetTensorMutableData<float>();

    int pred = 0;
    float max_val = output[0];
    for (int i = 1; i < 10; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            pred = i;
        }
    }

    cout << "预测结果(pred): " << pred << endl;
    return 0;
}
