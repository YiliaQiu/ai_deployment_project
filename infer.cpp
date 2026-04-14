#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>
using namespace std;

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mnist_infer");
    Ort::Session session(env, "cnn_mnist.onnx", Ort::SessionOptions{}); // macos/linux
    // Ort::Session session(env, L"cnn_mnist.onnx", Ort::SessionOptions{}); // windows

    vector<float> input(1 * 1 * 28 * 28, 0.0f);

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
