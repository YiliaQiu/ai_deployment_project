import numpy as np
import tritonclient.http as httpclient

class TritonClient:
    def __init__(self, url = "localhost:8000", model_name = "cnn_mnist"):
        '''
        初始化Triton客户端
        :param url: Triton 服务地址
        :param model_name: 模型仓库中的模型名字
        '''
        self.client = httpclient.InferenceServerClient(url=url)
        self.model_name = model_name

        self.inputs = [
            httpclient.InferInput(
                name = "image_input",
                shape = (1, 1, 28, 28),
                datatype = "FP32"
            )
        ]
        self.outputs = [
            httpclient.InferRequestedOutput("output")
        ]

    def infer(self, input_np):
        '''
        推理接口
        :param input_np: 输入numpy数组（1，1，28，28）
        :return: 推理结果
        '''
        self.inputs[0].set_data_from_numpy(input_np)
        response = self.client.infer(
            model_name = self.model_name,
            inputs = self.inputs,
            outputs = self.outputs
        )
        return response.as_numpy("output")

if __name__ == "__main__":
    triton = TritonClient()
    dummy_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
    result = triton.infer(dummy_input)
    print("✅ Triton推理成功，输出shape: ", result.shape)