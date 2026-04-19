'''
only useful for linux+nvidia
'''

import tensorrt as trt

class TRTModel:
    def __init__(self, engine_path = "cnn_mnist.trt"):
        logger = trt.Logger(trt.logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(onnx_path, "rb") as f:
            self.engine = runtime.deserailze_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
    def __call__(self):
        self.context.execute_v2([])

def build_trt_engine(onnx_path = "cnn_mnist.onnx", engine_path="cnn_mnist.trt"):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParset(network, logger)
    
    with open(onnx_path, "rb") as f:
        parser.parse(f.read())
    
    config = builder.create_builder_config()
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print("ONNX->TensorRT transform SUCCESS")

if __name__ == "__main__":
    build_trt_engine()