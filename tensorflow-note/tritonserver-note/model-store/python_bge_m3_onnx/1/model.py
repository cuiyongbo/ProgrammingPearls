#coding: utf-8

import os
import sys
import time
import traceback
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("TritonPythonModel args: {}".format(args))
        model_path = "{}/{}".format(args["model_repository"], args["model_version"])
        model_content = os.listdir(model_path)
        if model_content:
            self.logger.log_info("TritonPythonModel model_content: {}".format(model_content))
        else:
            self.logger.log_error("TritonPythonModel no model found in {}".format(model_path))
            sys.exit(-1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.create_model_for_provider(model_path+"/model.onnx")
        self.device = "{}:{}".format(args["model_instance_kind"].lower(), args["model_instance_device_id"])
        # GPU memory leak issue: https://github.com/triton-inference-server/onnxruntime_backend/issues/103
        self.session_run_opts = ort.RunOptions()
        self.session_run_opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", self.device)
        self.warmup()
        self.counter = 0

    def create_model_for_provider(self, model_path: str) -> InferenceSession: 
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#configuration-options
        provider = "CUDAExecutionProvider"
        assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"
        providers = [
            ('CUDAExecutionProvider', {
                #'device_id': 0,
                #'arena_extend_strategy': 'kNextPowerOfTwo',
                #'gpu_mem_limit': 512 * 1024 * 1024, # 1GB
            }),
            #'CPUExecutionProvider',
        ]
        # https://onnxruntime.ai/docs/api/python/api_summary.html#onnxruntime.SessionOptions
        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        #options.enable_cpu_mem_arena = False
        #options.enable_mem_pattern = False
        #options.enable_mem_reuse = False
        #options.intra_op_num_threads = 1
        #options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        # Load the model as a graph and prepare the GPU backend 
        session = InferenceSession(model_path, options, providers=providers)
        session.disable_fallback()
        return session
    
    def warmup(self):
        input_list = [
            "Is this your first time writing a config file?",
            "A minimal model configuration must specify the platform and/or backend properties,the max_batch_size property, and the input and output tensors of the model."
        ]
        for _ in range(10):
            inputs = self.tokenizer(input_list, padding=True, truncation=True, return_tensors="np")
            _ = self.model.run(None, input_feed=dict(inputs), run_options=self.session_run_opts)

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                start = time.time()
                input_tensor = pb_utils.get_input_tensor_by_name(request, "input")
                #self.logger.log_info("TritonPythonModel input_tensor.shape", input_tensor.shape())
                input_list = input_tensor.as_numpy().flatten().tolist()
                input_list = [s.decode("utf-8") for s in input_list]
                inputs = self.tokenizer(input_list, padding=True, truncation=True, return_tensors="np")
                outputs = self.model.run(None, input_feed=dict(inputs), run_options=self.session_run_opts)
                token_array = []
                for r in outputs[1]:
                    token_array.append(self.tokenizer.convert_ids_to_tokens(r))
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "dense_vecs", outputs[0],
                        ),
                        pb_utils.Tensor(
                            "token_ids", outputs[1],
                        ),
                        pb_utils.Tensor(
                            "token_weights", outputs[2],
                        ),
                        pb_utils.Tensor(
                            "tokens", np.array(token_array, dtype=np.object_),
                        ),
                    ]
                )
                responses.append(inference_response)
                duration = int((time.time()-start)*1000)
                self.logger.log_info("model batch_size: {}, coarse tokens: {}, inference using {}ms".format(
                                len(input_list),
                                outputs[1].shape[0]*outputs[1].shape[1],
                                duration)
                    )
                # for requests with large batch_size, we sleep a tiny second to wait onnxruntime to release GPU memory before returning
                if duration > 3000:
                    time.sleep(0.1 * duration)
            except:
                time.sleep(5) # the backend probably runs out of GPU memory, we sleep several seconds to wait onnxruntime to release GPU memory before next request
                raise Exception("model inference failed with exception: {}".format(traceback.format_exc()))

        return responses
