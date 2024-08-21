# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#coding: utf-8

import os
import sys
import time
import json
import traceback
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer
import onnxruntime as ort
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

'''
# Debug
tritonserver --model-repository=/model-store/model-repository --allow-metrics false  --log-verbose 2
tritonserver --model-repository=/root/code/huggingface-store/model-repository --allow-metrics=true  --log-verbose=2

# profiling tritonserver
nsys launch --cuda-memory-usage=true tritonserver --model-store=/model-store/model-repository --allow-metrics=false --allow-grpc=false --model-control-mode=explicit --load-model=python_bge_large_zh_onnx --log-file=/tmp/triton_inference_server.log
nsys start
nsys stop
nsys shutdown -kill sigkill


# 测试
tritonserver --pinned-memory-pool-byte-size=268435456 --cache-config=local,size=268435456 --allow-metrics=false --http-thread-count=16 --log-file=/tmp/triton_inference_server.log --model-control-mode=explicit --load-model=python_bge_large_zh_onnx --model-repository=/root/code/triton-model-store
tritonserver --pinned-memory-pool-byte-size=268435456 --cache-config=local,size=268435456 --allow-metrics=false --http-thread-count=16 --model-control-mode=explicit --load-model=python_bge_large_zh_onnx --model-repository=/root/code/triton-model-store

# 在线启动命令
tritonserver --pinned-memory-pool-byte-size=268435456 --cache-config=local,size=268435456 --allow-metrics=false --http-thread-count=16 --log-file=/tmp/triton_inference_server.log --model-control-mode=explicit --load-model=python_bge_large_zh_onnx --model-repository=/volcvikingdb-model-store/triton-model-repository

'''

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.logger.log_info("TritonPythonModel args: {}".format(args))
        self.model_name = json.loads(args["model_config"]).get("name", "bge-large-zh")
        model_path = "{}/{}".format(args["model_repository"], args["model_version"])
        model_content = os.listdir(model_path)
        if model_content:
            self.logger.log_info("TritonPythonModel model_content: {}".format(model_content))
        else:
            self.logger.log_error("TritonPythonModel no model found in {}".format(model_path))
            sys.exit(-1)
        self.max_sequence_len = 512 # BE CAUTIOUS, adjust the configuration before you reuse code 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = self.create_model_for_provider(model_path+"/model.onnx")
        self.device = "{}:{}".format(args["model_instance_kind"].lower(), args["model_instance_device_id"])
        # https://github.com/triton-inference-server/onnxruntime_backend/issues/103
        self.session_run_opts = ort.RunOptions()
        self.session_run_opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", self.device)
        self.warmup()
        self.counter = 0

    def create_model_for_provider(self, model_path: str) -> InferenceSession: 
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
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
            inputs = self.tokenizer(input_list, padding=True, truncation=True, return_tensors="np", max_length=self.max_sequence_len)
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
                inputs = self.tokenizer(input_list, padding=True, truncation=True, return_tensors="np", max_length=self.max_sequence_len)
                outputs = self.model.run(None, input_feed=dict(inputs), run_options=self.session_run_opts)
                total_tokens = np.sum(inputs["attention_mask"], axis=1, keepdims=True)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "dense_vecs", outputs[0],
                        ),
                        pb_utils.Tensor(
                            "total_tokens", total_tokens,
                        ),
                        pb_utils.Tensor(
                            "token_num", total_tokens,
                        ),
                    ]
                )
                responses.append(inference_response)
                duration_ms = int((time.time()-start)*1000)
                self.logger.log_info("{} batch_size: {}, coarse tokens: {}, inference using {}ms".format(
                                self.model_name,
                                len(input_list),
                                np.sum(total_tokens),
                                duration_ms)
                    )
                # for requests with large batch_size, we sleep a tiny millisecond to wait onnxruntime to release GPU memory before returning
                if duration_ms > 3000:
                    time.sleep(0.0001*duration_ms) # Be cautious, input of ``time.sleep`` is in second
            except:
                time.sleep(5) # the backend probably runs out of GPU memory, we sleep several seconds to wait onnxruntime to release GPU memory before next request
                raise Exception("{} inference failed with exception: {}".format(self.model_name, traceback.format_exc()))

        return responses
