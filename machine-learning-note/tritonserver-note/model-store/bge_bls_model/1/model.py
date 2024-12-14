# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import asyncio

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])

    def finalize(self):
        pass

    async def execute(self, requests):
        responses = []
        for request in requests:
            infer_rsp_awaits = []
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input")

            # request for sparse_vecs
            infer_req = pb_utils.InferenceRequest(
                model_name="python_bge_m3_onnx",
                requested_output_names=["sparse_vecs", "token_num"],
                inputs=[input_tensor]
            )
            infer_rsp_awaits.append(infer_req.async_exec())
            # request for dense_vecs
            infer_req = pb_utils.InferenceRequest(
                model_name="python_bge_large_zh_onnx",
                requested_output_names=["dense_vecs", "token_num"],
                inputs=[input_tensor]
            )
            infer_rsp_awaits.append(infer_req.async_exec())
            # wait response
            infer_responses = await asyncio.gather(*infer_rsp_awaits)
            # check for error
            for infer_rsp in infer_responses:
                if infer_rsp.has_error():
                    raise pb_utils.TritonModelException(infer_rsp.error().message())
            # assemble response
            sparse_vecs = pb_utils.get_output_tensor_by_name(infer_responses[0], "sparse_vecs")
            sparse_token_num = pb_utils.get_output_tensor_by_name(infer_responses[0], "token_num")
            dense_vecs = pb_utils.get_output_tensor_by_name(infer_responses[1], "dense_vecs")
            dense_token_num = pb_utils.get_output_tensor_by_name(infer_responses[1], "token_num")
            response = pb_utils.InferenceResponse(
                output_tensors = [
                    sparse_vecs,
                    dense_vecs,
                    pb_utils.Tensor("sparse_token_num", sparse_token_num.as_numpy()),
                    pb_utils.Tensor("dense_token_num", dense_token_num.as_numpy()),
                ]
            )
            responses.append(response)
        return responses
