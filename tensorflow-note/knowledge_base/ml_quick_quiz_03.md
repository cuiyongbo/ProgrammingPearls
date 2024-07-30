# Model inference Quiz - basic

```
Q: what is `triton_python_backend_stub` for in tritonserver's backend directory?

The `triton_python_backend_stub` is a crucial component of the Python backend in NVIDIA Triton Inference Server. It acts as an intermediary between the Triton server process and the Python runtime, facilitating communication and execution of Python-based models.

### Purpose of `triton_python_backend_stub`

1. **Inter-Process Communication (IPC)**: The `triton_python_backend_stub` handles IPC between the Triton server and the Python runtime. This allows the Triton server to offload model execution to a separate Python process, ensuring that the Python Global Interpreter Lock (GIL) does not interfere with the performance of the Triton server.

2. **Shared Memory Management**: The stub manages shared memory regions used for passing input and output tensors between the Triton server and the Python process. This ensures efficient data transfer with minimal overhead.

3. **Model Execution**: The stub is responsible for loading the Python model script (`model.py`), initializing the model, executing inference requests, and finalizing the model. It calls the appropriate methods (`initialize`, `execute`, `finalize`) in the `TritonPythonModel` class defined in the model script.

### How `triton_python_backend_stub` Works

1. **Initialization**: When the Triton server starts, it initializes the Python backend and launches the `triton_python_backend_stub` as a separate process. The stub loads the Python model script and calls the `initialize` method.

2. **Handling Requests**: For each inference request, the Triton server writes the input tensors to shared memory and sends an IPC message to the stub. The stub reads the input tensors, calls the `execute` method in the Python model, and writes the output tensors to shared memory.

3. **Finalization**: When the model is unloaded, the stub calls the `finalize` method in the Python model to clean up resources.

### Example Workflow

1. **Model Repository Structure**:
   ```
   model_repository/
   └── my_python_model/
       ├── 1/
       │   └── model.py
       └── config.pbtxt
   ```

2. **Configuration File (`config.pbtxt`)**:
   ```plaintext
   name: "my_python_model"
   platform: "python"
   input [
     {
       name: "INPUT0"
       data_type: TYPE_FP32
       dims: [1]
     }
   ]
   output [
     {
       name: "OUTPUT0"
       data_type: TYPE_FP32
       dims: [1]
     }
   ]
   ```

3. **Python Model Script (`model.py`)**:
   ```python
   import triton_python_backend_utils as pb_utils
   import numpy as np

   class TritonPythonModel:
       def initialize(self, args):
           """Initialize the model."""
           self.model_config = args['model_config']

       def execute(self, requests):
           """Execute inference on the input requests."""
           responses = []
           for request in requests:
               input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
               input0_data = input0.as_numpy()

               # Perform inference (example: simple addition)
               output0_data = input0_data + 1.0

               output0 = pb_utils.Tensor("OUTPUT0", output0_data)
               inference_response = pb_utils.InferenceResponse(output_tensors=[output0])
               responses.append(inference_response)

           return responses

       def finalize(self):
           """Clean up resources."""
           pass
   ```

### Communication Flow

1. **Client Request**: A client sends an inference request to Triton using gRPC or HTTP.
2. **Triton Server**: Triton receives the request and identifies the model and backend based on the model name and configuration.
3. **Shared Memory Setup**: Triton sets up shared memory regions for input and output tensors.
4. **IPC Message**: Triton sends an IPC message to the `triton_python_backend_stub` indicating that new input data is available.
5. **Stub Reads Input**: The stub reads the input tensors from shared memory.
6. **Model Execution**: The stub calls the `execute` method in the Python model script, passing the input tensors.
7. **Stub Writes Output**: The stub writes the output tensors to shared memory.
8. **IPC Response**: The stub sends an IPC message back to Triton indicating that the inference results are available.
9. **Triton Server Response**: Triton reads the output tensors from shared memory and sends the inference response back to the client.


process architecture:

```
root@s-20240707192827-jzg4p-6cf6945b8c-8mzv5:/opt/tritonserver/backends/python
UID          PID    PPID  C STIME TTY          TIME CMD
root         643       0  0 11:02 pts/0    00:00:00 bash
root         678     643  0 11:06 pts/0    00:00:00   ps -ef -H
root           1       0  0 Jul11 ?        02:24:22 tritonserver --pinned-memory-pool-byte-size=268435456 --allow-metrics=false --http-thread-count=16 --log-file=/tmp/triton_inference_serv
root          90       1  1 Jul11 ?        07:05:07   /opt/tritonserver/backends/python/triton_python_backend_stub /model-store/triton-model-store/python_bge_large_zh_onnx/1/m
root          91       1  1 Jul11 ?        07:06:46   /opt/tritonserver/backends/python/triton_python_backend_stub /model-store/triton-model-store/python_bge_large_zh_onnx/1/m
root          92       1  1 Jul11 ?        07:06:14   /opt/tritonserver/backends/python/triton_python_backend_stub /model-store/triton-model-store/python_bge_large_zh_onnx/1/m
root          96       1  1 Jul11 ?        07:08:00   /opt/tritonserver/backends/python/triton_python_backend_stub /model-store/triton-model-store/python_bge_large_zh_onnx/1/m

# ldd /opt/tritonserver/backends/python/triton_python_backend_stub
        linux-vdso.so.1 (0x00007ffd6e3c3000)
        libcudart.so.12 => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12 (0x00007f330f800000)
        libpython3.10.so.1.0 => /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 (0x00007f330f229000)
        libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f330effd000)
        libgcc_s.so.1 => /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f330fc04000)
        libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007f330edd4000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f330fd1f000)
        libdl.so.2 => /usr/lib/x86_64-linux-gnu/libdl.so.2 (0x00007f330fbfd000)
        libpthread.so.0 => /usr/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f330fbf8000)
        librt.so.1 => /usr/lib/x86_64-linux-gnu/librt.so.1 (0x00007f330fbf3000)
        libexpat.so.1 => /usr/lib/x86_64-linux-gnu/libexpat.so.1 (0x00007f330fbc2000)
        libz.so.1 => /usr/lib/x86_64-linux-gnu/libz.so.1 (0x00007f330fba6000)
        libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (0x00007f330fabd000)
```


### Conclusion

The `triton_python_backend_stub` is a critical component that enables the Python backend in Triton Inference Server. It handles IPC, shared memory management, and model execution, allowing you to implement custom inference logic in Python while benefiting from Triton's scalability and performance. By understanding how the stub works, you can effectively develop and deploy Python-based models with Triton.
```
