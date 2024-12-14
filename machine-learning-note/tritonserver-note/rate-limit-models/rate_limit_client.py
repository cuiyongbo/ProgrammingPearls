
import queue
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
from tritonclient.utils import InferenceServerException

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

def callback(user_data, result, error):
    user_data._completed_requests.put((result, error))


server_addr = "localhost:8001"
triton_client = grpcclient.InferenceServerClient(server_addr)

in0 = np.array([1], dtype=np.float32)
input_tensors = [
    grpcclient.InferInput("INPUT0", in0.shape, np_to_triton_dtype(in0.dtype)).set_data_from_numpy(in0),
]

output_tensors = [
    grpcclient.InferRequestedOutput("OUTPUT0"),
]

# async inference
user_data = UserData()
inference_count = 10
for i in range(inference_count):
    triton_client.async_infer(
        model_name="model_a",
        inputs=input_tensors,
        callback=partial(callback, user_data),
        outputs=output_tensors,
        request_id= "model_a_" + str(i),
    )
    triton_client.async_infer(
        model_name="model_b",
        inputs=input_tensors,
        callback=partial(callback, user_data),
        outputs=output_tensors,
        request_id= "model_b_" + str(i),
    )

processed_count = 0
while processed_count < 2*inference_count:
    processed_count += 1
    data_item, err = user_data._completed_requests.get()
    if err:
        print("inference failed with {}".format(err))
        continue
    this_id = data_item.get_response().id
    print("Request id {} has been executed".format(this_id))
