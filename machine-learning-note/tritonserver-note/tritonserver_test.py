#!/usr/bin/env python3
#coding=utf-8
import json
import requests
import time
import random

# Triton server URL and model name
triton_server_url = "http://localhost:8000"
model_name = "python_bge_m3_onnx"

demo_string = "hello world"
input_data = [demo_string]*1  # Use a list even for a single string
request_body = {
    "inputs": [
        {
            "name": "input",
            "datatype": "BYTES",
            "shape": [len(input_data), 1],
            "data": input_data,
        }
    ],
    "outputs": [
        {
            "name": "dense_vecs"
        },
        {
            "name": "token_ids"
        },
        {
            "name": "token_weights"
        },
        {
            "name": "tokens"
        },
    ]
}


# Send the inference request
response = requests.post(
    f"{triton_server_url}/v2/models/{model_name}/infer",
    headers={"Content-Type": "application/json"},
    json=request_body,
)


# Check the response status and process the output
if response.ok:
    print(response.text)
    print("succeeded in inference")
    #output_data = response.json()
    # The output data will be in the 'outputs' key
    #dense_vecs = output_data['outputs'][0]['data']
    #print(dense_vecs)
else:
    print(response.text)
    print("Inference request failed with status code", response.status_code)

