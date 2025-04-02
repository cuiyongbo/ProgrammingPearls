#!/usr/bin/env python3
# coding=utf-8

import sys
import time
import random
import logging
import argparse
import threading
import concurrent
import numpy as np
from tqdm import tqdm

# install tritonclient
# pip3 install tritonclient[all]

import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype, triton_to_np_dtype
from tritonclient.utils import InferenceServerException

from transformers import BertModel
from transformers import BertTokenizer


test_nlp_text = '''
Studies serve for delight, for ornament, and for ability. 
Their chief use for delight, is in privateness and retiring; 
for ornament, is in discourse; and for ability, is in the judgment 
and disposition of business. For expert men can execute, and perhaps 
judge of particulars, one by one; but the general counsels, and the 
plots and marshalling of affairs come best from those that are learned.
To spend too much time in studies is sloth; to use them too much for ornament 
is affection; to make judgment wholly by their rules is the humor of a scholar. 
They perfect nature and are perfected by experience: for natural abilities are 
like natural plants, that need pruning by study, and studies themselves do give 
forth directions too much at large, except they be bounded in by experience.
Crafty men contemn studies, simple men admire them, and wise men use them, 
for they teach not their own use; but that is a wisdom without them and above 
them, won by observation. Read not to contradict and confuse; nor to believe 
and take for granted; nor to find talk and discourse; but to weigh and consider.
Some books are to be tasted, others to be swallowed, and some few to be chewed 
and digested; that is some books are to be read only in parts; others to be read, 
but not curiously; and some few to be ready wholly, and with diligence and attention. 
Some books also may be read by deputy and extracts made of them by others; but that
would be only in the less important arguments, and the meaner sort of books; else 
distilled books are, like common distilled waters, flashy things.
Reading makes a full man; conference a ready man; and writing an exact man. 
And therefore, if a man write little, he had need have a great memory; 
if he confer little, he had need have a present wit; and if he read little, 
he had need have much cunning to seem to know that he doth not.
Histories make men wise; poets witty; the mathematics subtle; 
natural philosophy deep; moral grave; logic and rhetoric able to contend. 
Abeunt studia in mores. Nay there is no stond or impediment in the wit, 
but may be wrought out by fit studies: like as diseases of the body may 
have appropriate exercises. Bowling is good for the stone and reins; 
shooting for the lungs and breast; gentle walking for the stomach; 
riding for the head; and the like. So if a man’s wit be wandering, 
let him study the mathematics; for in demonstrations, if his wit be 
called away never so little, he must begin again. If his wit be not apt 
to distinguish or find differences, let him study the schoolmen; 
for they are cymini sectores. If he be not apt to beat over matters, 
and to call up one thing to prove and illustrate another, let him study 
the lawyers’ cases. So every defect of the mind may have a special receipt.
'''

client_timeout_in_second = 10

def do_bert_inference(client, model_name: str, inputs: list, verbose=False):
    def tokenize_text(text):
        enc = BertTokenizer.from_pretrained("/workspace/model-store/bert-large-uncased")
        encoded_text = enc(text, padding="max_length", max_length=512, truncation=True)
        return encoded_text["input_ids"], encoded_text["attention_mask"]
    input_ids, attention_mask = tokenize_text(inputs)
    in0 = np.array(input_ids, dtype=np.int32).reshape(-1, 512)
    in1 = np.array(attention_mask, dtype=np.int32).reshape(-1, 512)
    input_tensors = [
        grpcclient.InferInput("token_ids", in0.shape, np_to_triton_dtype(in0.dtype)).set_data_from_numpy(in0),
        grpcclient.InferInput("attn_mask", in1.shape, np_to_triton_dtype(in1.dtype)).set_data_from_numpy(in1),
    ]
    infer_rsp = client.infer(model_name, inputs=input_tensors, timeout=client_timeout_in_second)
    if verbose:
        out0 = infer_rsp.as_numpy("output")
        out1 = infer_rsp.as_numpy("3683")
        print(out0.shape, out0.dtype)
        print(out1.shape, out1.dtype)


def do_bge_inference(client, model_name: str, inputs: list, verbose=False):
    # inputs = ["hello world", "nice to meet you"]
    inputs = [s.encode() if isinstance(s, bytes) else s for s in inputs]
    in0 = np.array([inputs], dtype=np.object_).reshape(-1, 1)
    input_tensors = [
        grpcclient.InferInput("input", in0.shape, np_to_triton_dtype(in0.dtype)).set_data_from_numpy(in0),
    ]
    output_tensors = [
        grpcclient.InferRequestedOutput("dense_vecs"),
    ]
    if model_name == "python_bge_m3_onnx":
        output_tensors.append(grpcclient.InferRequestedOutput("sparse_vecs"))

    infer_rsp = client.infer(model_name, inputs=input_tensors, outputs=output_tensors, timeout=client_timeout_in_second)
    if verbose:
        dense_vecs = infer_rsp.as_numpy("dense_vecs")
        print(f"dense_vecs: {dense_vecs.shape}")
        if model_name == "python_bge_m3_onnx":
            sparse_vecs = infer_rsp.as_numpy("sparse_vecs")
            print(f"sparse_vecs: {sparse_vecs.shape}")


def do_inference(client, model_name: str, inputs: list, verbose=False):
    if model_name == "bert":
        return do_bert_inference(client, model_name, inputs, verbose)
    elif model_name in ["python_bge_m3_onnx", "python_bge_large_zh_onnx"]:
        return do_bge_inference(client, model_name, inputs, verbose)


def assemble_inputs(txt_len, batch_size):
    max_len = len(test_nlp_text)
    starts = random.choices(range(0, max_len - txt_len), k=batch_size)
    sentences = [test_nlp_text[s : (s + txt_len)] for s in starts]
    inputs = {
        "text": sentences,
    }
    return inputs


live = True
finished_tasks = set()
thread_cv = threading.Condition()
thread_cv_wait_timeout_sec = 60


def one_thread(worker_id, triton_client, model_name, inputs):
    global live, finished_tasks, thread_cv
    failures = 0
    max_requests = 200
    latency_list = []
    random.shuffle(inputs["text"])
    for num_request in tqdm(range(max_requests)):
        try:
            start = time.time()
            do_inference(triton_client, model_name, inputs["text"])
            latency = time.time() - start
            latency_list.append(latency)
            if not live:
                break
        except:
            failures += 1
            logging.exception(f"worker {worker_id} failed with exception at {num_request}th request")
            break
    if live:
        with thread_cv:
            finished_tasks.add(worker_id)
            thread_cv.notify(n=1)
            print(f"worker {worker_id} finished, total requests: {num_request+1}, failures: {failures}", flush=True)
    return latency_list


def current_performance(worker_num):
    global live, finished_tasks, thread_cv, thread_cv_wait_timeout_sec
    def is_finished():
        return len(finished_tasks) >= worker_num
    with thread_cv:
        thread_cv.wait_for(predicate=is_finished, timeout=thread_cv_wait_timeout_sec)
        live = False
        print("master finished", flush=True)


def one_profile(worker_num, triton_client, model_name, inputs):
    global live, finished_tasks, thread_cv
    live = True
    finished_tasks.clear()
    latency_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num + 1) as executor:
        executor.submit(current_performance, worker_num)
        futures = {}
        for i in range(worker_num):
            f = executor.submit(one_thread, i, triton_client, model_name, inputs)
            futures[f] = i
        for f in concurrent.futures.as_completed(futures):
            try:
                task_id = futures[f]
                result = f.result()
                latency_list.extend(result)
            except Exception as exc:
                logging.exception(f"Worker {task_id} failed with exception")
                #print(f"Task {task_id} generated an exception: {exc}")
    return latency_list


# set random seed to facilitate reproducibility
random.seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="bert", choices=["bert", "python_bge_m3_onnx", "python_bge_large_zh_onnx"])
parser.add_argument("-s", "--server-address", type=str, default="localhost:8001")
parser.add_argument("-c", "--max-concurrency", type=int, default=8)
parser.add_argument("--warmup-only", action="store_true")
args = parser.parse_args()
print(args)

server_addr = args.server_address
triton_client = grpcclient.InferenceServerClient(server_addr)

model_name = args.model
model_metas = {
    "python_bge_m3_onnx": {
        "max_batch_size": 16,
        # "max_input_length": 8192,
        "max_input_length": 1024,
    },
    "python_bge_large_zh_onnx": {
        "max_batch_size": 16,
        "max_input_length": 512,
    },
    "bert": {
        "max_batch_size": 16,
        "max_input_length": 512,
    },
}
max_batch_size = model_metas[model_name]["max_batch_size"]
max_input_length = model_metas[model_name]["max_input_length"]

# warmup
print("start warmup")
for i in range(10):
    inputs = assemble_inputs(200, i + 1)
    do_inference(triton_client, model_name, inputs["text"], verbose=True)
print("warmup done")
if args.warmup_only:
    sys.exit(0)

headers = [
    "text_length",
    "batch_size",
    "worker_num",
    "request_num",
    "total_time",
    "QPS",
    "throughput(k_token/s)",
    "latency.avg(ms)",
    "latency.p90(ms)",
    "latency.p95(ms)",
    "latency.p99(ms)",
]
outputfile = open(f"profile_tritonserver_{model_name}.csv", "w")
outputfile.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(*headers))
print("{:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s} {:>24s}".format(*headers))
for batch_size in range(1, max_batch_size+2, 2):
    for txt_len in range(128, max_input_length + 64, 128):
        for worker_num in range(1, args.max_concurrency, 2):
            inputs = assemble_inputs(txt_len, batch_size)
            start = time.time()
            latency_list = one_profile(worker_num, triton_client, model_name, inputs)
            if len(latency_list) <= 0:
                print(f"failed to collect data for <{txt_len}, {batch_size}, {worker_num}>")
                continue
            total_time = time.time() - start
            num_requests = len(latency_list)
            qps = num_requests / total_time
            throughput = int(num_requests * txt_len * batch_size / total_time / 1000)
            p50 = np.percentile(latency_list, 50) * 1000
            p90 = np.percentile(latency_list, 90) * 1000
            p95 = np.percentile(latency_list, 95) * 1000
            p99 = np.percentile(latency_list, 99) * 1000
            one_row = [
                txt_len,
                batch_size,
                worker_num,
                num_requests,
                total_time,
                qps,
                throughput,
                p50,
                p90,
                p95,
                p99,
            ]
            print(
                "{:>24d} {:>24d} {:>24d} {:>24d} {:>24.2f} {:>24.2f} {:>24d} {:>24.0f} {:>24.0f} {:>24.0f} {:>24.0f}".format(
                    *one_row
                )
            )
            outputfile.write(
                "{:d},{:d},{:d},{:d},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n".format(*one_row)
            )
            outputfile.flush()
            time.sleep(1)
outputfile.close()
