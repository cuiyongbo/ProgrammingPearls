#coding=utf-8
import os
import json
import logging
import numpy as np
from collections import defaultdict

import torch
import transformers
import sentence_transformers
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("sentence_transformers version {}".format(sentence_transformers.__version__))
logger.info("transformers version {}".format(transformers.__version__))


"""
# 打包命令
# 注意 model_name 要使用 "_" 而不是 "-", 否则会报错: ModuleNotFoundError: No module named 'tao-8k'
torch-model-archiver -f --model-name tao_8k --version 1.0 --serialized-file tao-8k/pytorch_model.bin --handler tao-8k/handler.py --extra-files "tao-8k/" 
# 本地调试启动命令
torchserve --start --ncs --model-store=model-store --models tao_8k=tao_8k.mar --ts-config=torchserve.conf --log-config=log4j2.xml

# 为了加速调试, 可以打包格式可以不选 mar
# 打包整个文件, 因为 SentenceTransformer 使用需要使用 pooling 配置
torch-model-archiver -f --model-name=tao_8k --version=1.0 --serialized-file=tao-8k/pytorch_model.bin --handler=tao-8k/handler.py --extra-files="tao-8k/" --archive-format=no-archive --export-path=no-archive
# 配合上面的启动命令
torchserve --start --ncs --model-store=no-archive --models tao_8k --ts-config=torchserve.conf --log-config=log4j2.xml


# 在线服务启动命令
torchserve --start --ncs --model-store=/volcvikingdb-model-store --models tao-8k=tao_8k.mar --ts-config=/volcvikingdb-model-store/torchserve.conf --log-config=/volcvikingdb-model-store/log4j2.xml
torchserve --start --ncs --model-store=/volcvikingdb-model-store --models tao-8k=tao_8k.mar --ts-config=/volcvikingdb-model-store/torchserve.conf --log-config=/volcvikingdb-model-store/log4j2.xml --foreground

# 停止服务. 服务停止后, 后台会自动拉起, 但是所做镜像内所做的改动也会被丢弃, 比如手动安装的 pip 包
torchserve --stop

# request torchserve
curl  http://localhost:8081/models
curl  http://localhost:8081/models/tao_8k?customized=true
curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/tao_8k --data-raw '{"input":"如何使用torchserve部署模型"}'
curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/tao_8k --data-raw '{"parameters":{"return_tokenization_result":true},"input":"如何使用torchserve部署模型"}'
# 压测
for i in $(seq 100); do curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/tao_8k --data-raw '{"input":["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]}'; done
"""

class Tao8kHandler(BaseHandler):
    def __init__(self):
        super(Tao8kHandler, self).__init__()
        self.parameters = {}
        self.initialized = False


    def initialize(self, ctx):
        #  running initialize, 
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_material_list = os.listdir(model_dir)
        logger.info("model_material_list: {}".format(model_material_list))
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        # 需要在这里指定 device, 不然只有一个 GPU 有负载
        self.model = sentence_transformers.SentenceTransformer(model_dir, device=self.device)

        # huggingface transformer
        #self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        #self.model = transformers.AutoModel.from_pretrained(model_dir)

        # use gpu if gpu is available
        self.model.to(self.device)
        # set the model in the evaluation mode, and it will return the layer structure
        self.model.eval()
        logger.info("running warmup")
        #sample_sentences = ["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]
        sample_sentences = ["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]
        for i in range(8):
            self.inference(sample_sentences)
        logger.info("Transformer model from path {} loaded successfully".format(model_dir))
        logger.info("running initialize, manifest: {}, properties: {}".format(self.manifest, properties))
        self.initialized = True


    def preprocess(self, requests):
        logger.debug("running preprocess, requests: {}".format(requests))
        self.parameters = {}
        input_batch = []
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            if isinstance(input_text, dict):
                format_input = input_text
            else:
                format_input = json.loads(input_text)
            self.parameters = format_input.get("parameters", {})
            # 注意不要开启 torchserve 的 batch 机制
            if isinstance(format_input["input"], str):
                input_batch.append(format_input["input"])
            elif isinstance(format_input["input"], list):
                input_batch = format_input["input"]
            else:
                raise TypeError("expect input to be either string or list[string]")
            logger.debug("Received {}th input: {}".format(idx, input_text))
        return input_batch


    def inference(self, input_batch):
        return self.inference_with_sentence_tranformers(input_batch)
        #return self.inference_with_transformers(input_batch)


    def inference_with_transformers(self, input_batch):
        logger.debug("running inference, input_batch: {}".format(input_batch))
        # Tokenize sentences
        encoded_input = self.tokenizer(input_batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embedding = model_output[0][:, 0]
        # normalize embeddings
        sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
        output = defaultdict()
        output["sentence_embedding"] = sentence_embedding.tolist()
        if self.parameters.get("return_tokenization_result", False):
            tokenization_result = []
            for i in range(encoded_input['input_ids'].shape[0]):
                tokenization_result.append(self.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][i]))
            output["tokenization_result"] = tokenization_result
        total_tokens = encoded_input["input_ids"].shape[0] * encoded_input["input_ids"].shape[1]
        output["usage"] = {}
        output["usage"]["completion_tokens"] = 0
        output["usage"]["total_tokens"] = total_tokens
        output["usage"]["prompt_tokens"] = total_tokens
        return output


    def inference_with_sentence_tranformers(self, input_batch):
        logger.debug("running inference, input_batch: {}".format(input_batch))
        sentence_embedding = self.model.encode(input_batch, show_progress_bar=False, normalize_embeddings=True)
        output = defaultdict()
        output["sentence_embedding"] = sentence_embedding.tolist()
        max_length = self.model.get_max_seq_length()
        tokenization_result = [self.model.tokenizer.tokenize(s, truncation=True, max_length=max_length) for s in input_batch]
        if self.parameters.get("return_tokenization_result", False):
            output["tokenization_result"] = tokenization_result
        total_tokens = sum([len(d) for d in tokenization_result])
        output["usage"] = {}
        output["usage"]["completion_tokens"] = 0
        output["usage"]["total_tokens"] = total_tokens
        output["usage"]["prompt_tokens"] = total_tokens
        return output


    def postprocess(self, inference_output):
        logger.debug("running postprocess, inference_output: {}".format(inference_output))
        # convert result to list to solve "Invalid model predict output" error
        resp = {
            "code": 0,
            "type": "success",
            "message": "success",
            "data": inference_output,
        }
        return [json.dumps(resp)]


    def describe_handle(self):
        logger.debug("running describe_handle")
        output_describe = {
            "model_type": "text embedding",
            "model_source": "https://huggingface.co/BAAI/tao-8k",
            "model_git_verson": "4649ce5aeab1b5d13c07bb4f2c1337b3ba03eb05",
            "input": {
                "element_type": "string",
                "shape": "[batch_size, -1]",
                "sequence_length": self.model.get_max_seq_length(),
            },
            "output": {
                "element_type": "float",
                "shape": "[batch_size, {}]".format(self.model.get_sentence_embedding_dimension()),
            },
        }
        return json.dumps(output_describe)
