# Deploy Huggingface Transformers using torchserve

## 目的

调研 torchserve 的使用方法, 如何对服务进行简单的配置.

## 简单上手

参考 [Get started](https://pytorch.org/serve/getting_started.html)

注意事项:
- 先用 CPU 推理进行测试, 可以规避一部分库的兼容问题
- 需要安装 jdk 11, 参考: [how to install java on debian 11](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-debian-11#step-2-managing-java)

总结一下主要步骤:
- 编写 handler, 定义请求的处理过程, 包括 initialize -> preprocess -> inference -> postprocess. 可以继承 [BaseHandler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py), 复用主体框架, 然后实现本模型的处理逻辑.
- 使用 torch-model-archiver 打包模型
- 使用 torchserve 加载打包的文件, 进行在线推理


## 部署 huggingface transformer

官方例子: [Serving Huggingface Transformers using TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers)

### 下载模型

待部署的模型: https://huggingface.co/BAAI/bge-large-zh, 下载时需要安装 git-lfs: ``sudo apt install git-lfs``

### 编写 handler

- 参考: [how to write custom handler](https://pytorch.org/serve/custom_service.html)
- [bge 模型 handler 示例](./bge_handler.py)
- [amu/tao-8k 模型 handler 示例](./tao_8k_handler.py)



### 打包模型

了解 torch-model-archiver 的使用方法: [introduction to torch-model-archiver](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)

```bash
# tree
.
├── 1_Pooling
│   └── config.json
├── README.md
├── config.json # 模型配置文件
├── config_sentence_transformers.json # 模型训练时的包依赖, 推理环境最好保持一致
├── handler.py  # 上一步生成的请求处理 handler
├── modules.json
├── pytorch_model.bin  # 模型 checkpoint, 包含模型权重, 结构信息
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── vocab.txt # 词表
```


#### 打包命令

```bash
torch-model-archiver -f --model-name bge_large_zh --version 1.0 --serialized-file bge-large-zh/pytorch_model.bin --handler bge-large-zh/handler.py --extra-files "bge-large-zh/" 

# 为了加速调试, 可以打包格式可以不选 mar
torch-model-archiver -f --model-name=bge_large_zh --version=1.0 --serialized-file=bge-large-zh/pytorch_model.bin --handler=bge-large-zh/handler.py --extra-files="bge-large-zh/" --archive-format=no-archive --export-path=no-archive
```


### 部署模型

了解 torchserve:
- [basic usage](https://pytorch.org/serve/server.html)
- [advanced configuration](https://pytorch.org/serve/configuration.html)

配置文件:
- [torchserve.config](https://github.com/pytorch/serve/blob/master/docker/config.properties)
- [log4j2.xml](https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml)

```bash
# cat torchserve.config 
# basic command options: https://pytorch.org/serve/server.html
# how to configure torchserve: https://pytorch.org/serve/configuration.html
# bind inference API to all network interfaces with SSL enabled
inference_address=http://0.0.0.0:8080
management_address=http://127.0.0.1:8081
metrics_address=http://127.0.0.1:8082
# set default_workers_per_model to 1 to prevent server from oom when debugging
default_workers_per_model=32
# Allow model specific custom python packages, Be cautious: it will slow down model loading
#install_py_dep_per_model=true
# log configuration: https://pytorch.org/serve/logging.html#modify-the-behavior-of-the-logs
# config demo: https://github.com/pytorch/serve/blob/master/frontend/server/src/main/resources/log4j2.xml 
async_logging=true
#vmargs=-Dlog4j.configurationFile=file:///volcvikingdb-model-store/log4j2.xml
#vmargs=-Dlog4j.configurationFile=file:///root/code/huggingface_store/log4j2.xml


# 启动 torchserve, 推理接口端口默认是 8080
torchserve --start --ncs --model-store=model-store/ --models bge_large_zh=bge_large_zh.mar --ts-config=torchserve.config
# 停止服务. 服务停止后, 后台会自动拉起
torchserve --stop
# 如果打包时没压缩, 可以使用下面的启动命令
torchserve --start --ncs --model-store=no-archive --models bge_large_zh=bge_large_zh --ts-config=torchserve.config
```


### 在线推理

API 介绍: [torchserve REST API](https://pytorch.org/serve/rest_api.html)

部署镜像:

```Dockerfile
FROM pytorch/torchserve:0.8.1-gpu
USER root
#ENV PIP_INDEX_URL=https://***/pypi/simple/ # switch to private source
RUN apt-get update && apt-get install -yq --no-install-recommends curl wget less
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir sentence_transformers==2.2.2
```


```bash
# 请求方式1: 上传文件
echo '{"input":"教练, 我想打篮球."}' > note.txt
curl -X POST  http://localhost:8080/predictions/bge_large_zh -T note.txt
# 请求方式2
curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":"如何使用torchserve部署模型"}'
# 简单压测
for i in $(seq 1000); do curl -X POST --header 'Content-Type: application/json'  curl -X POST --header 'Content-Type: application/json'  http://localhost:8080/predictions/bge_large_zh --data-raw '{"input":["教练, 我想打篮球.", "如何使用torchserve部署模型", "怎么训练bert模型", "怎么使用tensorflow训练bert模型", "怎么使用tfserving部署bert模型"]}'; done
```

查看 inference API 定义:

```bash
# https://github.com/pytorch/serve/blob/master/frontend/server/src/test/resources/inference_open_api.json
# curl -X OPTIONS http://localhost:8080
{
  "openapi": "3.0.1",
  "info": {
    "title": "TorchServe APIs",
    "description": "TorchServe is a flexible and easy to use tool for serving deep learning models",
    "version": "0.8.1"
  },
  "paths": {
    "/predictions/{model_name}": {
      "post": {
        "description": "Predictions entry point to get inference using default model version.",
        "operationId": "predictions",
        "parameters": [
          {
            "in": "path",
            "name": "model_name",
            "description": "Name of model.",
            "required": true,
            "schema": {
              "type": "string"
            }
          }
        ],
        "requestBody": {
          "description": "Input data format is defined by each model.",
          "content": {
            "*/*": {
              "schema": {
                "type": "string",
                "format": "binary"
              }
            }
          },
          "required": true
        },
        "responses": {
          "200": {
            "description": "Output data format is defined by each model.",
            "content": {
              "*/*": {
                "schema": {
                  "type": "string",
                  "format": "binary"
                }
              }
            }
          },
        ...
```


### 获取当前模型部署的模型列表

参考文档:  [MANAGEMENT API](https://pytorch.org/serve/management_api.html#management-api)

```bash
# curl "http://localhost:8081/models"
{
  "models": [
    {
      "modelName": "bge_large_zh",
      "modelUrl": "bge_large_zh.mar"
    }
  ]
}
```


## 问题记录

- 在线服务上没用没使用 GPU 卡: [how to make huggingface transformer model use gpu](https://github.com/huggingface/transformers/issues/2704)
- [把 huggingface transformer导出成 TorchScript 格式](https://huggingface.co/docs/transformers/torchscript)
- [torch-model-archiver 怎么打包整个文件夹](https://github.com/pytorch/serve/issues/1227)
- [对同一个文本, 使用不同的 batch size, batch_size=1 和 batch_size>1时得到的 embedding 结果有细微差别](https://huggingface.co/BAAI/bge-large-zh/discussions/5)


## 参考文档

- [deploy huggingface bert to production with torchserve](https://medium.com/analytics-vidhya/deploy-huggingface-s-bert-to-production-with-pytorch-serve-27b068026d18)
- [BERT TorchServe Tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/inference/tutorial-torchserve-neuronx.html)
- [A Quantitative Comparison of Serving Platforms for Neural Networks](https://biano-ai.github.io/research/2021/08/16/quantitative-comparison-of-serving-platforms-for-neural-networks.html)