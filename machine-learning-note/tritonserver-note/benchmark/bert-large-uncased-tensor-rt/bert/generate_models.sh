#!/bin/bash

cd `dirname $(readlink -f $0)`
python -m pip install transformers -i https://mirrors.aliyun.com/pypi/simple/
python onnx_exporter.py
trtexec --onnx=model.onnx --saveEngine=model_bs16.plan --minShapes=token_ids:1x512,attn_mask:1x512 --optShapes=token_ids:16x512,attn_mask:16x512 --maxShapes=token_ids:128x512,attn_mask:128x512 --fp16 --verbose | tee conversion_bs16_dy.txt