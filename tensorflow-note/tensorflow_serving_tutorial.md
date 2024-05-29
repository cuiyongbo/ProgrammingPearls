# Tensorflow Serving inference tutorial

**USE server log to trace how to load a model**


## devbox setup

```bash

# ~/.bash_profile
# display IP in bash prompt
export PS1='\u@$(hostname -I) \w\n> '
# sugar commands
alias cdw='cd /home/$USER/workspace'
alias ll='ls -lh'
alias grep='grep --color'
alias tailf='tail -f'
export TFHUB_CACHE_DIR=/home/$USER/workspace/keras_store/tfhub_models
source /usr/local/lib/bazel/bin/bazel-complete.bash
```

## 编译 tensorflow-serving

```bash
# install package dependencies according to Dockerfile: tensorflow/serving/tensorflow_serving/tools/docker/Dockerfile.devel
# or from Dockerfile of tensorflow/serving from docker hub: https://hub.docker.com/r/tensorflow/serving

# less serving/WORKSPACE 
# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("6.1.0")

# run bazel commands with ``--experimental_repo_remote_exec`` to avoid failure

# list all build targets
./tools/run_in_docker.sh bazel query tensorflow_serving/... --experimental_repo_remote_exec --verbose_failures

# display the location of a build target
./tools/run_in_docker.sh bazel cquery //tensorflow_serving/model_servers:tensorflow_model_server --experimental_repo_remote_exec --output=files
# bazel-out/k8-opt/bin/tensorflow_serving/model_servers/tensorflow_model_server

# build a specified target
./tools/run_in_docker.sh bazel build --config=release //tensorflow_serving/model_servers:tensorflow_model_server --experimental_repo_remote_exec --verbose_failures --copt=-Wno-error=maybe-uninitialized
./tools/run_in_docker.sh bazel build --config=release //tensorflow_serving/example:resnet_client_cc --experimental_repo_remote_exec --verbose_failures

# build in background
nohup bazel build --config=release //tensorflow_serving/model_servers:tensorflow_model_server >&1 2>&1 > compilation.log &
# re-run bazel build with `--experimental_local_memory_estimate --local_ram_resources=HOST_RAM*0.8` if compilation failed
nohup bazel build --config=release //tensorflow_serving/model_servers:tensorflow_model_server --experimental_repo_remote_exec --experimental_local_memory_estimate --local_ram_resources=HOST_RAM*0.8 >&1 2>&1 > compilation.log &

bazel build --config=release //tensorflow_serving/model_servers:tensorflow_model_server --experimental_repo_remote_exec --experimental_local_memory_estimate --local_ram_resources=HOST_RAM*0.8

# git diff
diff --git a/.bazelrc b/.bazelrc
index 085f520b..1c21ad50 100644
--- a/.bazelrc
+++ b/.bazelrc
@@ -65,6 +65,8 @@ build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0
 
 build --experimental_repo_remote_exec
+build --local_ram_resources=HOST_RAM*0.8
+build --experimental_local_memory_estimate

```


## 开发环境

```bash
# computer: MacBook Pro with Apple M1 Pro chip

# python --version
Python 3.9.6

# install tensorflow: https://developer.apple.com/metal/tensorflow-plugin/
# pip freeze | grep tensorflow
tensorflow==2.13.0rc1
tensorflow-datasets==4.9.2
tensorflow-estimator==2.13.0rc0
tensorflow-hub==0.13.0
tensorflow-macos==2.13.0rc1
tensorflow-metadata==1.13.1
tensorflow-metal==1.0.1
tensorflow-model-optimization==0.7.3

# docker images
REPOSITORY                   TAG                            IMAGE ID       CREATED         SIZE
emacski/tensorflow-serving   latest                         21d9dee010a7   22 months ago   377MB
```

## 加载模型

```bash
# 示例模型: https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5
cherry@QLK23GVKXR ~/keras_data/tfhub_modules/inception_resnet_v2
# tree
.
└── 5
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index

3 directories, 3 files
cherry@QLK23GVKXR ~/keras_data/tfhub_modules/inception_resnet_v2

# 用 docker 启动 tensorflow-serving, 注意要要发布 http 和 grpc 端口
# docker run -t --rm -p 8501:8501  -p 8500:8500 -v'/Users/cherry/keras_data/tfhub_modules/inception_resnet_v2:/models/inception_resnet_v2' -e MODEL_NAME=inception_resnet_v2 emacski/tensorflow-serving
2023-07-12 15:30:46.895801: I external/tf_serving/tensorflow_serving/model_servers/server.cc:89] Building single TensorFlow model file config:  model_name: inception_resnet_v2 model_base_path: /models/inception_resnet_v2
2023-07-12 15:30:46.896080: I external/tf_serving/tensorflow_serving/model_servers/server_core.cc:465] Adding/updating models.
2023-07-12 15:30:46.896095: I external/tf_serving/tensorflow_serving/model_servers/server_core.cc:591]  (Re-)adding model: inception_resnet_v2
2023-07-12 15:30:47.003164: I external/tf_serving/tensorflow_serving/core/basic_manager.cc:740] Successfully reserved resources to load servable {name: inception_resnet_v2 version: 5}
2023-07-12 15:30:47.003209: I external/tf_serving/tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: inception_resnet_v2 version: 5}
2023-07-12 15:30:47.003221: I external/tf_serving/tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: inception_resnet_v2 version: 5}
2023-07-12 15:30:47.003971: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:38] Reading SavedModel from: /models/inception_resnet_v2/5
2023-07-12 15:30:47.072230: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:90] Reading meta graph with tags { serve }
2023-07-12 15:30:47.072272: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:132] Reading SavedModel debug info (if present) from: /models/inception_resnet_v2/5
2023-07-12 15:30:47.076980: I external/org_tensorflow/tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2023-07-12 15:30:47.205726: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:211] Restoring SavedModel bundle.
2023-07-12 15:30:47.214927: W external/org_tensorflow/tensorflow/core/platform/profile_utils/cpu_utils.cc:87] Failed to get CPU frequency: -1
2023-07-12 15:30:49.309540: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:195] Running initialization op on SavedModel bundle at path: /models/inception_resnet_v2/5
2023-07-12 15:30:49.404863: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:283] SavedModel load for tags { serve }; Status: success: OK. Took 2400891 microseconds.
2023-07-12 15:30:49.417529: I external/tf_serving/tensorflow_serving/servables/tensorflow/saved_model_warmup_util.cc:59] No warmup data file found at /models/inception_resnet_v2/5/assets.extra/tf_serving_warmup_requests
2023-07-12 15:30:49.419595: I external/tf_serving/tensorflow_serving/core/loader_harness.cc:87] Successfully loaded servable version {name: inception_resnet_v2 version: 5}
2023-07-12 15:30:49.420203: I external/tf_serving/tensorflow_serving/model_servers/server_core.cc:486] Finished adding/updating models
2023-07-12 15:30:49.420248: I external/tf_serving/tensorflow_serving/model_servers/server.cc:133] Using InsecureServerCredentials
2023-07-12 15:30:49.420260: I external/tf_serving/tensorflow_serving/model_servers/server.cc:383] Profiler service is enabled
2023-07-12 15:30:49.421332: I external/tf_serving/tensorflow_serving/model_servers/server.cc:409] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2023-07-12 15:30:49.423147: I external/tf_serving/tensorflow_serving/model_servers/server.cc:430] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 245] NET_LOG: Entering the event loop ...
```

## 查询模型状态和元信息

```bash
# 1. 查看模型版本, 状态
# curl 'http://localhost:8501/v1/models/inception_resnet_v2'
{
 "model_version_status": [
  {
   "version": "5",
   "state": "AVAILABLE",
   "status": {
    "error_code": "OK",
    "error_message": ""
   }
  }
 ]
}

# 2. 查看模型元信息
# curl 'http://localhost:8501/v1/models/inception_resnet_v2/metadata'
{
"model_spec":{ 
 "name": "inception_resnet_v2", # request.model_spec.name
 "signature_name": "",
 "version": "5"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": { # request.model_spec.signature_name
   "inputs": {
    "inputs": { # input alias, request.inputs['inputs']
     "dtype": "DT_FLOAT",
     "tensor_shape": { # input shape: [-1, -1, -1, 3]
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "3",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_inputs:0" # actual tensor name in graph
    }
   },
   "outputs": {
    "feature_vector": { # output alias, outputs.keys
     "dtype": "DT_FLOAT",
     "tensor_shape": { # output shape: [-1, 1536]
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1536",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "StatefulPartitionedCall:0" # actual tensor name in graph
    }
   },
   "method_name": "tensorflow/serving/predict"
  },
  ...
}
```

## 模型推理
### Python 模型推理
#### http 接口

```py
import requests, json
import tensorflow as tf

# 图片预处理
img_path = '/Users/cherry/.keras/datasets/flower_photos/sunflowers/5923085671_f81dd1cf6f.jpg'
img = tf.io.read_file(img_path)
img = tf.io.decode_jpeg(img, channels=3, )
img = tf.image.convert_image_dtype(img, tf.float32)
img = img/255.0
img = tf.image.resize(img, [240, 240])
img = tf.expand_dims(img, axis=0)
img.shape
# TensorShape([1, 240, 240, 3])

headers = {"content-type": "application/json"}
payload = {"signature_name": "serving_default"}
payload["instances"] = img.numpy().tolist()
resp = requests.post('http://localhost:8501/v1/models/inception_resnet_v2:predict', data=json.dumps(payload), headers=headers)
result = resp.json()
result.keys()
len(result['predictions'])
#1
len(result['predictions'][0])
#1536
print(result['predictions'][0][:20])
#[0.00348012405, 0.0382439755, 0.022029506, 0.0103815347, 0.417011887, 0.222013533, 0.0, 0.00629411358, 0.0, 0.0836884752, 0.173402548, 0.00420226483, 0.0483214147, 0.0975027233, 0.0, 0.0, 0.211568788, 0.41907075, 0.159508303, 0.00715918839]
```

tensorflow 提供的使用 http 接口的脚本(注意使用的模型和这里使用的不一样):

```py
"""A client that performs inferences on a ResNet model using the REST API.

The client downloads a test image of a cat, queries the server over the REST API
with the test image repeatedly and measures how long it takes to respond.

The client expects a TensorFlow Serving ModelServer running a ResNet SavedModel
from: https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/classification/5
"""

from __future__ import print_function

import base64
import io
import json

import numpy as np
from PIL import Image
import requests

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/inception_resnet_v2:predict'

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False

def main():
  # Download the image
  dl_request = requests.get(IMAGE_URL, stream=True)
  dl_request.raise_for_status()

  if MODEL_ACCEPT_JPG:
    # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
  else:
    # Compose a JOSN Predict request (send the image tensor).
    jpeg_rgb = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
    predict_request = json.dumps({'instances': jpeg_rgb})

  # Send few requests to warm-up the model.
  print("start warmup")
  for _ in range(3):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()

  # Send few actual requests and report average latency.
  print("start request")
  total_time = 0
  num_requests = 10
  for _ in range(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()
    prediction = response.json()['predictions'][0]

  print('Prediction class: {}, avg latency: {} ms'.format(
      np.argmax(prediction), (total_time * 1000) / num_requests))

if __name__ == '__main__':
  main()
```


#### grpc 接口

```bash
# pwd
/Users/cherry/dev-repo/tensorflow/serving

# 把 util 脚本放到 python 的search path
# export PYTHONPATH=$PYTHONPATH:/Users/cherry/dev-repo/tensorflow/serving

# 生成用到的 proto 模块: model, predict, classification 等等
# python -m grpc.tools.protoc --python_out=. --grpc_python_out=. -I. -I path/to/tensorflow/  tensorflow_serving/apis/*.proto

# 执行测试脚本
python /Users/cherry/dev-repo/tensorflow/serving/tensorflow_serving/example/resnet_client_grpc.py
```


为了方便调试, 我把 resnet_client_grpc.py 也放进来了

```py
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.
"""

from __future__ import print_function

import io

# This is a placeholder for a Google-internal import.

import grpc
import numpy as np
from PIL import Image
import requests
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('image', '',
                                     'path to image in JPEG format')
FLAGS = tf.compat.v1.app.flags.FLAGS

# Current Resnet model in TF Model Garden (as of 7/2021) does not accept JPEG
# as input
MODEL_ACCEPT_JPG = False

def main(_):
  if FLAGS.image:
    with open(FLAGS.image, 'rb') as f:
      data = f.read()
  else:
    # Download the image since we weren't given one
    dl_request = requests.get(IMAGE_URL, stream=True)
    dl_request.raise_for_status()
    data = dl_request.content

  if not MODEL_ACCEPT_JPG:
    data = Image.open(io.BytesIO(dl_request.content))
    # Normalize and batchify the image
    data = np.array(data) / 255.0
    data = np.expand_dims(data, 0)
    data = data.astype(np.float32)

  channel = grpc.insecure_channel(FLAGS.server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  # Send request
  # See prediction_service.proto for gRPC request/response details.
  # 如何填写请求参数请参见 metadata 接口输出
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception_resnet_v2'
  request.model_spec.signature_name = 'serving_default'
  request.inputs['inputs'].CopyFrom(
      tf.make_tensor_proto(data))
  result = stub.Predict(request, 100.0)  # 10 secs timeout
  print(result)

if __name__ == '__main__':
  tf.compat.v1.app.run()

'''
outputs {
  key: "feature_vector"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 1536
      }
    }
    float_val: 0.18572344
    float_val: 0.092920996
    float_val: 0.013378174
    ...
  }
}
model_spec {
  name: "inception_resnet_v2"
  version {
    value: 5
  }
  signature_name: "serving_default"
}
'''
```

### C++ 模型推理
c++ client 示例: resnet_client.cc

```bash
# 使用的模型：https://hub.tensorflow.google.cn/google/imagenet/inception_v1/classification/5
cherry@QLK23GVKXR ~/dev-repo/tensorflow/serving
# git checkout -B r2.13 origin/r2.13

# bazel build --verbose_failures //tensorflow_serving/example:resnet_client_cc

# ./bazel-bin/tensorflow_serving/example/resnet_client_cc
dyld[20552]: symbol not found in flat namespace '_CFRelease'
Abort trap: 6
```

```c++
#include <setjmp.h>

#include <fstream>
#include <iostream>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "google/protobuf/map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/jpeg.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<tensorflow::string, tensorflow::TensorProto> OutMap;

struct tf_jpeg_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf setjmp_buffer;
};

typedef struct tf_jpeg_error_mgr* tf_jpeg_error_ptr;

METHODDEF(void)
tf_jpeg_error_exit(j_common_ptr cinfo) {
  tf_jpeg_error_ptr tf_jpeg_err = (tf_jpeg_error_ptr)cinfo->err;

  (*cinfo->err->output_message)(cinfo);

  longjmp(tf_jpeg_err->setjmp_buffer, 1);
}

class ServingClient {
 public:
  // JPEG decompression code following libjpeg-turbo documentation:
  // https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/example.txt
  int readJPEG(const char* file_name, tensorflow::TensorProto* proto) {
    struct tf_jpeg_error_mgr jerr;
    FILE* infile;
    JSAMPARRAY buffer;
    int row_stride;
    struct jpeg_decompress_struct cinfo;

    if ((infile = fopen(file_name, "rb")) == NULL) {
      fprintf(stderr, "can't open %s\n", file_name);
      return -1;
    }

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = tf_jpeg_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
      jpeg_destroy_decompress(&cinfo);
      fclose(infile);
      return -1;
    }

    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);

    (void)jpeg_read_header(&cinfo, TRUE);

    (void)jpeg_start_decompress(&cinfo);
    row_stride = cinfo.output_width * cinfo.output_components;
    CHECK(cinfo.output_components == 3)
        << "Only 3-channel (RGB) JPEG files are supported";

    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                        row_stride, 1);

    proto->set_dtype(tensorflow::DataType::DT_FLOAT);
    while (cinfo.output_scanline < cinfo.output_height) {
      (void)jpeg_read_scanlines(&cinfo, buffer, 1);
      for (size_t i = 0; i < cinfo.output_width; i++) {
        proto->add_float_val(buffer[0][i * 3] / 255.0);
        proto->add_float_val(buffer[0][i * 3 + 1] / 255.0);
        proto->add_float_val(buffer[0][i * 3 + 2] / 255.0);
      }
    }

    proto->mutable_tensor_shape()->add_dim()->set_size(1);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_height);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_width);
    proto->mutable_tensor_shape()->add_dim()->set_size(cinfo.output_components);

    (void)jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return 0;
  }

  ServingClient(std::shared_ptr<Channel> channel)
      : stub_(PredictionService::NewStub(channel)) {}

  tensorflow::string callPredict(const tensorflow::string& model_name,
                                 const tensorflow::string& model_signature_name,
                                 const tensorflow::string& file_path) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(model_name);
    predictRequest.mutable_model_spec()->set_signature_name(
        model_signature_name);

    google::protobuf::Map<tensorflow::string, tensorflow::TensorProto>& inputs =
        *predictRequest.mutable_inputs();

    tensorflow::TensorProto proto;

    const char* infile = file_path.c_str();

    if (readJPEG(infile, &proto)) {
      std::cout << "error constructing the protobuf";
      return "execution failed";
    }

    inputs["input_1"] = proto;

    Status status = stub_->Predict(&context, predictRequest, &response);

    if (status.ok()) {
      std::cout << "call predict ok" << std::endl;
      std::cout << "outputs size is " << response.outputs_size() << std::endl;
      OutMap& map_outputs = *response.mutable_outputs();
      OutMap::iterator iter;
      int output_index = 0;

      for (iter = map_outputs.begin(); iter != map_outputs.end(); ++iter) {
        tensorflow::TensorProto& result_tensor_proto = iter->second;
        tensorflow::Tensor tensor;
        bool converted = tensor.FromProto(result_tensor_proto);
        if (converted) {
          std::cout << "the result tensor[" << output_index
                    << "] is:" << std::endl
                    << tensor.SummarizeValue(1001) << std::endl;
        } else {
          std::cout << "the result tensor[" << output_index
                    << "] convert failed." << std::endl;
        }
        ++output_index;
      }
      return "Done.";
    } else {
      std::cout << "gRPC call return code: " << status.error_code() << ": "
                << status.error_message() << std::endl;
      return "gRPC failed.";
    }
  }

 private:
  std::unique_ptr<PredictionService::Stub> stub_;
};

int main(int argc, char** argv) {
  tensorflow::string server_port = "localhost:8500";
  tensorflow::string image_file = "";
  tensorflow::string model_name = "resnet";
  tensorflow::string model_signature_name = "serving_default";
  std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("server_port", &server_port,
                       "the IP and port of the server"),
      tensorflow::Flag("image_file", &image_file, "the path to the image"),
      tensorflow::Flag("model_name", &model_name, "name of model"),
      tensorflow::Flag("model_signature_name", &model_signature_name,
                       "name of model signature")};

  tensorflow::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result || image_file.empty()) {
    std::cout << usage;
    return -1;
  }

  ServingClient guide(
      grpc::CreateChannel(server_port, grpc::InsecureChannelCredentials()));
  std::cout << "calling predict using file: " << image_file << "  ..."
            << std::endl;
  std::cout << guide.callPredict(model_name, model_signature_name, image_file)
            << std::endl;
  return 0;
}
```


## 问题整理

* resnet_client_grpc.py 执行失败, 提示 ModuleNotFoundError

把这些脚本放到python包的搜索路径里: ``export PYTHONPATH=$PYTHONPATH:/Users/cherry/dev-repo/tensorflow/serving``
```
Traceback (most recent call last):
  File "/Users/cherry/dev-repo/tensorflow/serving/tensorflow_serving/example/resnet_client_grpc.py", line 31, in <module>
    from tensorflow_serving.apis import predict_pb2
ModuleNotFoundError: No module named 'tensorflow_serving'
```

* resnet_client_grpc.py rpc 请求失败

报错日志如下. 原因是docker启动时没有把 grpc 端口暴露出来.

```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
        details = "failed to connect to all addresses"
        debug_error_string = "{"created":"@1689230909.707902000","description":"Failed to pick subchannel","file":"src/core/ext/filters/client_channel/client_channel.cc","file_line":3261,"referenced_errors":[{"created":"@1689230909.707902000","description":"failed to connect to all addresses","file":"src/core/lib/transport/error_utils.cc","file_line":167,"grpc_status":14}]}"
```

* 执行 c++ resnet_client_cc 失败 [未解决]

```
# ./bazel-bin/tensorflow_serving/example/resnet_client_cc
dyld[20552]: symbol not found in flat namespace '_CFRelease'
Abort trap: 6
# 换个 windows 电脑，装了个 WSL ubuntu 20.04 后重新编译运行后可以了
root@PF20AAFA-IBX:~/linux_dev_repo/tensorflow/serving# ./bazel-bin/tensorflow_serving/example/resnet_client_cc --image_file=/home/cherry/linux_dev_repo/tensorflow/tensorflow/tensorflow/lite/g3doc/inference_with_metadata/task_library/images/dogs.jpg --model_name=inception_resnet_v2
calling predict using file: /home/cherry/linux_dev_repo/tensorflow/tensorflow/tensorflow/lite/g3doc/inference_with_metadata/task_library/images/dogs.jpg  ...
call predict ok
outputs size is 1
the result tensor[0] is:
[-0.668525815 0.418405086 -0.58332926 -0.42304197 -0.69078815 0.171937108 -0.259618402 
...
```

* more

## 参考文档

- [Tensorflow Serving RESTful API](https://www.tensorflow.org/tfx/serving/api_rest)
- [SignatureDefs in SavedModel for TensorFlow Serving](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md)