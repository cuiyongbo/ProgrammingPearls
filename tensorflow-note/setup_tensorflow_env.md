# Setup Tensorflow for R&D

- Install tensorflow
    - install tensorflow using pip: ``pip install --upgrade tensorflow``
        - [Install tensorflow](https://tensorflow.google.cn/install)
        - [Install tensorflow on apple m1 machine](https://developer.apple.com/metal/tensorflow-plugin/)
    - Install using docker (*recommended*)
        - [tensorflow on dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/tags)
    ```bash
    docker pull tensorflow/tensorflow:latest  # Download latest stable image
    docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server
    # install tensorflow-serving container for macbook with m1 chip: https://github.com/tensorflow/serving/issues/1816
    # verify the GPU setup
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

- Install tensorflow tools

```bash
pip3 install "tensorflow>=2.0.0"
pip3 install --upgrade tensorflow-hub
pip3 install tensorflow-datasets
pip3 install --upgrade tensorflow-model-optimization
pip3 install tfds-nightly
pip3 install tensorflow-text
pip3 install tensorflow_addons
pip3 install tf-models-official
```

- start tfx container

```bash
# fetch tfx image
docker pull tensorflow/tfx
# if you encounter 'No id provided.' error when running tfx container, you may need specify `--entrypoint` option
docker run -it --mount type=bind,src=$(pwd),dst=/workspace --entrypoint bash tensorflow/tfx
#docker run -p 33243:6006 -ti --entrypoint bash --mount type=bind,src=/opt/home/cuiyongbo/docker-scaffold,dst=/workspace 0fbc116a552e
cd && mkdir .keras && cd .keras/ && ln -fs /workspace/datasets/ datasets
# attach to a running container
docker container exec -it bffd65ffbadb bash
# install tensorflow-doc
pip3 install git+https://github.com/tensorflow/docs
```

- [start tensorflow-serving with docker](https://tensorflow.google.cn/tfx/serving/docker)

```bash
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving
# docker pull emacski/tensorflow-serving # for macbook with m1 chip

# clone tensorflow-serving for model demos
git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# peek model structure with saved_model_cli
saved_model_cli show --dir $TESTDATA/saved_model_half_plus_two_cpu/00000123/ --tag_set serve
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"

saved_model_cli show --dir $TESTDATA/saved_model_half_plus_two_cpu/00000123/ --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
inputs['x'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: x:0
The given SavedModel SignatureDef contains the following output(s):
outputs['y'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 1)
    name: y:0
Method name is: tensorflow/serving/predict

# docker run -t --rm -p 8501:8501 -p 8500:8500 -v'/Users/cherry/keras_data/tfhub_modules/inception_resnet_v2:/models/inception_resnet_v2' -e MODEL_NAME=inception_resnet_v2 emacski/tensorflow-serving

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
# Return: { "predictions": [2.5, 3.0, 4.5] }
```

- [tensorflow-serving API spec](https://tensorflow.google.cn/tfx/serving/api_rest)

```bash
# load pre-trained mnist model demo
docker run -t --rm -p 8501:8501 -v'/tmp/mnist:/models/mnist' -e MODEL_NAME=mnist emacski/tensorflow-serving

# run inference with python
import requests, json
headers = {"content-type": "application/json"}
data = json.dumps({"signature_name": "serving_default", "instances": test_images[0:3].tolist()})
json_response = requests.post('http://localhost:8501/v1/models/mnist:predict', data=data, headers=headers)
pred = json_response.json()['predictions']
np.argmax(pred, axis=1)
# array([7, 2, 1])

# curl 'http://localhost:8501/v1/models/mnist'
{
    "model_version_status": [
        {
            "version": "2",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        },
        {
            "version": "1",
            "state": "END",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}

# curl 'http://localhost:8501/v1/models/mnist/versions/3'
{
    "model_version_status": [
    {
    "version": "3",
    "state": "AVAILABLE",
    "status": {
    "error_code": "OK",
    "error_message": ""
    }
    }
    ]
}

# curl 'http://localhost:8501/v1/models/mnist/metadata'
{
    "model_spec": {
        "name": "mnist",
        "signature_name": "",
        "version": "2"
    },
    "metadata": {
        "signature_def": {
            "signature_def": {
                "serving_default": {
                    "inputs": {
                        "dense_input": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "784",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "serving_default_dense_input:0"
                        }
                    },
                    "outputs": {
                        "dense_1": {
                            "dtype": "DT_FLOAT",
                            "tensor_shape": {
                                "dim": [
                                    {
                                        "size": "-1",
                                        "name": ""
                                    },
                                    {
                                        "size": "10",
                                        "name": ""
                                    }
                                ],
                                "unknown_rank": false
                            },
                            "name": "StatefulPartitionedCall:0"
                        }
                    },
                    "method_name": "tensorflow/serving/predict"
                },
                "__saved_model_init_op": {
                    "inputs": {},
                    "outputs": {
                        "__saved_model_init_op": {
                            "dtype": "DT_INVALID",
                            "tensor_shape": {
                                "dim": [],
                                "unknown_rank": true
                            },
                            "name": "NoOp"
                        }
                    },
                    "method_name": ""
                }
            }
        }
    }
}
```


- python3 to start tensorboard: ``python3 -m tensorboard.main --logdir=/path/to/logs``

- Supress tensorflow warnings

```bash
# in scripts
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# in bash add environment variable
export TF_CPP_MIN_LOG_LEVEL=2
```

- start tensorflow in jupyter notebook

```bash
# https://hub.docker.com/r/jupyter/tensorflow-notebook
docker pull jupyter/tensorflow-notebook

docker run  -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/tensorflow-notebook

# attach to the running container so as to install addtional dependencies
docker -exec -u root -it container_id bash
```

- how to develop with a jupyter notebook server resident in a docker container
    - [Running the Notebook ](https://docs.jupyter.org/en/latest/running.html#starting-the-notebook-server)

```bash
# start docker container, keep it running in background
docker run --gpus all -t -d --rm --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/data-store/huggingface-store:/model-store nvcr.io/nvidia/pytorch:24.01-py3
# -t, --tty                              Allocate a pseudo-TTY
# -d, --detach                           Run container in background and print container ID
# --rm means Automatically remove the container when it exits
# --ipc string                       IPC mode to use #https://docs.docker.com/reference/cli/docker/container/run/#ipc
#   --ipc=host means sharing the host's IPC namespace, for example, shared memory
# --network=host                  start the container using the host's network, also mapping used ports to host

# docker container list
CONTAINER ID   IMAGE                                     COMMAND                  CREATED        STATUS        PORTS                                                           NAMES
cb64ff444240   nvcr.io/nvidia/pytorch:24.01-py3          "/opt/nvidia/nvidia_…"   2 hours ago    Up 2 hours                                                                    vibrant_jemison
ffb0aaab9336   nvcr.io/nvidia/tensorflow:24.03-tf2-py3   "/opt/nvidia/nvidia_…"   29 hours ago   Up 29 hours   6006/tcp, 6064/tcp, 8888/tcp                                    zen_ptolemy
5bba2a72277d   nvcr.io/nvidia/tritonserver:24.01-py3     "/opt/nvidia/nvidia_…"   2 months ago   Up 2 months   0.0.0.0:8000-8002->8000-8002/tcp, :::8000-8002->8000-8002/tcp   strange_matsumoto

# go inside the container
docker exec -u root -it cb64ff444240 bash

# start jupyter notebook server and keep it running in background
nohup jupyter notebook --no-browse --allow-root --autoreload --notebook-dir=/model-store &

# --no-browser
#     Don't open the notebook in a browser after startup.
# --allow-root
#     Allow the notebook to be run from root user.
#     Equivalent to: [--NotebookApp.allow_root=True]
# --autoreload
#     Autoreload the webapp
# --port=<Int>
#     The port the notebook server will listen on (env: JUPYTER_PORT).
#     Default: 8888
# --notebook-dir=<Unicode>
#    The directory to use for notebooks and kernels.

# list running jupyter notebook servers
# jupyter notebook list
Currently running servers:
http://0.0.0.0:8888/?token=49f59c67c533c9bf3d23e6c0243fcb3dc79f6480ff106609 :: /model-store
```
