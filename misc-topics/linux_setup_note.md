# Ubuntu Setup

Table Of Content:

- [Ubuntu Setup](#ubuntu-setup)
  - [Shortcuts](#shortcuts)
  - [Apt Packages](#apt-packages)
  - [Git Setup](#git-setup)
  - [Docker Setup](#docker-setup)
  - [Compiler ToolChain Setup](#compiler-toolchain-setup)
  - [Python Packages](#python-packages)
  - [Machine Learning Env Setup](#machine-learning-env-setup)
  - [Miscellaneous Topics](#miscellaneous-topics)


## Shortcuts

- Open Terminal: Ctrl+Alt+T
- Copy in Terminal: Ctrl+Shift+C
- Paste in Terminal: Ctrl+Shift+V
- Maximize Window: Alt+F10


## Apt Packages

- [How to use the traditional `vi` editor?](./traditional_vi_note.md)

- change apt source list: [aliyun unbuntu sources](https://developer.aliyun.com/mirror/ubuntu?spm=a2c6h.13651102.0.0.3e221b114p7WHD)

```bash
# lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy  # refer to configurations with the same Codename 
```

- install packages:

```bash
sudo apt update
sudo apt install -y vim tree htop net-tools

# setup ssh
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
sudo systemctl status ssh
```

- package settings:

```bash
# for vim
# in ~/.vimrc
set tabstop=4
set expandtab
syntax on
set hlsearch
set number
set ruler
set wrapscan
set list

# in ~/.bashrc
# display IP in bash prompt
export PS1='\u@$(hostname -I|cut -d" " -f1) \w\n# '

# in ~/.inputrc
# no newline when copying/pasting code block in python interpreter
set enable-bracketed-paste off

# in ~/.bashrc
# some aliases
alias cdw='cd /workspace'
alias ll='ls -lh'
alias grep='grep --color'
alias g++='g++ -std=c++11'
alias tailf='tail -f'
```

- [安装中文输入法](https://blog.csdn.net/windson_f/article/details/124932523)


## Git Setup

```bash
sudo apt update
sudo apt install -y git git-lfs

# set user identity
git config --global user.name "Cherry Luo"
git config --global user.email "csu20120504@126.com"

# set vim as the default commit message editor
git config --global core.editor vim

# force line ending to CRLF
git config --global core.autocrlf true

# save username and password
git config --global credential.helper store

# automatically remote deleted remote branches
git config --global --add fetch.prune true
```


## Docker Setup

- [install docker](hello-world/my_wiki/programmer_note/docker_note/docker_note.rst)


## Compiler ToolChain Setup

- GCC/G++
- GDB
- CMake
- [Protocol Buffer](hello-world/my_wiki/programmer_note/grpc/protobuf_faq.md)


## Python Packages

- install pip3: `sudo apt install python3-pip`
- change pip package index url

```bash
# pip3 install -h
  -i, --index-url <url>       Base URL of the Python Package Index (default https://pypi.org/simple). This should point to a repository compliant with PEP 503 (the simple repository API) or a local directory laid out in the same format.
  --extra-index-url <url>     Extra URLs of package indexes to use in addition to --index-url. Should follow the same rules as --index-url.

# aliyun images: https://developer.aliyun.com/mirror/pypi
# cmd to set: pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# cmd to get: pip config get global.index-url
# change forever: ~/.pip/pip.conf
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
[install]
trusted-host=mirrors.aliyun.com

# or install package with customized source
# pip3 option: -i, --index-url <url>  Base URL of the Python Package Index 
$ sudo -H pip3 install package_name -i https://mirrors.aliyun.com/pypi/simple/
```

- [requirements.txt for Python3.12](./py3_12_requirements.txt)
    - [jupyter notebook](https://docs.jupyter.org/en/latest/install.html)
    - [mermaid-python: draw mermaid diagram in jupyter notebook](https://pypi.org/project/mermaid-python/)
    - [Python code formatter: Black](https://pypi.org/project/black/)
    - [docarray](https://docs.docarray.org/)
    - [Pillow - image process lib](https://pillow.readthedocs.io)
    - sentence-transformers
    - matplotlib
    - pandas


## Machine Learning Env Setup

- install pytorch

```bash
# install pytorch: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio
```

- [Setup TensorFlow](../machine-learning-note/setup_tensorflow_env.md)
- [Setup nanoGPT](../machine-learning-note/transformer/setup_nanoGPT_env.md)
- [Setup Nvidia Triton Inference Server](../machine-learning-note/tritonserver-note/nvidia_triton_inference_server_note.md)
- [how to use panda?](./panda_abc_demo.ipynb)
- [NGC Containers](https://catalog.ngc.nvidia.com/containers)
    - **Be sure to check the compatibility between cuda and nividia driver before downloading NGC**
    - [CUDA Programming](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda)
    - [tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)
    - [pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
    - [tritonserver](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)

- Mirrors
    - HuggingFace Mirrors
        - https://modelscope.cn/models
        - https://hf-mirror.com/models
            - 修改 HF_ENDPOINT: 往 `~/.bashrc` 中注入: `export HF_ENDPOINT=https://hf-mirror.com`
    - GitHub Mirrors
        - https://gitee.com/

- start/stop Nvidia GPU: [start_gpu.sh](./start_gpu.sh)/[stop_gpu.sh](./stop_gpu.sh)

- how to change GPU fan speed?
```bash
# https://github.com/NVIDIA/open-gpu-kernel-modules/issues/395
# echo $DISPLAY
:1
# disable auto-control for fan-setting
sudo DISPLAY=:1   nvidia-settings -a GPUFanControlState=1
# change fan speed
sudo DISPLAY=:1   nvidia-settings -a GPUTargetFanSpeed=20
```


## Miscellaneous Topics

- [Redis](hello-world/my_wiki/programmer_note/redis_note.rst)
- Kafka


```
# docker images
REPOSITORY                                                       TAG                        IMAGE ID       CREATED             SIZE
pytorch-transformer                                              latest                     ccddd1385dc8   About an hour ago   4.68GB
nvcr.io/nvidia/tritonserver                                      24.12-py3                  33a797709c53   5 weeks ago         17.7GB
nvcr.io/nvidia/tensorflow                                        24.12-tf2-py3              80e44d2cc5d8   6 weeks ago         17.1GB
nvcr.io/nvidia/pytorch                                           24.12-py3                  eec0906cea58   6 weeks ago         21.7GB
nvidia/cuda                                                      11.8.0-base-ubuntu22.04    1e75b7decac0   14 months ago       239MB
swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda   11.8.0-base-ubuntu22.04    1e75b7decac0   14 months ago       239MB
nvidia/cuda                                                      12.0.1-devel-ubuntu22.04   678edd19cf8f   14 months ago       7.34GB
swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda   12.0.1-devel-ubuntu22.04   678edd19cf8f   14 months ago       7.34GB
zhaokundev/annotated-transformer                                 latest                     c8b3f21e0613   15 months ago       7.66GB

docker run --gpus all -t -d --rm --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/cherry/workspace:/workspace nvcr.io/nvidia/tritonserver:24.12-py3
docker run --gpus all -t -d --rm --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/cherry/workspace:/workspace nvcr.io/nvidia/pytorch:24.12-py3

docker run --d --rm --gpus all -ipc=host --network=host -v /home/cherry/workspace:/scaffold zhaokundev/annotated-transformer

docker run --gpus all -t -d --rm --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/cherry/workspace:/workspace nvidia/cuda:11.8.0-base-ubuntu22.04


pip3 install pandas spacy altair jupytext flake8 black GPUtil wandb

docker run --gpus all -t -d --rm --ipc=host --network=host -v /home/cherry/workspace:/workspace transformer-env:latest
docker run --gpus all -t -d --rm --ipc=host --network=host -v /home/cherry/workspace:/scaffold zhaokundev/annotated-transformer:latest
docker run --gpus all -t -d --rm --ipc=host --network=host -v /home/cherry/workspace:/workspace nvidia/cuda:12.0.1-devel-ubuntu22.04



sed -i 's/archive.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list
sed -i 's/security.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list

apt update
apt install -y vim python3 python3-pip
pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/


#--find-links https://download.pytorch.org/whl/torch_stable.html
--find-links https://mirrors.aliyun.com/pytorch-wheels/cu113

torch==1.11.0+cu113
torchdata==0.3
torchtext==0.12
spacy==3.2.6
pandas==1.3.5
altair==4.1
jupytext==1.13
flake8
black
GPUtil
wandb


pip3 install 'numpy<2.0'

# download and install tokenizer
# https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.2.0/de_core_news_sm-3.2.0-py3-none-any.whl
#python3 -m spacy download de_core_news_sm
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl
#python3 -m spacy download en_core_web_sm
pip3 install de_core_news_sm-3.2.0-py3-none-any.whl en_core_web_sm-3.2.0-py3-none-any.whl
# github: https://github.com/multi30k/dataset
# download dataset: https://gitcode.com/open-source-toolkit/66124
# and etract it to 
tar  -C /root/ -xvf multi30k.tar.gz

--find-links https://mirrors.aliyun.com/pytorch-wheels/cu121

torch==2.2.0+cu121
torchtext
spacy==3.2.6
pandas
altair
jupyter
flake8
black
GPUtil
wandb

pip3 install torch==2.2.0+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121

pip3 install torchdata==0.8 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
pip3 install torchtext==0.17 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
```