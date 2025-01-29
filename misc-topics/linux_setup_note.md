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
    - [HuggingFace Mirrors](../../hello-world/my_wiki/machine_learning/huggingface_model_store.md)
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
- netron