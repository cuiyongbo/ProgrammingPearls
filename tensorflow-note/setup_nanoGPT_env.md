# Setup nanoGPT

* [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
    * [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
    * [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)
* how to develop with a jupyter notebook server resident in a docker container
    * [Running the Notebook](https://docs.jupyter.org/en/latest/running.html#starting-the-notebook-server)

```bash
# download ngc image for pytorch: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# start docker container, keep it running in background
docker run --gpus all -t -d --rm --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/data-store/huggingface-store:/model-store nvcr.io/nvidia/pytorch:24.01-py3
# -t, --tty                              Allocate a pseudo-TTY
# -d, --detach                           Run container in background and print container ID
# --rm means Automatically remove the container when it exits
# --ipc string                       IPC mode to use #https://docs.docker.com/reference/cli/docker/container/run/#ipc
#   --ipc=host means sharing the host's IPC namespace, for example, shared memory
# --network=host                  start the container using the host's network, also mapping all used ports to host, you may not use this option on product env, or it will suffer from security issues

# docker container list
CONTAINER ID   IMAGE                                     COMMAND                  CREATED        STATUS        PORTS                                                           NAMES
cb64ff444240   nvcr.io/nvidia/pytorch:24.01-py3          "/opt/nvidia/nvidia_…"   2 hours ago    Up 2 hours                                                                    vibrant_jemison

# go inside the container
docker exec -u root -it cb64ff444240 bash

# chang the location of HuggingFace dataset cache dir, default to ``~/.cache/huggingface``
# refer to ``/home/cherry/.local/lib/python3.10/site-packages/datasets/config.py`` for the configurations (python version may be different from example)
# but I suggest you to link to default cache directory to some directory resident on file system with enough space, say 200GB disk space at least if you try to reproduce GPT2 with openwebtext dataset
cd && rm -rf .cache # remember to backup data
ln -s  /model-store/data-store/huggingface-cache-dir .cache

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

# create a notebook, adn select the kernel in the docker container
```
