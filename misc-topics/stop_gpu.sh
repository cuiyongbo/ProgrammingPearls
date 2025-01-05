#!/usr/bin/env bash

# stop processes which may use GPU
sudo systemctl stop gdm

# how to find GPU BUS ID
# use nvidia-smi
#nvidia-smi
#+-----------------------------------------------------------------------------------------+
#| NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
#|-----------------------------------------+------------------------+----------------------+
#| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
#| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
#|                                         |                        |               MIG M. |
#|=========================================+========================+======================|
#|   0  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:06:00.0 Off |                  N/A |
#| 16%   37C    P0             63W /  250W |       1MiB /  22528MiB |      0%      Default |
#|                                         |                        |                  N/A |
#+-----------------------------------------+------------------------+----------------------+
#
# or use lspci
#lspci | grep NVIDIA
#06:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] (rev a1)
#06:00.1 Audio device: NVIDIA Corporation TU102 High Definition Audio Controller (rev a1)
#06:00.2 USB controller: NVIDIA Corporation TU102 USB 3.1 Host Controller (rev a1)
#06:00.3 Serial bus controller: NVIDIA Corporation TU102 USB Type-C UCSI Controller (rev a1)

# turn off persistence mode
sudo nvidia-smi -i 00000000:06:00.0 -pm 0

# (optional) hide GPU from nvidia-smi
sudo nvidia-smi drain -p 0000:06:00.0 -m 1

