#!/usr/bin/env bash

sudo nvidia-smi drain -p 0000:06:00.0 -m 0
sudo nvidia-smi -i 00000000:06:00.0 -pm 1
sudo systemctl start gdm
