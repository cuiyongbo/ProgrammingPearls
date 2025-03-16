# Transformer note

## References

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    - [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)

## Env Setup

- os
  
```bash
# lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.1 LTS
Release:        24.04
Codename:       noble

# nvidia-smi 
Thu Jan 23 00:28:04 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:06:00.0  On |                  N/A |
| 22%   25C    P8             32W /  250W |     544MiB /  22528MiB |      9%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

- [install pyenv on ubuntu 24.04](https://idroot.us/install-pyenv-ubuntu-24-04/)

- create a python3.10 virtualenv

```bash
# pyenv install 3.10.12
Downloading Python-3.10.12.tar.xz...
-> https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz

Installing Python-3.10.12...
Installed Python-3.10.12 to /home/cherry/.pyenv/versions/3.10.12

# pyenv versions
* system (set by /home/cherry/.pyenv/version)
  3.10.12
  3.10.12/envs/sd-webui-env
  sd-webui-env --> /home/cherry/.pyenv/versions/3.10.12/envs/sd-webui-env

# pyenv virtualenv 3.10.12 transformer-env

# pyenv activate transformer-env

# set index-url to aliyun to accelerate installation
# pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

- install packages in venv

```bash
# IMPORTANT: make sure there is no compatibility problem among torch packages: https://www.cnblogs.com/phillee/p/18599125
# try cu118 if you have older cuda driver installed
# pip3 install torch==2.1.0+cu118 torchtext==0.16 -f https://mirrors.aliyun.com/pytorch-wheels/cu118
pip3 install torch==2.1.0+cu121 torchtext==0.16 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
pip3 install spacy pandas altair jupyter jupytext flake8 black GPUtil wandb 
pip3 install 'numpy>=1.22.4,<2.0'
pip3 install 'portalocker>=2.0.0'
```

- download and install spacy tokenizer

```
# tokenizer: https://github.com/explosion/spacy-models
# https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl
python3 -m spacy download de_core_news_sm
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python3 -m spacy download en_core_web_sm
# if you have download whl packages to local host, you may install them manually with
# pip3 install *.whl
```

- download dataset

```
# [UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 37: invalid start byte](https://github.com/pytorch/text/issues/2221#issuecomment-1958604103)
# you may unzip dataset again to solve this error
# github: https://github.com/multi30k/dataset
# github mirrors: https://gitcode.com/open-source-toolkit/66124

# move downloaded dataset file to `/root/.cache/torch/text/datasets/Multi30k` and unzip it.
# the hierarchy may look like
root@di-20241115115906-kfh5w:~/.cache/torch/text/datasets/Multi30k# ls -l
total 5468
-rw-r--r-- 1 root root    67079 Jan 21 04:51 mmt16_task1_test.tar.gz
-rw-rw-r-- 1  501 staff   70649 Oct 17  2016 test.de
-rw-rw-r-- 1  501 staff   62076 Oct 17  2016 test.en
-rw-rw-r-- 1  501 staff   72261 Feb 11  2017 test.fr
-rw-r--r-- 1 root root  2110399 Jan 21 04:51 train.de
-rw-r--r-- 1 root root  1801239 Jan 21 04:51 train.en
-rw-r--r-- 1 root root  1207136 Jan 21 04:51 training.tar.gz
-rw-r--r-- 1 root root    75920 Jan 21 04:51 val.de
-rw-r--r-- 1 root root    63298 Jan 21 04:51 val.en
-rw-r--r-- 1 root root    46329 Jan 21 04:51 validation.tar.gz
```

## training log

- with nvidia L4 GPU

```bash
root@di-20241115115906-kfh5w:~/.cache/torch/text/datasets/Multi30k# nvidia-smi 
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA L4                      On  | 00000000:71:02.0 Off |                    0 |
| N/A   76C    P0              70W /  72W |   4172MiB / 23034MiB |     97%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+

(myenv) root@di-20241115115906-kfh5w:~/code/annotated-transformer# python3 the_annotated_transformer.py 
Building German Vocabulary ...
Building English Vocabulary ...
Finished.
Vocabulary sizes:
8316
6384
Train worker process using GPU: 0 for training
[GPU0] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   7.60 | Tokens / Sec:  1433.8 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.38 | Tokens / Sec:  3816.2 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   6.97 | Tokens / Sec:  3967.4 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   6.67 | Tokens / Sec:  3877.4 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   6.50 | Tokens / Sec:  3942.2 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   6.30 | Tokens / Sec:  3963.4 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.09 | Tokens / Sec:  3935.4 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   5.94 | Tokens / Sec:  3923.6 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   5.74 | Tokens / Sec:  3948.7 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   5.63 | Tokens / Sec:  3946.0 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   5.29 | Tokens / Sec:  3854.9 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.16 | Tokens / Sec:  3943.7 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.04 | Tokens / Sec:  3941.3 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   4.70 | Tokens / Sec:  3917.8 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   4.61 | Tokens / Sec:  3892.5 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.52 | Tokens / Sec:  3957.6 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.46 | Tokens / Sec:  3971.9 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.30 | Tokens / Sec:  3863.9 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.33 | Tokens / Sec:  3918.3 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.06 | Tokens / Sec:  3846.9 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.18 | Tokens / Sec:  3892.0 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.14 | Tokens / Sec:  3928.8 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.30 | Tokens / Sec:  3898.6 | Learning Rate: 2.4e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 15% |
[GPU0] Epoch 0 Validation ====
(tensor(3.8770, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.86 | Tokens / Sec:  4457.3 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.78 | Tokens / Sec:  3817.2 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.65 | Tokens / Sec:  3859.7 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.58 | Tokens / Sec:  3868.6 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.76 | Tokens / Sec:  3892.3 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.71 | Tokens / Sec:  3767.5 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.68 | Tokens / Sec:  3873.0 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.66 | Tokens / Sec:  3804.7 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.48 | Tokens / Sec:  3906.5 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.41 | Tokens / Sec:  3808.4 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.65 | Tokens / Sec:  3813.7 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.44 | Tokens / Sec:  3838.5 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.26 | Tokens / Sec:  3863.4 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.26 | Tokens / Sec:  3762.4 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.20 | Tokens / Sec:  3792.1 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.19 | Tokens / Sec:  3782.8 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.97 | Tokens / Sec:  3793.3 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.19 | Tokens / Sec:  3851.9 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.17 | Tokens / Sec:  3794.9 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.91 | Tokens / Sec:  3768.6 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.05 | Tokens / Sec:  3778.8 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.23 | Tokens / Sec:  3762.4 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.02 | Tokens / Sec:  3777.6 | Learning Rate: 4.8e-04
| ID | GPU | MEM |
------------------
|  0 | 96% | 18% |
[GPU0] Epoch 1 Validation ====
(tensor(2.8317, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.86 | Tokens / Sec:  3649.8 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.73 | Tokens / Sec:  3798.3 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.78 | Tokens / Sec:  3769.7 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.81 | Tokens / Sec:  3730.6 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.66 | Tokens / Sec:  3756.4 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.63 | Tokens / Sec:  3744.2 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.61 | Tokens / Sec:  3780.2 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.59 | Tokens / Sec:  3707.2 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.60 | Tokens / Sec:  3734.7 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.59 | Tokens / Sec:  3804.9 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.48 | Tokens / Sec:  3741.2 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.73 | Tokens / Sec:  3763.9 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.74 | Tokens / Sec:  3777.3 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.62 | Tokens / Sec:  3630.8 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.36 | Tokens / Sec:  3825.7 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.45 | Tokens / Sec:  3788.2 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.57 | Tokens / Sec:  3766.1 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.55 | Tokens / Sec:  3860.5 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.42 | Tokens / Sec:  3752.4 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.21 | Tokens / Sec:  3747.5 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.66 | Tokens / Sec:  3794.8 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.25 | Tokens / Sec:  3798.2 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.13 | Tokens / Sec:  3777.8 | Learning Rate: 7.3e-04
| ID | GPU | MEM |
------------------
|  0 | 97% | 18% |
[GPU0] Epoch 2 Validation ====
(tensor(2.1319, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.18 | Tokens / Sec:  4234.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.97 | Tokens / Sec:  3808.3 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.10 | Tokens / Sec:  3735.1 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.70 | Tokens / Sec:  3780.3 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.94 | Tokens / Sec:  3621.2 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.82 | Tokens / Sec:  3748.8 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.01 | Tokens / Sec:  3833.0 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.29 | Tokens / Sec:  3757.5 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.15 | Tokens / Sec:  3802.3 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.99 | Tokens / Sec:  3799.1 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.90 | Tokens / Sec:  3770.8 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.82 | Tokens / Sec:  3762.7 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.05 | Tokens / Sec:  3765.0 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.00 | Tokens / Sec:  3770.9 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.99 | Tokens / Sec:  3784.0 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.89 | Tokens / Sec:  3775.2 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.75 | Tokens / Sec:  3774.9 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.81 | Tokens / Sec:  3757.2 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.73 | Tokens / Sec:  3796.3 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.94 | Tokens / Sec:  3716.2 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.03 | Tokens / Sec:  3776.3 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.88 | Tokens / Sec:  3728.5 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.08 | Tokens / Sec:  3794.3 | Learning Rate: 7.4e-04
| ID | GPU | MEM |
------------------
|  0 | 96% | 18% |
[GPU0] Epoch 3 Validation ====
(tensor(1.7542, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.61 | Tokens / Sec:  4153.3 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.57 | Tokens / Sec:  3832.6 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.52 | Tokens / Sec:  3825.8 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.32 | Tokens / Sec:  3817.4 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.48 | Tokens / Sec:  3745.7 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.75 | Tokens / Sec:  3721.2 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.60 | Tokens / Sec:  3824.4 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.61 | Tokens / Sec:  3714.1 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.65 | Tokens / Sec:  3831.6 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.38 | Tokens / Sec:  3695.7 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.55 | Tokens / Sec:  3733.3 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.34 | Tokens / Sec:  3796.5 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.67 | Tokens / Sec:  3730.3 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.47 | Tokens / Sec:  3700.2 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.67 | Tokens / Sec:  3800.9 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.56 | Tokens / Sec:  3807.7 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.58 | Tokens / Sec:  3691.8 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.70 | Tokens / Sec:  3771.3 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.52 | Tokens / Sec:  3718.8 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.68 | Tokens / Sec:  3774.8 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.49 | Tokens / Sec:  3737.1 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.64 | Tokens / Sec:  3779.8 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.37 | Tokens / Sec:  3682.3 | Learning Rate: 6.6e-04
| ID | GPU | MEM |
------------------
|  0 | 97% | 18% |
[GPU0] Epoch 4 Validation ====
(tensor(1.5888, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.32 | Tokens / Sec:  4113.5 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.41 | Tokens / Sec:  3725.2 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.11 | Tokens / Sec:  3787.0 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.37 | Tokens / Sec:  3813.5 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.36 | Tokens / Sec:  3803.8 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.19 | Tokens / Sec:  3767.9 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.25 | Tokens / Sec:  3762.5 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.34 | Tokens / Sec:  3764.2 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.22 | Tokens / Sec:  3821.4 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.25 | Tokens / Sec:  3738.2 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.42 | Tokens / Sec:  3764.5 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.28 | Tokens / Sec:  3730.1 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.39 | Tokens / Sec:  3748.1 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.43 | Tokens / Sec:  3756.2 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.46 | Tokens / Sec:  3821.1 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.32 | Tokens / Sec:  3854.7 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.27 | Tokens / Sec:  3685.5 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.31 | Tokens / Sec:  3745.2 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.27 | Tokens / Sec:  3715.4 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.28 | Tokens / Sec:  3815.9 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.36 | Tokens / Sec:  3708.7 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.21 | Tokens / Sec:  3719.0 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.25 | Tokens / Sec:  3853.4 | Learning Rate: 6.0e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 18% |
[GPU0] Epoch 5 Validation ====
(tensor(1.5020, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.12 | Tokens / Sec:  3909.6 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.20 | Tokens / Sec:  3837.4 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.19 | Tokens / Sec:  3752.9 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.32 | Tokens / Sec:  3773.1 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.13 | Tokens / Sec:  3697.7 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.14 | Tokens / Sec:  3761.2 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.03 | Tokens / Sec:  3810.8 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.03 | Tokens / Sec:  3732.0 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.98 | Tokens / Sec:  3787.7 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.01 | Tokens / Sec:  3821.6 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.10 | Tokens / Sec:  3809.0 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.23 | Tokens / Sec:  3787.4 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.07 | Tokens / Sec:  3774.2 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.19 | Tokens / Sec:  3789.9 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.06 | Tokens / Sec:  3776.6 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.21 | Tokens / Sec:  3776.8 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.27 | Tokens / Sec:  3758.6 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.18 | Tokens / Sec:  3812.2 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.14 | Tokens / Sec:  3774.6 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.01 | Tokens / Sec:  3757.2 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.23 | Tokens / Sec:  3755.9 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.26 | Tokens / Sec:  3699.5 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.28 | Tokens / Sec:  3804.9 | Learning Rate: 5.6e-04
| ID | GPU | MEM |
------------------
|  0 | 97% | 18% |
[GPU0] Epoch 6 Validation ====
(tensor(1.4607, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
[GPU0] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.08 | Tokens / Sec:  4086.4 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.09 | Tokens / Sec:  3753.1 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.15 | Tokens / Sec:  3805.9 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.00 | Tokens / Sec:  3817.1 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.89 | Tokens / Sec:  3764.8 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.02 | Tokens / Sec:  3713.0 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.03 | Tokens / Sec:  3821.1 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.13 | Tokens / Sec:  3778.9 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.02 | Tokens / Sec:  3711.1 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.89 | Tokens / Sec:  3788.7 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.97 | Tokens / Sec:  3738.7 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.99 | Tokens / Sec:  3800.7 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.98 | Tokens / Sec:  3809.7 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.97 | Tokens / Sec:  3798.2 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.10 | Tokens / Sec:  3720.5 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.07 | Tokens / Sec:  3811.6 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.03 | Tokens / Sec:  3753.1 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.08 | Tokens / Sec:  3732.9 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.05 | Tokens / Sec:  3811.8 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.98 | Tokens / Sec:  3839.2 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.17 | Tokens / Sec:  3786.4 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.18 | Tokens / Sec:  3748.4 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.95 | Tokens / Sec:  3752.9 | Learning Rate: 5.2e-04
| ID | GPU | MEM |
------------------
|  0 | 97% | 18% |
[GPU0] Epoch 7 Validation ====
(tensor(1.4455, device='cuda:0'), <__main__.TrainState object at 0x7f1fa035ea70>)
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein Mann führt zwei kleine <unk> in einem Park spazieren . </s>
Target Text (Ground Truth) : <s> A man is leading two small <unk> on a walk at a park . </s>
Model Output               : <s> A man is walking two small walk in a park . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein Schwimmer schwimmt in einem Swimmingpool . </s>
Target Text (Ground Truth) : <s> An swimmer swimming in a swimming pool . </s>
Model Output               : <s> A swimmer is swimming in a swimming pool . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Eine Frau steht auf einem Bein auf einer hohen Klippe und blickt über einen Fluss . </s>
Target Text (Ground Truth) : <s> A woman standing on a high cliff on one leg looking over a river . </s>
Model Output               : <s> A woman stands on a high cliff over a river . </s>
```

- with nvidia RTX 2080Ti

```bash
# nvidia-smi 
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off |   00000000:06:00.0  On |                  N/A |
| 34%   59C    P0            241W /  250W |    4687MiB /  22528MiB |     94%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     43279      C   ...ersions/transformer-env/bin/python3       4124MiB |
+-----------------------------------------------------------------------------------------+

# python3 the_annotated_transformer.py 
Building German Vocabulary ...
Building English Vocabulary ...
Finished.
Vocabulary sizes:
8316
6384
Train worker process using GPU: 0 for training
[GPU0] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   7.61 | Tokens / Sec:  2806.2 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.40 | Tokens / Sec:  4377.0 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   7.00 | Tokens / Sec:  4294.1 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   6.64 | Tokens / Sec:  4163.0 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   6.45 | Tokens / Sec:  4307.9 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   6.33 | Tokens / Sec:  4269.2 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.21 | Tokens / Sec:  4336.6 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   5.99 | Tokens / Sec:  4280.0 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   5.77 | Tokens / Sec:  4254.5 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   5.49 | Tokens / Sec:  4241.2 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   5.45 | Tokens / Sec:  4287.5 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.02 | Tokens / Sec:  4289.0 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   4.88 | Tokens / Sec:  4170.2 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   4.78 | Tokens / Sec:  4269.6 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   4.50 | Tokens / Sec:  4299.7 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.46 | Tokens / Sec:  4268.7 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.48 | Tokens / Sec:  4239.3 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.25 | Tokens / Sec:  4276.9 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.15 | Tokens / Sec:  4202.7 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.04 | Tokens / Sec:  3898.4 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.12 | Tokens / Sec:  4137.1 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.02 | Tokens / Sec:  4137.4 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.07 | Tokens / Sec:  4262.3 | Learning Rate: 2.4e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 17% |
[GPU0] Epoch 0 Validation ====
(tensor(3.8710, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.97 | Tokens / Sec:  4361.1 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   4.16 | Tokens / Sec:  4315.2 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.91 | Tokens / Sec:  4244.1 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.82 | Tokens / Sec:  4150.1 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.53 | Tokens / Sec:  3978.9 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.56 | Tokens / Sec:  4117.5 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.78 | Tokens / Sec:  4055.9 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.54 | Tokens / Sec:  3976.7 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.26 | Tokens / Sec:  4306.3 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.40 | Tokens / Sec:  4121.6 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.28 | Tokens / Sec:  4167.2 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.19 | Tokens / Sec:  4212.1 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.98 | Tokens / Sec:  4121.8 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.28 | Tokens / Sec:  3952.5 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.07 | Tokens / Sec:  4176.2 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.44 | Tokens / Sec:  4108.5 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.13 | Tokens / Sec:  3920.2 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.30 | Tokens / Sec:  4070.1 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.21 | Tokens / Sec:  4212.7 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.94 | Tokens / Sec:  4201.3 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.08 | Tokens / Sec:  4246.3 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.85 | Tokens / Sec:  4197.6 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.80 | Tokens / Sec:  4140.8 | Learning Rate: 4.8e-04
| ID | GPU | MEM |
------------------
|  0 | 95% | 21% |
[GPU0] Epoch 1 Validation ====
(tensor(2.8402, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.83 | Tokens / Sec:  4495.1 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.66 | Tokens / Sec:  4254.6 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.76 | Tokens / Sec:  4216.2 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.77 | Tokens / Sec:  4202.2 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.88 | Tokens / Sec:  3870.5 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.84 | Tokens / Sec:  4171.0 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.62 | Tokens / Sec:  4178.7 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.52 | Tokens / Sec:  4199.2 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.45 | Tokens / Sec:  4226.4 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.67 | Tokens / Sec:  4155.2 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.49 | Tokens / Sec:  3839.2 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.47 | Tokens / Sec:  4016.3 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.36 | Tokens / Sec:  4222.9 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.32 | Tokens / Sec:  4129.7 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.72 | Tokens / Sec:  4238.5 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.34 | Tokens / Sec:  4289.2 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.29 | Tokens / Sec:  4269.1 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.63 | Tokens / Sec:  4090.5 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.33 | Tokens / Sec:  4272.2 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.20 | Tokens / Sec:  4219.5 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.43 | Tokens / Sec:  4271.9 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.95 | Tokens / Sec:  4251.3 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.10 | Tokens / Sec:  4242.9 | Learning Rate: 7.3e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 21% |
[GPU0] Epoch 2 Validation ====
(tensor(2.1210, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.09 | Tokens / Sec:  4196.9 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.82 | Tokens / Sec:  4207.1 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.89 | Tokens / Sec:  4224.5 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.07 | Tokens / Sec:  3768.1 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.90 | Tokens / Sec:  3851.5 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.97 | Tokens / Sec:  4229.4 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.82 | Tokens / Sec:  4191.3 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.03 | Tokens / Sec:  4187.0 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.08 | Tokens / Sec:  3993.9 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.96 | Tokens / Sec:  3953.3 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.82 | Tokens / Sec:  3971.3 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.84 | Tokens / Sec:  4089.6 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.53 | Tokens / Sec:  4189.1 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.86 | Tokens / Sec:  4231.4 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.05 | Tokens / Sec:  4213.3 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.90 | Tokens / Sec:  4263.5 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.84 | Tokens / Sec:  4235.9 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.21 | Tokens / Sec:  4199.7 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.77 | Tokens / Sec:  4152.5 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.95 | Tokens / Sec:  4250.9 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.81 | Tokens / Sec:  4248.8 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.76 | Tokens / Sec:  4184.4 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.66 | Tokens / Sec:  4167.9 | Learning Rate: 7.4e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 21% |
[GPU0] Epoch 3 Validation ====
(tensor(1.7545, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.31 | Tokens / Sec:  4798.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.89 | Tokens / Sec:  4256.4 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.41 | Tokens / Sec:  4240.6 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.34 | Tokens / Sec:  3968.3 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.34 | Tokens / Sec:  4173.2 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.39 | Tokens / Sec:  4269.7 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.25 | Tokens / Sec:  4124.2 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.83 | Tokens / Sec:  4205.7 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.40 | Tokens / Sec:  4142.2 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.28 | Tokens / Sec:  4140.5 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.47 | Tokens / Sec:  4162.5 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.42 | Tokens / Sec:  4183.7 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.65 | Tokens / Sec:  4265.0 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.53 | Tokens / Sec:  4205.0 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.01 | Tokens / Sec:  4255.7 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.68 | Tokens / Sec:  4284.1 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.52 | Tokens / Sec:  4146.9 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.59 | Tokens / Sec:  4249.6 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.41 | Tokens / Sec:  4264.9 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.52 | Tokens / Sec:  4186.0 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.48 | Tokens / Sec:  4218.6 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.58 | Tokens / Sec:  4278.8 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.49 | Tokens / Sec:  4189.1 | Learning Rate: 6.6e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 21% |
[GPU0] Epoch 4 Validation ====
(tensor(1.5927, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.29 | Tokens / Sec:  5005.3 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.22 | Tokens / Sec:  4225.6 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.29 | Tokens / Sec:  4221.1 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.08 | Tokens / Sec:  4226.6 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.17 | Tokens / Sec:  4241.7 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.31 | Tokens / Sec:  4187.4 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.36 | Tokens / Sec:  4216.4 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.30 | Tokens / Sec:  4259.0 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.49 | Tokens / Sec:  4109.9 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.38 | Tokens / Sec:  4169.8 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.16 | Tokens / Sec:  4177.0 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.19 | Tokens / Sec:  4263.1 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.01 | Tokens / Sec:  4197.3 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.30 | Tokens / Sec:  4268.3 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.32 | Tokens / Sec:  4147.7 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.64 | Tokens / Sec:  4153.8 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.47 | Tokens / Sec:  4136.1 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.26 | Tokens / Sec:  3913.1 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.06 | Tokens / Sec:  4221.4 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.38 | Tokens / Sec:  4209.6 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.26 | Tokens / Sec:  4219.2 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.29 | Tokens / Sec:  4192.2 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.31 | Tokens / Sec:  4266.1 | Learning Rate: 6.0e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 21% |
[GPU0] Epoch 5 Validation ====
(tensor(1.5168, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.13 | Tokens / Sec:  5061.3 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.29 | Tokens / Sec:  4295.7 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.26 | Tokens / Sec:  4081.8 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.07 | Tokens / Sec:  4272.5 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.25 | Tokens / Sec:  4167.5 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.25 | Tokens / Sec:  4132.6 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.11 | Tokens / Sec:  3992.5 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.11 | Tokens / Sec:  4067.6 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.26 | Tokens / Sec:  3977.0 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.23 | Tokens / Sec:  4152.5 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.24 | Tokens / Sec:  4089.6 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.95 | Tokens / Sec:  3978.0 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.33 | Tokens / Sec:  4077.8 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.25 | Tokens / Sec:  3911.4 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.24 | Tokens / Sec:  4015.8 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.22 | Tokens / Sec:  4085.2 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.22 | Tokens / Sec:  3929.9 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.02 | Tokens / Sec:  4046.3 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.06 | Tokens / Sec:  4052.6 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.16 | Tokens / Sec:  4058.4 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.23 | Tokens / Sec:  4180.2 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.15 | Tokens / Sec:  4124.5 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.26 | Tokens / Sec:  4078.3 | Learning Rate: 5.6e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 21% |
[GPU0] Epoch 6 Validation ====
(tensor(1.4683, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
[GPU0] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.04 | Tokens / Sec:  4586.5 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.90 | Tokens / Sec:  4157.5 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.07 | Tokens / Sec:  3997.9 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.89 | Tokens / Sec:  3928.3 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.99 | Tokens / Sec:  4085.8 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.06 | Tokens / Sec:  4042.3 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.05 | Tokens / Sec:  4108.5 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.19 | Tokens / Sec:  4137.8 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.15 | Tokens / Sec:  4080.8 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.94 | Tokens / Sec:  4146.4 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.00 | Tokens / Sec:  4134.2 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.94 | Tokens / Sec:  4002.8 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.88 | Tokens / Sec:  3945.3 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.15 | Tokens / Sec:  4068.0 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.17 | Tokens / Sec:  4128.7 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.01 | Tokens / Sec:  4039.3 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.17 | Tokens / Sec:  4182.5 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.11 | Tokens / Sec:  4085.3 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.94 | Tokens / Sec:  4115.3 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.11 | Tokens / Sec:  4021.9 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.00 | Tokens / Sec:  4088.6 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.32 | Tokens / Sec:  4127.1 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.10 | Tokens / Sec:  4130.2 | Learning Rate: 5.2e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 20% |
[GPU0] Epoch 7 Validation ====
(tensor(1.4561, device='cuda:0'), <__main__.TrainState object at 0x78bfd53d9330>)
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein kleines Mädchen streckt die Hand aus , um ein Reh zu streicheln . </s>
Target Text (Ground Truth) : <s> Young girl reaches out to pet a deer . </s>
Model Output               : <s> A little girl is sticking her hand out to a deer . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Drei Frauen sitzen da und lächeln . </s>
Target Text (Ground Truth) : <s> Three women smiling and sitting down . </s>
Model Output               : <s> Three women are sitting and smiling . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Der Mann telefoniert vor einem Sportgeschäft mit seinem Handy . </s>
Target Text (Ground Truth) : <s> The man is talking on his cellphone in front of a sports store . </s>
Model Output               : <s> The man is talking on the cellphone outside of a shop with his cellphone . </s>
```

- with nivdia T4

```bash
# nvidia-smi 
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GRID T4-8C          On   | 00000000:00:02.0 Off |                    0 |
| N/A   N/A    P8    N/A /  N/A |      0MiB /  8192MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
Building German Vocabulary ...
Building English Vocabulary ...
Finished.
Vocabulary sizes:
8316
6384
Train worker process using GPU: 0 for training
[GPU0] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   7.66 | Tokens / Sec:   887.9 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.43 | Tokens / Sec:  1922.0 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   6.95 | Tokens / Sec:  1901.2 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   6.64 | Tokens / Sec:  1914.5 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   6.46 | Tokens / Sec:  1872.9 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   6.36 | Tokens / Sec:  1897.9 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.18 | Tokens / Sec:  1882.3 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   5.99 | Tokens / Sec:  1854.1 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   5.82 | Tokens / Sec:  1861.1 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   5.53 | Tokens / Sec:  1868.9 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   5.32 | Tokens / Sec:  1885.2 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.05 | Tokens / Sec:  1857.0 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.02 | Tokens / Sec:  1860.0 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   4.60 | Tokens / Sec:  1833.3 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   4.55 | Tokens / Sec:  1823.7 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.47 | Tokens / Sec:  1827.1 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.25 | Tokens / Sec:  1854.5 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.29 | Tokens / Sec:  1848.0 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.32 | Tokens / Sec:  1838.7 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.12 | Tokens / Sec:  1863.1 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.13 | Tokens / Sec:  1865.4 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.11 | Tokens / Sec:  1866.5 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.85 | Tokens / Sec:  1836.0 | Learning Rate: 2.4e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 48% |
[GPU0] Epoch 0 Validation ====
(tensor(3.8650, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.97 | Tokens / Sec:  2055.2 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.88 | Tokens / Sec:  1830.4 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.83 | Tokens / Sec:  1818.5 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.88 | Tokens / Sec:  1837.1 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.43 | Tokens / Sec:  1814.6 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.64 | Tokens / Sec:  1840.8 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.70 | Tokens / Sec:  1837.4 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.34 | Tokens / Sec:  1842.7 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.50 | Tokens / Sec:  1859.7 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.34 | Tokens / Sec:  1860.6 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.42 | Tokens / Sec:  1805.3 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.26 | Tokens / Sec:  1853.7 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.63 | Tokens / Sec:  1854.9 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.18 | Tokens / Sec:  1856.3 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.09 | Tokens / Sec:  1823.7 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.19 | Tokens / Sec:  1852.8 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.12 | Tokens / Sec:  1853.1 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.27 | Tokens / Sec:  1859.7 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.02 | Tokens / Sec:  1863.9 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.28 | Tokens / Sec:  1834.1 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.09 | Tokens / Sec:  1826.4 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.14 | Tokens / Sec:  1837.2 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.82 | Tokens / Sec:  1841.8 | Learning Rate: 4.8e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 58% |
[GPU0] Epoch 1 Validation ====
(tensor(2.8495, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.91 | Tokens / Sec:  2086.5 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.86 | Tokens / Sec:  1864.0 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.07 | Tokens / Sec:  1841.8 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.72 | Tokens / Sec:  1847.6 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.91 | Tokens / Sec:  1860.2 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.55 | Tokens / Sec:  1859.4 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.01 | Tokens / Sec:  1830.5 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.86 | Tokens / Sec:  1837.4 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.60 | Tokens / Sec:  1826.6 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.33 | Tokens / Sec:  1852.9 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.65 | Tokens / Sec:  1862.8 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.35 | Tokens / Sec:  1839.5 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.26 | Tokens / Sec:  1832.1 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.36 | Tokens / Sec:  1847.3 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.19 | Tokens / Sec:  1833.0 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.49 | Tokens / Sec:  1846.4 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.79 | Tokens / Sec:  1863.3 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.30 | Tokens / Sec:  1818.9 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.49 | Tokens / Sec:  1841.9 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.31 | Tokens / Sec:  1849.6 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.29 | Tokens / Sec:  1791.7 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.50 | Tokens / Sec:  1837.9 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.41 | Tokens / Sec:  1830.2 | Learning Rate: 7.3e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 58% |
[GPU0] Epoch 2 Validation ====
(tensor(2.1520, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.19 | Tokens / Sec:  2158.4 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.01 | Tokens / Sec:  1852.7 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.75 | Tokens / Sec:  1847.5 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.00 | Tokens / Sec:  1840.3 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.01 | Tokens / Sec:  1837.6 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.27 | Tokens / Sec:  1841.9 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.99 | Tokens / Sec:  1868.9 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.14 | Tokens / Sec:  1856.7 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.12 | Tokens / Sec:  1840.9 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.04 | Tokens / Sec:  1841.0 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.82 | Tokens / Sec:  1850.5 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.06 | Tokens / Sec:  1838.5 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.81 | Tokens / Sec:  1822.0 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.82 | Tokens / Sec:  1839.8 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.06 | Tokens / Sec:  1875.6 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.60 | Tokens / Sec:  1842.2 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.13 | Tokens / Sec:  1845.9 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.87 | Tokens / Sec:  1853.1 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.99 | Tokens / Sec:  1839.5 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.23 | Tokens / Sec:  1845.8 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.97 | Tokens / Sec:  1833.5 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.81 | Tokens / Sec:  1835.8 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.01 | Tokens / Sec:  1824.3 | Learning Rate: 7.4e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 58% |
[GPU0] Epoch 3 Validation ====
(tensor(1.7605, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.47 | Tokens / Sec:  2067.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.66 | Tokens / Sec:  1816.9 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.60 | Tokens / Sec:  1824.0 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.53 | Tokens / Sec:  1843.0 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.54 | Tokens / Sec:  1852.2 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.44 | Tokens / Sec:  1847.6 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.49 | Tokens / Sec:  1838.5 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.49 | Tokens / Sec:  1849.4 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.33 | Tokens / Sec:  1828.9 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.87 | Tokens / Sec:  1886.4 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.61 | Tokens / Sec:  1856.0 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.79 | Tokens / Sec:  1855.9 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.41 | Tokens / Sec:  1838.6 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.60 | Tokens / Sec:  1842.9 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.42 | Tokens / Sec:  1860.8 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.46 | Tokens / Sec:  1850.6 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.62 | Tokens / Sec:  1837.4 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.75 | Tokens / Sec:  1855.4 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.44 | Tokens / Sec:  1878.7 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.18 | Tokens / Sec:  1843.5 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.39 | Tokens / Sec:  1835.9 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.41 | Tokens / Sec:  1826.5 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.56 | Tokens / Sec:  1822.7 | Learning Rate: 6.6e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 58% |
[GPU0] Epoch 4 Validation ====
(tensor(1.5791, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.47 | Tokens / Sec:  2018.4 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.25 | Tokens / Sec:  1854.8 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.45 | Tokens / Sec:  1831.8 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.35 | Tokens / Sec:  1841.1 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.08 | Tokens / Sec:  1812.7 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.16 | Tokens / Sec:  1854.0 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.18 | Tokens / Sec:  1831.6 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.18 | Tokens / Sec:  1834.6 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.32 | Tokens / Sec:  1851.5 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.56 | Tokens / Sec:  1844.9 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.36 | Tokens / Sec:  1864.7 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.54 | Tokens / Sec:  1869.9 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.32 | Tokens / Sec:  1853.9 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.20 | Tokens / Sec:  1859.0 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.21 | Tokens / Sec:  1839.8 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.48 | Tokens / Sec:  1833.3 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.52 | Tokens / Sec:  1863.0 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.18 | Tokens / Sec:  1839.1 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.47 | Tokens / Sec:  1837.4 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.42 | Tokens / Sec:  1839.6 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.45 | Tokens / Sec:  1832.8 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.36 | Tokens / Sec:  1841.3 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.36 | Tokens / Sec:  1843.7 | Learning Rate: 6.0e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 58% |
[GPU0] Epoch 5 Validation ====
(tensor(1.5049, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.15 | Tokens / Sec:  2126.7 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.22 | Tokens / Sec:  1841.6 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.04 | Tokens / Sec:  1822.1 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.14 | Tokens / Sec:  1837.8 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.10 | Tokens / Sec:  1847.0 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.82 | Tokens / Sec:  1843.9 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.12 | Tokens / Sec:  1847.1 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.06 | Tokens / Sec:  1833.3 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.14 | Tokens / Sec:  1849.8 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.10 | Tokens / Sec:  1862.8 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.06 | Tokens / Sec:  1844.1 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.08 | Tokens / Sec:  1834.7 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.19 | Tokens / Sec:  1818.7 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.14 | Tokens / Sec:  1847.9 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.12 | Tokens / Sec:  1858.9 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.40 | Tokens / Sec:  1847.1 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.15 | Tokens / Sec:  1856.1 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.09 | Tokens / Sec:  1834.5 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.07 | Tokens / Sec:  1832.2 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.17 | Tokens / Sec:  1834.6 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.08 | Tokens / Sec:  1850.9 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.34 | Tokens / Sec:  1875.2 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.23 | Tokens / Sec:  1868.3 | Learning Rate: 5.6e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 58% |
[GPU0] Epoch 6 Validation ====
(tensor(1.4714, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
[GPU0] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.10 | Tokens / Sec:  2190.0 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.06 | Tokens / Sec:  1860.9 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.05 | Tokens / Sec:  1859.7 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.17 | Tokens / Sec:  1830.3 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.14 | Tokens / Sec:  1826.1 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.04 | Tokens / Sec:  1843.4 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.85 | Tokens / Sec:  1832.4 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.88 | Tokens / Sec:  1851.6 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.13 | Tokens / Sec:  1849.5 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.96 | Tokens / Sec:  1837.2 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.92 | Tokens / Sec:  1845.4 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.18 | Tokens / Sec:  1851.2 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.02 | Tokens / Sec:  1854.6 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.16 | Tokens / Sec:  1849.9 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.97 | Tokens / Sec:  1822.0 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.12 | Tokens / Sec:  1845.7 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.26 | Tokens / Sec:  1875.2 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.21 | Tokens / Sec:  1818.2 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.06 | Tokens / Sec:  1850.2 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.11 | Tokens / Sec:  1862.6 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.99 | Tokens / Sec:  1844.5 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.95 | Tokens / Sec:  1849.2 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.21 | Tokens / Sec:  1858.4 | Learning Rate: 5.2e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 58% |
[GPU0] Epoch 7 Validation ====
(tensor(1.4475, device='cuda:0'), <__main__.TrainState object at 0x7f0435dc15d0>)
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Eine Frau steht vor Bäumen und lächelt . </s>
Target Text (Ground Truth) : <s> A woman standing in front of trees and smiling . </s>
Model Output               : <s> A woman standing in front of trees and smiling . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein alter Mann läuft mit einem Ordner in der Hand herum </s>
Target Text (Ground Truth) : <s> An old man walking with a folder in his hand </s>
Model Output               : <s> An old man is running with a each other . </s>
Preparing Data ...
Loading Trained Model ...
Checking Model Outputs:

Example 0 ========

Source Text (Input)        : <s> Ein Mädchen mit geschminktem Gesicht und einem orangen Pullover steht bei ihrer Truppe . </s>
Target Text (Ground Truth) : <s> A girl with face paint and an orange sweater stands with her party . </s>
Model Output               : <s> A girl with a painted face and an orange sweater stands by her band . </s>
```


- with MSI GeForce RTX 3090 Ventus 3X 24G OC


```bash
# nvidia-smi
Fri Mar 14 15:06:41 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:06:00.0  On |                  N/A |
|  0%   42C    P8             25W /  350W |     474MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
# python3 '/home/cherry/workspace/programming-pearls/machine-learning-note/transformer/harvardnlp_transformer.py'
Building German Vocabulary ...
Building English Vocabulary ...
load_vocab Finished.
Vocabulary sizes:
English Vocabulary size: 8316
Germany Vocabulary size: 6384
train_worker on GPU 0
[GPU 0] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   8.88 | Tokens / Sec:  2722.1 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   8.63 | Tokens / Sec:  6317.9 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   8.14 | Tokens / Sec:  6365.9 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   7.80 | Tokens / Sec:  6323.2 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   7.57 | Tokens / Sec:  6393.5 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   7.37 | Tokens / Sec:  6230.5 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   7.19 | Tokens / Sec:  6350.6 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   6.93 | Tokens / Sec:  6246.3 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   6.71 | Tokens / Sec:  6232.7 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   6.51 | Tokens / Sec:  6186.8 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   6.15 | Tokens / Sec:  6180.2 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.94 | Tokens / Sec:  6238.8 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.83 | Tokens / Sec:  6228.6 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   5.61 | Tokens / Sec:  6241.1 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   5.41 | Tokens / Sec:  6083.4 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   5.06 | Tokens / Sec:  6263.4 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.97 | Tokens / Sec:  6355.4 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.99 | Tokens / Sec:  6195.1 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.95 | Tokens / Sec:  6337.4 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.81 | Tokens / Sec:  6272.9 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.58 | Tokens / Sec:  6245.2 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.72 | Tokens / Sec:  6255.2 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.51 | Tokens / Sec:  6316.7 | Learning Rate: 2.4e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 16% |
[GPU 0] Epoch 0 Validation ====
Validation loss: tensor(4.4732, device='cuda:0')
[GPU 0] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   4.67 | Tokens / Sec:  6508.6 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   4.23 | Tokens / Sec:  6255.5 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   4.52 | Tokens / Sec:  6157.2 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   4.54 | Tokens / Sec:  6280.8 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   4.04 | Tokens / Sec:  6220.6 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.94 | Tokens / Sec:  6176.9 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   4.12 | Tokens / Sec:  6230.7 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   4.10 | Tokens / Sec:  6233.8 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   4.08 | Tokens / Sec:  6210.3 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   4.08 | Tokens / Sec:  6153.8 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   4.11 | Tokens / Sec:  6266.1 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.79 | Tokens / Sec:  6285.0 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.80 | Tokens / Sec:  6124.9 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.56 | Tokens / Sec:  6265.8 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.71 | Tokens / Sec:  6181.8 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.42 | Tokens / Sec:  6208.8 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.90 | Tokens / Sec:  6002.4 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.69 | Tokens / Sec:  6173.6 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.42 | Tokens / Sec:  5929.1 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.43 | Tokens / Sec:  6322.7 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.63 | Tokens / Sec:  6131.0 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.34 | Tokens / Sec:  6235.0 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.20 | Tokens / Sec:  6148.9 | Learning Rate: 4.8e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 17% |
[GPU 0] Epoch 1 Validation ====
Validation loss: tensor(3.2476, device='cuda:0')
[GPU 0] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.49 | Tokens / Sec:  6532.6 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.15 | Tokens / Sec:  6291.7 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.21 | Tokens / Sec:  6132.9 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.41 | Tokens / Sec:  6167.7 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.22 | Tokens / Sec:  6243.9 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.20 | Tokens / Sec:  6168.5 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.22 | Tokens / Sec:  6179.4 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.04 | Tokens / Sec:  6215.8 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.12 | Tokens / Sec:  6146.8 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.72 | Tokens / Sec:  6192.0 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.97 | Tokens / Sec:  6164.4 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.89 | Tokens / Sec:  6308.2 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.85 | Tokens / Sec:  6130.0 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.88 | Tokens / Sec:  6236.8 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.62 | Tokens / Sec:  6197.4 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.99 | Tokens / Sec:  5906.8 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.66 | Tokens / Sec:  6082.0 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.84 | Tokens / Sec:  6213.4 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.79 | Tokens / Sec:  6218.9 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.27 | Tokens / Sec:  6181.2 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.36 | Tokens / Sec:  6244.6 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.52 | Tokens / Sec:  6200.3 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.44 | Tokens / Sec:  6246.7 | Learning Rate: 7.3e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 17% |
[GPU 0] Epoch 2 Validation ====
Validation loss: tensor(2.2418, device='cuda:0')
[GPU 0] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.18 | Tokens / Sec:  7313.6 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.56 | Tokens / Sec:  6145.7 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.29 | Tokens / Sec:  6176.6 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.21 | Tokens / Sec:  6182.9 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.20 | Tokens / Sec:  6212.7 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.04 | Tokens / Sec:  6157.6 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.21 | Tokens / Sec:  6110.3 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.26 | Tokens / Sec:  6189.4 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.86 | Tokens / Sec:  6205.7 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.35 | Tokens / Sec:  6169.6 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.93 | Tokens / Sec:  6217.2 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.48 | Tokens / Sec:  6180.3 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.13 | Tokens / Sec:  6205.0 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.47 | Tokens / Sec:  6110.1 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.31 | Tokens / Sec:  6234.3 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.17 | Tokens / Sec:  6243.1 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.88 | Tokens / Sec:  6193.2 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.92 | Tokens / Sec:  6162.4 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.86 | Tokens / Sec:  6246.4 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.11 | Tokens / Sec:  6136.5 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.00 | Tokens / Sec:  6234.0 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.73 | Tokens / Sec:  6231.0 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.85 | Tokens / Sec:  6209.3 | Learning Rate: 7.4e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 16% |
[GPU 0] Epoch 3 Validation ====
Validation loss: tensor(1.6776, device='cuda:0')
[GPU 0] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.74 | Tokens / Sec:  7019.6 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.64 | Tokens / Sec:  6198.0 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.93 | Tokens / Sec:  6259.8 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.74 | Tokens / Sec:  6188.7 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.86 | Tokens / Sec:  6246.5 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.94 | Tokens / Sec:  6177.6 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.78 | Tokens / Sec:  6161.6 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.88 | Tokens / Sec:  6239.3 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.60 | Tokens / Sec:  6189.5 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.60 | Tokens / Sec:  6193.1 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.80 | Tokens / Sec:  6184.3 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.25 | Tokens / Sec:  6150.7 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.51 | Tokens / Sec:  6126.1 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.48 | Tokens / Sec:  6225.9 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.86 | Tokens / Sec:  6160.1 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.44 | Tokens / Sec:  6235.4 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.86 | Tokens / Sec:  6199.3 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.40 | Tokens / Sec:  6249.7 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.45 | Tokens / Sec:  6293.8 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.49 | Tokens / Sec:  6237.7 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.49 | Tokens / Sec:  6130.2 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.63 | Tokens / Sec:  6146.3 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.94 | Tokens / Sec:  6152.8 | Learning Rate: 6.6e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 16% |
[GPU 0] Epoch 4 Validation ====
Validation loss: tensor(1.3260, device='cuda:0')
[GPU 0] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.47 | Tokens / Sec:  7001.2 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.33 | Tokens / Sec:  6229.4 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.56 | Tokens / Sec:  6333.9 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.20 | Tokens / Sec:  6288.0 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.25 | Tokens / Sec:  6162.7 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.30 | Tokens / Sec:  6191.7 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.35 | Tokens / Sec:  6147.0 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.47 | Tokens / Sec:  6244.2 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.35 | Tokens / Sec:  6145.3 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.33 | Tokens / Sec:  6164.4 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.35 | Tokens / Sec:  6255.3 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.34 | Tokens / Sec:  6137.9 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.40 | Tokens / Sec:  6110.8 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.21 | Tokens / Sec:  6192.4 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.48 | Tokens / Sec:  6223.7 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.35 | Tokens / Sec:  6203.7 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.52 | Tokens / Sec:  6220.9 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.57 | Tokens / Sec:  6156.4 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.57 | Tokens / Sec:  6229.8 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.45 | Tokens / Sec:  6220.8 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.40 | Tokens / Sec:  6208.6 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.62 | Tokens / Sec:  6209.8 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.24 | Tokens / Sec:  6199.9 | Learning Rate: 6.0e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 17% |
[GPU 0] Epoch 5 Validation ====
Validation loss: tensor(1.0998, device='cuda:0')
[GPU 0] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.17 | Tokens / Sec:  6668.5 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.19 | Tokens / Sec:  6175.8 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.09 | Tokens / Sec:  6190.0 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.18 | Tokens / Sec:  6200.6 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.39 | Tokens / Sec:  6170.6 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.38 | Tokens / Sec:  6198.2 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.06 | Tokens / Sec:  6267.9 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.53 | Tokens / Sec:  6250.5 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.28 | Tokens / Sec:  6213.1 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.11 | Tokens / Sec:  6190.6 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.04 | Tokens / Sec:  6198.9 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.41 | Tokens / Sec:  6200.9 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.12 | Tokens / Sec:  6156.4 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.10 | Tokens / Sec:  6200.6 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.18 | Tokens / Sec:  6154.9 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.26 | Tokens / Sec:  6153.3 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.38 | Tokens / Sec:  6252.6 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.32 | Tokens / Sec:  6273.0 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.36 | Tokens / Sec:  6160.6 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.24 | Tokens / Sec:  6243.6 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.36 | Tokens / Sec:  6180.7 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.45 | Tokens / Sec:  6208.4 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.35 | Tokens / Sec:  6235.4 | Learning Rate: 5.6e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 17% |
[GPU 0] Epoch 6 Validation ====
Validation loss: tensor(0.9247, device='cuda:0')
[GPU 0] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.00 | Tokens / Sec:  6535.9 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.04 | Tokens / Sec:  6184.7 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.03 | Tokens / Sec:  6189.0 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.98 | Tokens / Sec:  6164.7 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.93 | Tokens / Sec:  6289.2 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.97 | Tokens / Sec:  6319.8 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.27 | Tokens / Sec:  6222.1 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.78 | Tokens / Sec:  6167.4 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.10 | Tokens / Sec:  6148.5 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.08 | Tokens / Sec:  6145.9 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.10 | Tokens / Sec:  6258.6 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.19 | Tokens / Sec:  6176.7 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.02 | Tokens / Sec:  6156.0 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.11 | Tokens / Sec:  6189.2 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.20 | Tokens / Sec:  6187.5 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.21 | Tokens / Sec:  6265.3 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.13 | Tokens / Sec:  6136.6 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.06 | Tokens / Sec:  6288.5 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.09 | Tokens / Sec:  6152.0 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.08 | Tokens / Sec:  6277.1 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.04 | Tokens / Sec:  6197.3 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.08 | Tokens / Sec:  6283.9 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.20 | Tokens / Sec:  6194.0 | Learning Rate: 5.2e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 16% |
[GPU 0] Epoch 7 Validation ====
Validation loss: tensor(0.7807, device='cuda:0')
Preparing Data ...
Checkin Model Output:

Example 0 ==================

Source Text (Input): <s> Viele weiß gekleidete Menschen in einem Stadion unterhalten sich miteinander . </s>
Target Text (Ground Truth): <s> Many people in a stadium dressed in white are conversing with each other . </s>
Target Text (Output): <s> Many people in white uniforms are talking to each other in a stadium . </s>

Example 1 ==================

Source Text (Input): <s> Ein Motocrossfahrer wird bei einem Sprung auf einer Rennstrecke leicht durch die Luft getragen . </s>
Target Text (Ground Truth): <s> A motocross rider is slightly airborne on a competition circuit jump . </s>
Target Text (Output): <s> A motocross rider is being carried in the air on a racetrack . </s>

Example 2 ==================

Source Text (Input): <s> Ein Mann in einem grauen T-Shirt ruht sich aus . </s>
Target Text (Ground Truth): <s> A man in a gray t - shirt rests . </s>
Target Text (Output): <s> A man in a gray t - shirt is resting . </s>

Example 3 ==================

Source Text (Input): <s> Ein älterer Mann sitzt im Freien vor einem großen Banner mit der Aufschrift „ <unk> <unk> <unk> <unk> “ auf einer Bank . </s>
Target Text (Ground Truth): <s> An older man is sitting outside on a bench in front a large banner that says , " <unk> <unk> <unk> <unk> . " </s>
Target Text (Output): <s> An older man sits outside of a large banner that says " <unk> " on a bench . </s>

Example 4 ==================

Source Text (Input): <s> Drei kleine Hunde schnüffeln an etwas . </s>
Target Text (Ground Truth): <s> Three small dogs <unk> at something . </s>
Target Text (Output): <s> Three small dogs sniffing something on something . </s>

Example 5 ==================

Source Text (Input): <s> Ein Boot mit Menschen und ihrem Hab und Gut befindet sich im Wasser . </s>
Target Text (Ground Truth): <s> A boat with people and their belongings is in the water . </s>
Target Text (Output): <s> A boat with people and their belongings in the water . </s>

Example 6 ==================

Source Text (Input): <s> Ein Boot mit Menschen und ihrem Hab und Gut befindet sich im Wasser . </s>
Target Text (Ground Truth): <s> A boat with people and their belongings is in the water . </s>
Target Text (Output): <s> A boat with people and their belongings in the water . </s>

Example 7 ==================

Source Text (Input): <s> Zwei Männer in Shorts arbeiten an einem blauen Fahrrad . </s>
Target Text (Ground Truth): <s> Two men wearing shorts are working on a blue bike . </s>
Target Text (Output): <s> Two men in shorts working on a blue bike . </s>

Example 8 ==================

Source Text (Input): <s> Ein Mann fährt ein altmodisches rotes Rennauto . </s>
Target Text (Ground Truth): <s> A man drives an old - fashioned red race car . </s>
Target Text (Output): <s> A man is driving an old red race car . </s>

Example 9 ==================

Source Text (Input): <s> Ein kleines blondes Mädchen hält ein Sandwich . </s>
Target Text (Ground Truth): <s> A small blond girl is holding a sandwich . </s>
Target Text (Output): <s> A little blond girl holding a sandwich . </s>

Example 10 ==================

Source Text (Input): <s> Ein schwarz gekleideter Junge schlägt ein Rad am Strand . </s>
Target Text (Ground Truth): <s> A boy in black clothes is doing a cartwheel on the beach . </s>
Target Text (Output): <s> A boy in black hits a wheel on the beach . </s>
Source Text (Input): <s> Eine Mutter füttert ihren Sohn mit <unk> </s>
Target Text (Ground Truth): <s> a mother is feeding her son with milk </s>
Target Text (Output): <s> A mother feeding her son with her son . </s>
```

- with ASUS TUF GeForce RTX 3090 O24G Gaming


```bash
# nvidia-smi 
Sat Mar 15 15:05:52 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.35.05              Driver Version: 560.35.05      CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:06:00.0  On |                  N/A |
|  0%   40C    P8             21W /  350W |     265MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
# python3 '/home/cherry/workspace/programming-pearls/machine-learning-note/transformer/harvardnlp_transformer.py'
Building German Vocabulary ...
Building English Vocabulary ...
load_vocab Finished.
Vocabulary sizes:
English Vocabulary size: 8316
Germany Vocabulary size: 6384
train_worker on GPU 0
[GPU 0] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   8.95 | Tokens / Sec:  2793.0 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   8.72 | Tokens / Sec:  6365.1 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   8.31 | Tokens / Sec:  6417.9 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   7.91 | Tokens / Sec:  6265.6 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   7.67 | Tokens / Sec:  6359.1 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   7.44 | Tokens / Sec:  6332.0 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   7.24 | Tokens / Sec:  6317.4 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   7.12 | Tokens / Sec:  6327.1 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   6.70 | Tokens / Sec:  6330.2 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   6.51 | Tokens / Sec:  6326.6 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   6.33 | Tokens / Sec:  6342.1 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.91 | Tokens / Sec:  6279.4 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.72 | Tokens / Sec:  6340.8 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   5.41 | Tokens / Sec:  6464.2 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   5.43 | Tokens / Sec:  6327.4 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   5.26 | Tokens / Sec:  6253.7 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   5.22 | Tokens / Sec:  6318.6 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.81 | Tokens / Sec:  6268.4 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   5.13 | Tokens / Sec:  6308.7 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.87 | Tokens / Sec:  6400.0 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.70 | Tokens / Sec:  6292.5 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.82 | Tokens / Sec:  6251.9 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.65 | Tokens / Sec:  6312.5 | Learning Rate: 2.4e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 15% |
[GPU 0] Epoch 0 Validation ====
Validation loss: tensor(4.4914, device='cuda:0')
[GPU 0] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   4.49 | Tokens / Sec:  7171.6 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   4.63 | Tokens / Sec:  6284.6 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   4.58 | Tokens / Sec:  6228.8 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   4.33 | Tokens / Sec:  5947.8 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   4.11 | Tokens / Sec:  6307.2 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   4.17 | Tokens / Sec:  6278.6 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   4.08 | Tokens / Sec:  6318.9 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   4.05 | Tokens / Sec:  6063.9 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   4.14 | Tokens / Sec:  6297.6 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.94 | Tokens / Sec:  6269.5 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.95 | Tokens / Sec:  6287.9 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   4.05 | Tokens / Sec:  6321.7 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   4.03 | Tokens / Sec:  6357.3 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.51 | Tokens / Sec:  6335.3 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.78 | Tokens / Sec:  6296.7 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.69 | Tokens / Sec:  6305.6 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.46 | Tokens / Sec:  6341.4 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.70 | Tokens / Sec:  6380.4 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.50 | Tokens / Sec:  6221.3 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.74 | Tokens / Sec:  6318.0 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.45 | Tokens / Sec:  6243.4 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.55 | Tokens / Sec:  6229.1 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.41 | Tokens / Sec:  6296.9 | Learning Rate: 4.8e-04
| ID | GPU | MEM |
------------------
|  0 | 94% | 16% |
[GPU 0] Epoch 1 Validation ====
Validation loss: tensor(3.2178, device='cuda:0')
[GPU 0] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.22 | Tokens / Sec:  7107.2 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.18 | Tokens / Sec:  6382.2 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.13 | Tokens / Sec:  6327.2 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.31 | Tokens / Sec:  6344.5 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.12 | Tokens / Sec:  6313.9 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.03 | Tokens / Sec:  6320.7 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.01 | Tokens / Sec:  6192.0 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.86 | Tokens / Sec:  6311.5 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.18 | Tokens / Sec:  6250.8 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.03 | Tokens / Sec:  6329.0 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.82 | Tokens / Sec:  6255.6 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.88 | Tokens / Sec:  6239.4 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.84 | Tokens / Sec:  6245.6 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.98 | Tokens / Sec:  6341.1 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.60 | Tokens / Sec:  6327.2 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.69 | Tokens / Sec:  6317.5 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.70 | Tokens / Sec:  6233.5 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.49 | Tokens / Sec:  6182.0 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.51 | Tokens / Sec:  6298.7 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.59 | Tokens / Sec:  6286.5 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.48 | Tokens / Sec:  6291.0 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.60 | Tokens / Sec:  6285.2 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.56 | Tokens / Sec:  6297.9 | Learning Rate: 7.3e-04
| ID | GPU | MEM |
------------------
|  0 | 92% | 16% |
[GPU 0] Epoch 2 Validation ====
Validation loss: tensor(2.2520, device='cuda:0')
[GPU 0] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.20 | Tokens / Sec:  7359.3 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.03 | Tokens / Sec:  6322.0 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.52 | Tokens / Sec:  6225.7 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.05 | Tokens / Sec:  6307.7 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.07 | Tokens / Sec:  6280.6 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.22 | Tokens / Sec:  6361.9 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.13 | Tokens / Sec:  6274.6 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.25 | Tokens / Sec:  6308.8 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.09 | Tokens / Sec:  6339.1 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.04 | Tokens / Sec:  6307.8 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.14 | Tokens / Sec:  6261.5 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.20 | Tokens / Sec:  6269.2 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.29 | Tokens / Sec:  6267.1 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.19 | Tokens / Sec:  6259.9 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.10 | Tokens / Sec:  6262.0 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.03 | Tokens / Sec:  6306.0 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.26 | Tokens / Sec:  6269.4 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.18 | Tokens / Sec:  6283.1 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.91 | Tokens / Sec:  6264.3 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.98 | Tokens / Sec:  6306.6 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.15 | Tokens / Sec:  6321.1 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.83 | Tokens / Sec:  6309.1 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.11 | Tokens / Sec:  6338.7 | Learning Rate: 7.4e-04
| ID | GPU | MEM |
------------------
|  0 | 91% | 16% |
[GPU 0] Epoch 3 Validation ====
Validation loss: tensor(1.6669, device='cuda:0')
[GPU 0] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.70 | Tokens / Sec:  7095.5 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.38 | Tokens / Sec:  6132.6 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.73 | Tokens / Sec:  6197.8 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.74 | Tokens / Sec:  6257.1 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.61 | Tokens / Sec:  6299.5 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.92 | Tokens / Sec:  6279.5 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.87 | Tokens / Sec:  6329.8 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.81 | Tokens / Sec:  6330.8 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.46 | Tokens / Sec:  6289.6 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.74 | Tokens / Sec:  6312.5 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.82 | Tokens / Sec:  6286.1 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.75 | Tokens / Sec:  6276.6 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.73 | Tokens / Sec:  6350.7 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.74 | Tokens / Sec:  6241.6 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.88 | Tokens / Sec:  6256.8 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.56 | Tokens / Sec:  6304.0 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.35 | Tokens / Sec:  6321.0 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.51 | Tokens / Sec:  6378.1 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.94 | Tokens / Sec:  6334.6 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.67 | Tokens / Sec:  6339.6 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.63 | Tokens / Sec:  6239.5 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.14 | Tokens / Sec:  6219.9 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.55 | Tokens / Sec:  6256.0 | Learning Rate: 6.6e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 16% |
[GPU 0] Epoch 4 Validation ====
Validation loss: tensor(1.3317, device='cuda:0')
[GPU 0] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.43 | Tokens / Sec:  7015.1 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.38 | Tokens / Sec:  6331.2 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.36 | Tokens / Sec:  6291.5 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.49 | Tokens / Sec:  6305.7 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.10 | Tokens / Sec:  6178.4 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.54 | Tokens / Sec:  6352.8 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.56 | Tokens / Sec:  6285.6 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.52 | Tokens / Sec:  6290.5 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.33 | Tokens / Sec:  6222.1 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.59 | Tokens / Sec:  6266.1 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.56 | Tokens / Sec:  6352.3 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.53 | Tokens / Sec:  6304.6 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.69 | Tokens / Sec:  6298.7 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.30 | Tokens / Sec:  6271.7 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.26 | Tokens / Sec:  6334.7 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.55 | Tokens / Sec:  6272.6 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.51 | Tokens / Sec:  6261.6 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.39 | Tokens / Sec:  6255.6 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.38 | Tokens / Sec:  6284.7 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.31 | Tokens / Sec:  6260.6 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.56 | Tokens / Sec:  6268.9 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.39 | Tokens / Sec:  6273.7 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.54 | Tokens / Sec:  6259.1 | Learning Rate: 6.0e-04
| ID | GPU | MEM |
------------------
|  0 | 93% | 16% |
[GPU 0] Epoch 5 Validation ====
Validation loss: tensor(1.0953, device='cuda:0')
[GPU 0] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.29 | Tokens / Sec:  6316.4 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.12 | Tokens / Sec:  6297.3 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.15 | Tokens / Sec:  6249.1 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.93 | Tokens / Sec:  6315.6 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.33 | Tokens / Sec:  6302.7 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.00 | Tokens / Sec:  6220.0 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.33 | Tokens / Sec:  6226.7 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.37 | Tokens / Sec:  6292.7 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.27 | Tokens / Sec:  6284.1 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.20 | Tokens / Sec:  6179.6 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.37 | Tokens / Sec:  6325.7 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.49 | Tokens / Sec:  6260.1 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.39 | Tokens / Sec:  6307.2 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.20 | Tokens / Sec:  6236.3 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.11 | Tokens / Sec:  6145.0 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.14 | Tokens / Sec:  6298.4 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.42 | Tokens / Sec:  6404.3 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.20 | Tokens / Sec:  6299.1 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.25 | Tokens / Sec:  6353.1 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.09 | Tokens / Sec:  6313.3 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.01 | Tokens / Sec:  6382.9 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.48 | Tokens / Sec:  6257.4 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.53 | Tokens / Sec:  6260.9 | Learning Rate: 5.6e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 16% |
[GPU 0] Epoch 6 Validation ====
Validation loss: tensor(0.9349, device='cuda:0')
[GPU 0] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.90 | Tokens / Sec:  7278.5 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.09 | Tokens / Sec:  6385.6 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.99 | Tokens / Sec:  6329.9 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.16 | Tokens / Sec:  6294.4 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.05 | Tokens / Sec:  6313.6 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.96 | Tokens / Sec:  6307.3 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.08 | Tokens / Sec:  6275.7 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.10 | Tokens / Sec:  6248.8 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.17 | Tokens / Sec:  6339.7 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.88 | Tokens / Sec:  6257.6 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.99 | Tokens / Sec:  6241.9 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.89 | Tokens / Sec:  6212.7 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.96 | Tokens / Sec:  6222.0 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.21 | Tokens / Sec:  5692.2 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.02 | Tokens / Sec:  6345.8 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.06 | Tokens / Sec:  6112.5 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.18 | Tokens / Sec:  6324.8 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.14 | Tokens / Sec:  6189.5 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.10 | Tokens / Sec:  6164.1 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.21 | Tokens / Sec:  6251.5 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.21 | Tokens / Sec:  6221.4 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.05 | Tokens / Sec:  6161.0 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.15 | Tokens / Sec:  6212.9 | Learning Rate: 5.2e-04
| ID | GPU | MEM |
------------------
|  0 | 90% | 16% |
[GPU 0] Epoch 7 Validation ====
Validation loss: tensor(0.7845, device='cuda:0')
Preparing Data ...
Checkin Model Output:

Example 0 ==================

Source Text (Input): <s> Eine lächelnde Frau mit einem pfirsichfarbenen Trägershirt hält ein Mountainbike </s>
Target Text (Ground Truth): <s> A smiling woman in a peach tank top stands holding a mountain bike </s>
Target Text (Output): <s> A smiling woman with a peach tank top holds a mountain bike . </s>

Example 1 ==================

Source Text (Input): <s> Eine Frau steht vor Bäumen und lächelt . </s>
Target Text (Ground Truth): <s> A woman standing in front of trees and smiling . </s>
Target Text (Output): <s> A woman is standing in front of trees and smiling . </s>

Example 2 ==================

Source Text (Input): <s> Eine Gruppe von Menschen sitzt draußen an einem Tisch , trinkt etwas und unterhält sich . </s>
Target Text (Ground Truth): <s> A group of people are sitting at a table outside having drinks and talking . </s>
Target Text (Output): <s> A group of people are sitting outside at a table having a drink . </s>

Example 3 ==================

Source Text (Input): <s> Das Bild eines Jungen in einem grünen T-Shirt , der auf einem Fahrrad sitzt , spiegelt sich in einer Schaufensterscheibe . </s>
Target Text (Ground Truth): <s> A boy wearing a green shirt on a bicycle reflecting off a store window . </s>
Target Text (Output): <s> This picture of a boy in a green shirt , taking a nap on a bicycle in a store . </s>

Example 4 ==================

Source Text (Input): <s> <unk> in Uniformen marschieren in einer Parade und spielen dabei <unk> Instrumente . </s>
Target Text (Ground Truth): <s> <unk> in uniforms march in a parade while playing flute - like instruments . </s>
Target Text (Output): <s> <unk> in uniforms are marching and playing instruments in a parade . </s>

Example 5 ==================

Source Text (Input): <s> Kinder baden im Wasser aus großen Fässern . </s>
Target Text (Ground Truth): <s> Children bathe in water from large drums . </s>
Target Text (Output): <s> Children are bathing in the water <unk> a large metal object . </s>

Example 6 ==================

Source Text (Input): <s> Ein Mann in einem weißen T-Shirt sitzt auf einer Kiste . </s>
Target Text (Ground Truth): <s> A man in a white shirt is sitting on a crate . </s>
Target Text (Output): <s> A man in a white t - shirt is sitting on a box . </s>

Example 7 ==================

Source Text (Input): <s> Ein Mann rennt mithilfe von Schneeschuhen durch den Schnee . </s>
Target Text (Ground Truth): <s> A man runs through the snow with the aid of snowshoes . </s>
Target Text (Output): <s> A man is running with his snowshoes through the snow . </s>

Example 8 ==================

Source Text (Input): <s> Männer in orangen Anzügen beobachten , wie eine Maschine in der Nähe von etwas gräbt , das wie <unk> aussieht . </s>
Target Text (Ground Truth): <s> Men in orange suits watching a machine dig near what looks to be subway tracks . </s>
Target Text (Output): <s> Men in orange suits watch as a machine digging near something . </s>

Example 9 ==================

Source Text (Input): <s> Ein junges Mädchen spielt ein Musikinstrument und singt in ein Mikrofon . </s>
Target Text (Ground Truth): <s> A young girl is playing a musical instrument and singing into a microphone . </s>
Target Text (Output): <s> A young girl is playing a musical instrument and singing into a microphone . </s>

Example 10 ==================

Source Text (Input): <s> Ein junges Mädchen spielt ein Musikinstrument und singt in ein Mikrofon . </s>
Target Text (Ground Truth): <s> A young girl is playing a musical instrument and singing into a microphone . </s>
Target Text (Output): <s> A young girl is playing a musical instrument and singing into a microphone . </s>
Source Text (Input): <s> Eine Mutter füttert ihren Sohn mit <unk> </s>
Target Text (Ground Truth): <s> a mother is feeding her son with milk </s>
Target Text (Output): <s> A mother is feeding her son with a <unk> . </s>
```