# Transformer note

## References

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
    - [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)
    - [docker image](https://hub.docker.com/r/zhaokundev/annotated-transformer/)

## Env Setup

- install `python 3.10.12`: https://docs.vultr.com/update-python3-on-debian
    - https://www.python.org/downloads/source/
- set pip index-url: `pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/`
- create virtual environment with python3.10

- install packages in venv

```bash
pip3 install torch==2.1.0+cu121 torchtext==0.16 -f https://mirrors.aliyun.com/pytorch-wheels/cu121
pip3 install spacy pandas altair jupyter jupytext flake8 black GPUtil wandb 
pip3 install 'numpy>=1.22.4,<2.0'
pip3 install 'portalocker>=2.0.0'
```

- download and install spacy tokenizer

```
# github: https://github.com/explosion/spacy-models/releases
# https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.2.0/de_core_news_sm-3.2.0-py3-none-any.whl
#python3 -m spacy download de_core_news_sm
# https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/en_core_web_sm-3.2.0-py3-none-any.whl
#python3 -m spacy download en_core_web_sm
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