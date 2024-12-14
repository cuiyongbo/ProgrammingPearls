#!/usr/bin/env python3
#coding=utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

# asymmetric quantization: data is quantized with actual value range, resulting in possible less loss
'''
def calc_scale_z(x, int_min, int_max):
    scale = (x.max()-x.min())/(int_max-int_min)
    z = int_max - np.round(x.max()/scale)
    return scale, z

def saturate(x, int_min, int_max):
    return np.clip(x, int_min, int_max)

def quant(x, scale, z, int_min, int_max):
    return saturate(np.round(x/scale + z), int_min, int_max)

def dequant(xq, scale, z):
    return ((xq-z)*scale).astype('float32')

if __name__ == '__main__':
    np.random.seed(0)
    #raw_data = np.array([-0.61, -0.52, 1.62])
    raw_data = np.array([-1.62, -0.61, -0.52, 1.62])
    #raw_data = np.random.randn(3).astype('float32')
    print('raw data:', raw_data)
    #int_min, int_max = 0, 255
    #int_min, int_max = -128, 127
    int_min, int_max = -127, 127
    print('Q_min={}, Q_max={}'.format(int_min, int_max))
    scale, z = calc_scale_z(raw_data, int_min, int_max)
    print('scale={}, z={}'.format(scale, z))
    data_int = quant(raw_data, scale, z, int_min, int_max)
    print('quant data:', data_int)
    dequant_data = dequant(data_int, scale, z)
    print('dequant data:', dequant_data)
    print('diff:', dequant_data - raw_data)
'''


# symmetric quantization
def calc_scale_z(x, int_max):
    return x.max()/int_max, 0.0

def saturate(x, int_min, int_max):
    return np.clip(x, int_min, int_max)

def quant(x, scale, int_min, int_max):
    return saturate(np.round(x/scale), int_min, int_max)

def dequant(xq, scale):
    return ((xq)*scale).astype('float32')

def histogram_range(x, int_max):
    hist, range = np.histogram(x, 100)
    total = len(x)
    left = 0
    right = len(hist)-1
    threshold = 0.99
    while True:
        percent = hist[left:right+1].sum()/total
        if percent <= threshold:
            break
        if hist[left] < hist[right]:
            left += 1
        else:
            right -= 1
    left_val = range[left]
    right_val = range[right]
    dynamic_range = max(abs(left_val), abs(right_val))
    return dynamic_range/int_max

def histogram_demo(bins=20):
    np.random.seed(0)
    data = np.random.rand(1000)
    plt.hist(data, bins=bins)
    plt.title('histogram')
    plt.xlabel('value')
    plt.ylabel('freq')
    plt.show() 

if __name__ == '__main__':
    np.random.seed(0)
    #raw_data = np.array([-0.61, -0.52, 1.62])
    #raw_data = np.array([-1.62, -0.61, -0.52, 1.62])
    raw_data = np.random.randn(10000).astype('float32')
    #print('raw data:', raw_data)
    #int_min, int_max = 0, 255
    #int_min, int_max = -128, 127
    int_min, int_max = -127, 127
    print('Q_min={}, Q_max={}'.format(int_min, int_max))
    scale, z = calc_scale_z(raw_data, int_max)
    print('scale={}, z={}'.format(scale, z))
    scale2 = histogram_range(raw_data, int_max)
    print('histogram_range scale: {}'.format(scale2))
    #sys.exit(0)
    data_int = quant(raw_data, scale, int_min, int_max)
    print('quant data:', data_int)
    dequant_data = dequant(data_int, scale)
    print('dequant data:', dequant_data)
    print('diff:', dequant_data - raw_data)
