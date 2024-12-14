
# how to build tensorflow c lib: https://www.tensorflow.org/install/lang_c
# if you use macos, following these steps:
# 1. install bazel with bazelisk: brew install bazelisk
# 2. download tensorflow source code, and build c++ library: bazel build --verbose_failures //tensorflow:tensorflow_cc
#	and take bazel-bin/tensorflow/libtensorflow_cc.2.11.0.dylib
# 3. download tensorflow source code, and build c++ headers:  bazel build --verbose_failures //tensorflow:install_headers
# 4. if bazel failed to build install_headers, then try the following
#	download tensorflow pip package:
#		for macbook with m1 chip: https://developer.apple.com/metal/tensorflow-plugin/ 
# 		pip download tensorflow-macos
# 	extract wheel file: python -m --extract tensorflow_macos-{version}.whl
# 	take following directories and files:
#		drwxr-xr-x  10 godness  staff   320B Jan 19 21:11 include
#		-rw-r--r--   1 godness  staff    31M Jan 19 21:11 libtensorflow_framework.2.11.0.dylib
#		-rw-r--r--   1 godness  staff    31M Jan 19 21:11 libtensorflow_framework.2.dylib
#		-rw-r--r--   1 godness  staff    31M Jan 19 21:11 libtensorflow_framework.dylib
#  5. and oragnize previous files and directories as follows:
# 		> tree -L 2 libtensorflow/
# 		libtensorflow/
# 		├── include
# 		│   ├── Eigen
# 		│   ├── absl
# 		│   ├── external
# 		│   ├── google
# 		│   ├── include
# 		│   ├── tensorflow
# 		│   ├── third_party
# 		│   └── unsupported
# 		└── lib
# 		    ├── libtensorflow_cc.2.11.0.dylib
# 		    ├── libtensorflow_cc.2.dylib -> libtensorflow_cc.2.11.0.dylib
# 		    ├── libtensorflow_cc.dylib -> libtensorflow_cc.2.dylib
# 		    ├── libtensorflow_framework.2.11.0.dylib
# 		    ├── libtensorflow_framework.2.dylib -> libtensorflow_framework.2.11.0.dylib
# 		    └── libtensorflow_framework.dylib -> libtensorflow_framework.2.dylib

# some useful references:
#. https://tensorflow.google.cn/guide/basics
#. https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/architecture.md
#. https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/cc.md

cur_dir=${shell pwd}
export TF_HEADER_DIR = ${cur_dir}/libtensorflow/include
export TF_LIBRARY_DIR = ${cur_dir}/libtensorflow/lib

# how to print a variable in makefile: https://stackoverflow.com/questions/16467718/how-to-print-out-a-variable-in-makefile
$(info TF_HEADER_DIR: ${TF_HEADER_DIR})
$(info TF_LIBRARY_DIR: ${TF_LIBRARY_DIR})

# how to change environment var: https://stackoverflow.com/questions/8941110/how-i-could-add-dir-to-path-in-makefile
export LIBRARY_PATH := ${LIBRARY_PATH}:${TF_HEADER_DIR}
export DYLD_LIBRARY_PATH := ${DYLD_LIBRARY_PATH}:${TF_LIBRARY_DIR}

# add tensorflow libs to os env if you want to execute compiled binaries in shell
# export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$(pwd)/libtensorflow/lib

# Undefined symbols for architecture arm64: "_TF_Version", referenced from: ** ld: symbol(s) not found for architecture arm64
# add `-ltensorflow` when building TARGET_LIST

CXX = g++
CXXFLAGS += -I${TF_HEADER_DIR} -L${TF_LIBRARY_DIR}

# DO NOT link both of them
#CXXFLAGS += -ltensorflow # link c library
CXXFLAGS += -ltensorflow_cc # link c++ library

# Undefined symbols for architecture arm64: "tensorflow::NewSession
CXXFLAGS += -ltensorflow_framework

# Undefined symbols for architecture arm64: "tsl::io::internal::JoinPathImpl
CXXFLAGS += -std=c++1z # use c++17 standard

TARGET_LIST = hello_tf_main \
  			  basic_operations_main tensor_ops_main \
			  label_image_main wav_to_spectrogram_main mnist_demo_main

all: ${TARGET_LIST}

# makefile:23: *** missing separator.  Stop.
# indent using tab, and convert space to tab
hello_tf_main: hello_tf.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	./hello_tf_main

basic_operations_main: basic_ops/basic_operations.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	./basic_operations_main

tensor_ops_main: basic_ops/tensor_ops.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	./tensor_ops_main

label_image_main: label_image/label_image.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	#./label_image_main --image=./data/grace_hopper.jpg --graph=./data/inception_v3_2016_08_28_frozen.pb --labels=./data/imagenet_slim_labels.txt

wav_to_spectrogram/wav_to_spectrogram.o: wav_to_spectrogram/wav_to_spectrogram.cpp
	${CXX} -c ${CXXFLAGS} $? -o $@
wav_to_spectrogram_main: wav_to_spectrogram/main.cpp  wav_to_spectrogram/wav_to_spectrogram.o
	${CXX} ${CXXFLAGS} $? -o $@

mnist_demo_main: mnist_demo/mnist_demo.cpp
	${CXX} ${CXXFLAGS} $? -o $@
	./mnist_demo_main --image=./data/mnist_demo.jpg --graph=./data/mnist_cpu

clean:
	rm -rf ${TARGET_LIST} */*.o */*.out */*.dSYM
