#/usr/bin/env bash

set -x

SOURCE_DIR=`pwd`
BUILD_DIR=${BUILD_DIR:-./_build}

mkdir -p ${BUILD_DIR} \
	&& cd ${BUILD_DIR} \
	&& cmake ${SOURCE_DIR} \
	&& make $*

