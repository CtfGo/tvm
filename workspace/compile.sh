#!/bin/bash

tvmc compile --target "llvm -mcpu=core-avx2" --output resnet50-v2-7-tvm.tar resnet50-v2-7.onnx
