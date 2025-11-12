# C++ example

## build
In the project main directory, cmake the project with option `-DBUILD_EXAMPLES` and then make it.

## example
1. infer_mobilenet     
infer the mnist net
```shell
python examples/cplus/infer_mobilenet/download_model.py
./build/debug/examples/cplus/infer_mobilenet
```

2. move_tensor       
move a tensor from CPU to GPU
```shell
./build/debug/examples/cplus/move_tensor
```