# PyTorch based on OpenCL backend

DLPrimitives-OpenCL out of tree backend for PyTorch, more details can be found [here](https://github.com/artyom-beilis/pytorch_dlprim).


## Tested Devices

pytorch_dlprim is tested on following devices:

- [Nvidia](https://www.nvidia.cn/): GTX 960, A100/A5000, RTX 3060
- [AMD](https://www.amd.com/zh-cn.html): rx 6600XT/rx560,
- [Intel](https://www.intel.cn/content/www/cn/zh/homepage.html):  HD 530/8570, UHD 630/770, Arc A380
- [Glenfly](https://www.glenfly.com/): Arise 2030
- [Innosilicon](https://www.innosilicon.cn/): Fantasy II-M
- [ZhaoXin](https://www.zhaoxin.com/): ZX C-1190


## Build Project
```shell
git clone --recursive https://github.com/JingliangGao/pytorch_dlprim.git
cd pytorch_dlprim && git checkout dev-opencl
chmod +x build-for-debug.sh && ./build-for-debug.sh
```

## Examples
1. train a mnist net

```shell
python3 examples/python/train_mnist.py
```

2. profile a mnist net
```shell
python3 examples/python/train_mnist.py --profile ./profile_log  # './profile_log' is the directory to save data
```

## Reformat code style
```shell
chmod +x tools/check_code_style.sh && ./tools/check_code_style.sh
```
