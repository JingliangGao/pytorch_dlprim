# Pytorch based on OpenCL backend

DLPrimitives-OpenCL out of tree backend for PyTorch, more details can be found [here](https://github.com/artyom-beilis/pytorch_dlprim).


# Tested Devices

DLPrimitves itself is tested on following devies: 

- [Nvidia](https://www.nvidia.cn/): GTX 960, A100/A5000, RTX 3060
- [AMD](https://www.amd.com/zh-cn.html): rx 6600XT/rx560, 
- [Intel](https://www.intel.cn/content/www/cn/zh/homepage.html):  HD 530/8570, UHD 630/770, Arc A380
- [Glenfly](https://www.glenfly.com/): Arise 2030
- [Innosilicon](https://www.innosilicon.cn/): Fantasy II-M
- [ZhaoXin](https://www.zhaoxin.com/): ZX C-1190


# Use     
1. set up environment
```shell
# system environment
sudo apt update
sudo apt install -y ocl-icd-opencl-dev opencl-headers
sudo mkdir -p /usr/local/include/CL && sudo mv tools/opencl.hpp /usr/local/include/CL/
sudo apt install -y libfmt-dev


# conda environment
conda create -n torch_cpu  python=3.12
conda activate torch_cpu
pip install torch torchvision  --index-url https://download.pytorch.org/whl/cpu
```

2. build project
```shell
chmod +x build-for-debug.sh && ./build-for-debug.sh
```

3. run example
```python
python3 examples/python/train_mnist.py
```

