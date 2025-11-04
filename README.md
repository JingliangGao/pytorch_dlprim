# Pytorch based on OpenCL backend

DLPrimitives-OpenCL out of tree backend for PyTorch, more details can be found [here](https://github.com/artyom-beilis/pytorch_dlprim).


# Tested Devices

DLPrimitves itself is tested on following devies: 

- AMD: rx 6600XT/rx560, 
- Nvidia: GTX 960, A100/A5000, RTX 3060
- Intel:  HD 530/8570, UHD 630/770, Arc A380
- Glenfly: Arise 2030

# Use     
1. set up environment
```shell
# system environment
sudo apt update
sudo apt install -y ocl-icd-opencl-dev opencl-headers

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
python3 examples/mnist.py
```

