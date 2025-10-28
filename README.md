# Pytorch based on OpenCL backend

DLPrimitives-OpenCL out of tree backend for pytorch, more details can be found [here](https://github.com/artyom-beilis/pytorch_dlprim).


# Tested Devices

DLPrimitves itself is tested on following devies: 

- AMD rx 6600XT with ROCM drivers, rx560 16cu with AMDGPU-pro drivers
- Nvidia: GTX 960
- Intel:  HD 530, UHD 630/770, Arc A380

# Use     
1. set up environment
```shell
conda create -n torch_cpu  python=3.12
conda activate torch_cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

2. build project
```shell
chmod +x build-for-debug.sh && ./build-for-debug.sh
```

3. run example
```python
python3 mnist.py --device ocl:0
```

