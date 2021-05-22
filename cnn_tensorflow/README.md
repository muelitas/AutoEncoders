In this folder I will implement a CNN Autoencoder using tensorflow and keras.

First, clean install of Pop!_OS 20.04 LTS (the one with NVIDIA driver)

Then, these are the links used to install CUDA 11.2, cuDNN 8.1 and tensorflow 2.5.0
PRIMARY LINKS:
- https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-linux/index.html#post-installation-actions

When done with the post-instructions of the link above, run the following:
  sudo find / -name 'libcudart.so.11.0'
Assuming the output is:  
  /usr/local/cuda-11.2/targets/x86_64-linux/lib/libcudart.so.11.0
Add the following to your .bashrc file:
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/targets/x86_64-linux/lib
  
- https://developer.nvidia.com/cuda-11.2.2-download-archive
- https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-810/install-guide/index.html
- https://developer.nvidia.com/rdp/cudnn-archive
- https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html

Before installing tensorflow, I recommend installing Miniconda and creating an environment.
I created my environment with python=3.8.5

- https://www.tensorflow.org/install/gpu
- https://www.tensorflow.org/install/source#gpu

SECONDARY:
- https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d

When done with the links above, run the following:
  conda install -c conda-forge librosa 
  conda install -c anaconda spyder 
#Matplotlib, scikit-learn and tensorboard shoult had been installed already.
