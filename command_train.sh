/opt/TurboVNC/bin/vncserver -geometry 1920x1080 -depth 24 :8
export DISPLAY=:8
export XAUTHORITY=/home/asus/.Xauthority 

# for install ME
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVCC=$CUDA_HOME/bin/nvcc
export SETUPTOOLS_USE_DISTUTILS=stdlib
export PIP_DISABLE_PEP517=1



CUDA_VISIBLE_DEVICES=4 


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python train.py --camera kinect --log_dir logs/log_kn --batch_size 4 --learning_rate 0.001 \
--model_name minkuresunet --dataset_root /home/asus/Research/datasets/graspnet


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python tp.py --camera kinect --log_dir logs/log_kn --batch_size 4 --learning_rate 0.001 \
--model_name minkuresunet --dataset_root /home/asus/Research/datasets/graspnet


export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python train_vggt.py --camera kinect --log_dir logs/log_kn --batch_size 4 --learning_rate 0.001 \
--model_name minkuresunet --dataset_root /home/asus/Research/datasets/graspnet