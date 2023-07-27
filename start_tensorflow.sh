# salloc -p gpu_test -t 0-02:00 --mem=148000 --gres=gpu:2 #gpu_test
# module load python/3.8.5-fasrc01
# module load cuda/11.7.1-fasrc01
# module load cudnn/8.5.0.96_cuda11-fasrc01
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/11.7.1-fasrc01/cuda
# source activate tf2.10_cuda11

salloc -p gpu_test -t 0-02:00 --mem=148000 --gres=gpu:2 #gpu_test
module load python/3.10.9-fasrc01
source activate tf2.12_cuda11

# conda create -n tf2.10_cuda12 python=3.10 pip numpy six wheel scipy pandas matplotlib seaborn h5py jupyterlab