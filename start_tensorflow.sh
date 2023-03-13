salloc -p gpu_test -t 0-03:00 --mem=128000 --gres=gpu:1
module load python/3.8.5-fasrc01
module load cuda/11.7.1-fasrc01
module load cudnn/8.5.0.96_cuda11-fasrc01
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/helmod/apps/centos7/Core/cuda/11.7.1-fasrc01/cuda
source activate tf2.10_cuda11
