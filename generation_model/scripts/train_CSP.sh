#PBS -N Train
#PBS -l nodes=g00:ppn=8
#PBS -l walltime=100:00:00
#PBS -q ml

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
source activate ${env}
cd ${raw_path}/Xrd2Mof-master/generation_model

PYTHONPATH=${raw_path}/Xrd2Mof-master/generation_model \
python -u diffcsp/run.py data=CGmof_50 expname=CSP_CGmof50_layer_3 > ${raw_path}/Xrd2Mof-master/generation_model/train.txt
# tensorboard --logdir lightning_logs