#PBS -N Generation
#PBS -l nodes=g04:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
source activate ${env}
cd  ${raw_path}/Xrd2Mof-master/generation_model/scripts

PYTHONPATH=${raw_path}/Xrd2Mof-master/generation_model \
python evaluate.py --model_path ${raw_path}/Xrd2Mof-master/generation_model/output/hydra/singlerun/2025-04-03/CSP_CGmof50 --num_evals 5 --label test --out_path ${raw_path}/Xrd2Mof-master/data/generation_data > ${raw_path}/Xrd2Mof-master/generation_model/test.txt