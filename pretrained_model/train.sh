#PBS -N Train
#PBS -l nodes=g00:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source activate ${env}
cd ${raw_path}/Xrd2Mof-master/pretrained_model

python -u train.py > ${raw_path}/Xrd2Mof-master/generation_model/train.out
