#PBS -N cal_embedding
#PBS -l nodes=g04:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source activate ${env}
cd ${raw_path}/Xrd2Mof-master/pretrained_model

# Execute the Python script with arguments
python -u cal_feature_embedding.py > ${raw_path}/Xrd2Mof-master/generation_model/cal_feature_embedding.out