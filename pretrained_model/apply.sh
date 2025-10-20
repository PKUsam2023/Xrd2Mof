#PBS -N Apply
#PBS -l nodes=g06:ppn=8
#PBS -l walltime=1000:00:00
#PBS -q ml

source activate ${env}
cd ${raw_path}/Xrd2Mof-master/pretrained_model

# Execute the Python script with arguments
python -u apply.py > ${raw_path}/Xrd2Mof-master/generation_model/apply.out
