#PBS -N Assemble
#PBS -l nodes=1:ppn=20
#PBS -l walltime=1000:00:00
#PBS -q d

source .bashrc
conda activate ${env}
cd ${raw_path}/Xrd2Mof-master/assemble

python -u mofdiff/scripts/assemble.py --input ${raw_path}/Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed.pt