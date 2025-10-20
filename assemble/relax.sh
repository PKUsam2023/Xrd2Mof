#PBS -N Relax
#PBS -l nodes=1:ppn=96
#PBS -l walltime=100:00:00
#PBS -q nature

source .bashrc
conda activate ${env}
cd ${raw_path}/Xrd2Mof-master/assemble

python -u mofdiff/scripts/uff_relax.py \
  --input_folder      ${raw_path}/Xrd2Mof-master/data/generation_data/cif \
  --cif_output_folder ${raw_path}/Xrd2Mof-master/data/generation_data/relax \
  --mof_id_folder     ${raw_path}/Xrd2Mof-master/data/generation_data/mofid
