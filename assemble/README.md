# Assemble Generation Data

Convert the generated outputs into full atomic structures and optional downstream artifacts (CIFs, relaxed structures, PXRD).

---

## Acknowledgements

This assembly pipeline is **built upon**:

* [MOFDiff](https://github.com/microsoft/MOFDiff)

---

## Requirements

1. **Install package (editable)**
   Change into the assemble module and install the project so its modules can be imported anywhere:

```bash
cd ./Xrd2Mof-master/assemble
pip install -e .
```

2. **Inputs**
   Use the previously generated embedding outputs, e.g.:

* `./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5.pt`
* `./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt`

---

## Process

### Single-linker workflow

Reorganize generated samples using the single-linker pipeline:

```bash
python -u ./Xrd2Mof-master/assemble/Single-MOF_reorganize.py \
  --input_pt  ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5.pt \
  --bbs_pt    ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt \
  --output_pt ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed.pt
```

### Dual-linker workflow

For dual-linker MOFs, use the multiprocessing pipeline to improve throughput:

```bash
python -u ./Xrd2Mof-master/assemble/Dual-MOF/Dual_data_split.py \
  --input_pt  ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5.pt \
  --bbs_pt    ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt \
  --output_pt ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed

python -u ./Xrd2Mof-master/assemble/Dual-MOF/Dual_ewald_summation.py \
  --input_dir  ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed \
  --output_dir ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed_pro \
  --workers    20

python -u ./Xrd2Mof-master/assemble/Dual-MOF/Dual_data_merge.py \
  --input_folder ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed_pro \
  --output_file  ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed.pt
```

---

## Assemble CIFs

### Local run

Build final CIF structures from the processed dataset:

```bash
python -u ./Xrd2Mof-master/assemble/mofdiff/scripts/assemble.py \
  --input ./Xrd2Mof-master/data/generation_data/gen_csp_test_num_eval_5_processed.pt
```

### PBS cluster (optional)

```bash
qsub -v raw_path=/absolute/path/to/your/project \
        env=your_env_name \
     ./Xrd2Mof-master/assemble/assemble.sh
```

**Output**

* Generated `.cif` files will be written to the `./Xrd2Mof-master/data/generation_data/cif` (or the folder configured in your script).

---

## Optional: Structure Relaxation

For analysis you may use [MOFid](https://github.com/snurr-group/mofid). Install MOFid following its [compilation guide](https://github.com/snurr-group/mofid/blob/master/compiling.md).

> Note: The generative modeling and simulation parts of this repository do **not** depend on MOFid.

Run UFF relaxation:

```bash
python -u ./Xrd2Mof-master/assemble/mofdiff/scripts/uff_relax.py \
  --input_folder      ./Xrd2Mof-master/data/generation_data/cif \
  --cif_output_folder ./Xrd2Mof-master/data/generation_data/relax \
  --mof_id_folder     ./Xrd2Mof-master/data/generation_data/mofid
```

---

## Optional: PXRD Simulation

After CIFs are produced, you can compute simulated PXRD using the provided script:

```bash
python -u ./Xrd2Mof-master/assemble/get_xrd.py
```

---

## Notes & Tips

* **Paths**: Prefer absolute paths in cluster jobs. Ensure the runtime environment activates the correct conda env and exposes required variables (e.g., `PYTHONPATH`, `PROJECT_ROOT` if used).
* **Throughput**: The dual-linker pipeline supports multiprocessing via `--workers`; tune to your hardware.
* **Data integrity**: Ensure input datasets (`*.pt`) come from matching preprocessing versions to avoid key mismatches.
