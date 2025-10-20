# Generation Model

This directory contains the code for Crystal Structure Pretiction. Follow the steps below to configure the environment and run evaluations.

---

## Acknowledgements

The main framework of this codebase is **built** upon:

* [DiffCSP](https://github.com/jiaor17/DiffCSP)
* [MOFDiff](https://github.com/microsoft/MOFDiff)

---

## Setup

1. **Environment file**
   Rename `.env.template` to `.env` and set the variables:

   ```text
   PROJECT_ROOT=<absolute path to this project>/generation_model
   HYDRA_JOBS=<absolute path where Hydra outputs should be stored>
   WABDB_DIR=<absolute path where WABDB outputs should be stored>
   ```

2. **Hydra run configuration**
   Before your first run, edit the generated Hydra config to point to your data:

   * Open
     `./Xrd2Mof-master/generation_model/output/hydra/singlerun/2025-04-03/CSP_CGmof50/hparams.yaml`
   * Replace the value of `data_name` with **your dataset filename**.

> Tip: If you prefer, you can instead pass/override `data_name` via CLI or a custom config to avoid manual edits.

---

## Training

For details on the training pipeline and configuration options, please refer to the original project:
[**MOFDiff**](https://github.com/microsoft/MOFDiff).

---

## Inference: Multiple Samples (CSP Task)

Run the evaluation script to generate multiple samples per input.

### Local run

```bash
PYTHONPATH=/absolute/path/to/your/project/Xrd2Mof-master/generation_model \
python ./Xrd2Mof-master/generation_model/scripts/evaluate.py \
  --model_path ./Xrd2Mof-master/generation_model/output/hydra/singlerun/2025-04-03/CSP_CGmof50 \
  --num_evals 5 \
  --label test \
  --out_path ./Xrd2Mof-master/data/generation_data \
  > ./Xrd2Mof-master/generation_model/test.txt
```

* `--model_path` points to the trained model directory or checkpoint.
* `--num_evals` is the number of samples per input.
* `--label` is an optional tag used in logs/filenames.
* `--out_path` is the output location for generated data.
* `PYTHONPATH` ensures the package modules in `generation_model` are discoverable.

### PBS cluster (optional)

```bash
qsub -v raw_path=/absolute/path/to/your/project \
        env=your_env_name \
     ./Xrd2Mof-master/generation_model/scripts/gen_CSP.sh
```

> Ensure your cluster job script activates the correct environment (e.g., `conda activate …`) and exports any required environment variables (e.g., `PROJECT_ROOT`, `PYTHONPATH`).
