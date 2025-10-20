# Create Your Own Test Dataset

Prepare and run your data through the feature-extraction and diffusion-data pipelines.
All paths below are **relative to the project root**.
All raw inputs must be placed under `./Xrd2Mof-master/data/original_data/`.

---

## 1) Extract Feature Embeddings

Use the previously generated `dataset.pt` to compute feature embeddings.

1. **Configure project root**

   * In `./Xrd2Mof-master/pretrained_model/conf/config.yaml`, set:

     ```yaml
     raw_path: /absolute/path/to/your/project
     ```

2. **Run locally**

   ```bash
   python -u ./Xrd2Mof-master/pretrained_model/clip_cal_feature_embedding.py \
     > clip_cal_feature_embedding.out
   ```

3. **Run on a PBS cluster (optional)**

   ```bash
   qsub -v raw_path=/absolute/path/to/your/project \
           env=your_env_name \
        ./Xrd2Mof-master/pretrained_model/cal_feature_embedding.sh
   ```

**Output**

* `./Xrd2Mof-master/data/feature_extract_data/dataset_emb.pt`

---

## 2) Post-process Feature Embeddings

Combine the embedding file with metal-node and organic-linker metadata.
The BBS database `valid_bbs_space_408626.pt` and `linker_valence_table.pt` should be downloaded in advance.

```bash
python -u ./Xrd2Mof-master/pretrained_model/script/process_feature_embedding.py \
  --pt_file   ./Xrd2Mof-master/data/feature_extract_data/dataset_emb.pt \
  --smi_csv   ./Xrd2Mof-master/data/original_data/test_cases.csv \
  --bbs_space ./Xrd2Mof-master/data/database/valid_bbs_space_408626.pt \
  --map_linker_table ./Xrd2Mof-master/data/database/linker_valence_table.pt \
  --output    ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed.pt
```

**Output**

* `./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed.pt`

---

## 3) Create Diffusion-Ready Data

Use `dataset_emb_processed.pt` to compute the atom counts and ratios required by the structure-generation (diffusion) models.
Choose the command that matches your MOF type:

### 3.1 Single-linker type (single metal + single linker)

```bash
python -u ./Xrd2Mof-master/pretrained_model/script/cal_num_type_to_diffusion_single.py \
  --input_pt  ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed.pt \
  --output_pt ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt
```

### 3.2 Dual-linker type (single metal + two linkers)

```bash
python -u ./Xrd2Mof-master/pretrained_model/script/cal_num_type_to_diffusion_single.py \
  --input_pt  ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed.pt \
  --output_pt ./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt
```

**Output**

* `./Xrd2Mof-master/data/feature_extract_data/dataset_emb_processed_diff.pt`

---

## Notes & Tips

* **CSV format**: **If you’re reasonably certain of the linker valence, please provide it.**
* **CSV format**: **Ensure the CSV data format matches the provided example exactly**—column names, types, and value formats (e.g., `Materials_name`, `Metal_nodes`, `Organic_linkers`, etc.).
* **Logging & errors**: Scripts emit warnings/errors (e.g., unparsable SMILES, missing mappings). Follow these messages to correct data issues.
