# Create a Custom Test Dataset (CPU)

Use this guide to prepare your own experimental data for inference with the model. All paths below are relative to the project root.

> **Directory convention**
> Place all raw inputs in `./Xrd2Mof-master/data/original_data/`.

---

## 1) XRD Pattern Conversion

Convert each XRD pattern—regardless of its original format—into a single‐line PyTorch tensor file (`.pt`) containing relative intensities sampled at **0.02°** steps over **2θ = 5°–110°**.

**Requirements**

* File type: `.pt`
* Content: one 1-D tensor (relative intensities only; no headers)
* Length: **5250** values (because `(110 − 5) / 0.02 = 5250`)
* Grid: 2θ = **5.00°, 5.02°, …, 109.98°**

> Tip: ensure intensities are normalized (relative scale) and aligned to the exact grid above.

---

## 2) Metal Nodes & Organic Linkers (CSV)

Provide prior information about metals and linkers in a CSV with **six columns**:

| Column              | Description                                                                  |
| ------------------- | ---------------------------------------------------------------------------- |
| `Materials_name`    | Unique identifier per structure                                              |
| `Metal_nodes`       | **Exactly one** metal element symbol (e.g., `Zn`, `Er`) — **case-sensitive** |
| `Organic_linkers`   | SMILES strings for organic linkers, **comma-separated**                      |
| `Linker_proportion` | Optional. If there is only one linker, this may be left blank                |
| `Metal_valence`     | Valence of the metal                                                         |
| `Linker_valence`    | Optional. May be left blank                                                  |

**Notes**

* “case-sensitive” means `Fe` ≠ `fe` (use proper element capitalization).
* We have provided a template: "test_cases.csv" and xrd folder.

---

## 3) Run the Pre-Processing Scripts

From the **project root** (the parent directory of `Xrd2Mof-master/`), execute:

```bash
python -u ./Xrd2Mof-master/create_data/metal_nodes.py \
  --input_csv ./Xrd2Mof-master/data/original_data/test_cases.csv \
  --output_csv ./Xrd2Mof-master/data/original_data

python -u ./Xrd2Mof-master/create_data/organic_linkers.py \
  --input_csv ./Xrd2Mof-master/data/original_data/test_cases.csv \
  --output_dir  ./Xrd2Mof-master/data/original_data

python -u ./Xrd2Mof-master/create_data/create_dataset.py \
  --input_csv      ./Xrd2Mof-master/data/original_data/test_cases.csv \
  --output_pt      ./Xrd2Mof-master/data/original_data/dataset.pt \
  --error_log_file ./Xrd2Mof-master/data/original_data/log.txt
```

**Outputs**

* Processed metals file "metal.csv" under `./Xrd2Mof-master/data/original_data/`
* Processed linkers file "linker.pt" under `./Xrd2Mof-master/data/original_data/`
* Final dataset tensor at `./Xrd2Mof-master/data/original_data/dataset.pt`
* Error log (if any) at `./Xrd2Mof-master/data/original_data/log.txt`

---

## Validation Checklist

* ✅ Each `.pt` XRD file contains exactly **5250** floats (no headers, no NaNs).
* ✅ `Metal_nodes` contains **one** valid element symbol (proper case).
* ✅ `Organic_linkers` are valid **SMILES** (bare strings, comma-separated).
* ✅ CSV uses UTF-8 without BOM; no trailing spaces in headers or values.
* ✅ Paths in the commands match your actual directory layout.

---

## Troubleshooting (common)

* **KeyError on CSV columns**: verify exact column names (including case and spaces).
* **RDKit SMILES parse errors**: ensure values are bare SMILES, not list-like strings (e.g., convert `...` → `['...']`).
* **Wrong XRD length**: resample to the required 5°–110°/0.02° grid to reach 5250 points.
