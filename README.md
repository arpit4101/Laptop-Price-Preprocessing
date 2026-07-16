# 💻 Laptop Price Dataset — Preprocessing & Feature Engineering

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

End-to-end data cleaning & feature engineering pipeline on a **raw, uncleaned laptop price dataset** scraped from an e-commerce platform. This project turns a messy 12-column dataset into a fully numeric, ML-ready feature set — with every preprocessing decision backed by explicit statistical and domain reasoning.

> 📄 Originally submitted as a graduate assignment for MSc Data Science & Artificial Intelligence (Data Preprocessing course). Republished here as a portfolio project.

---

## 📌 Problem

The raw dataset contained **1,303 records and 12 features** describing laptop hardware specs and price — but it was far from analysis-ready:

- Combined/compound fields (`ScreenResolution`, `Cpu`, `Memory` each packed multiple pieces of information into one string)
- Inconsistent units (screen size mixing cm and inches)
- Physically impossible values (e.g., a 0.0002 kg laptop)
- A mislabeled brand (`Vero`, actually an Acer product line)
- 30 completely empty rows hiding among the data

The goal: produce a clean, fully numeric, leakage-free dataset suitable for regression modeling — and validate that the cleaning actually improved model performance.

---

## 🧹 What Was Done

| Stage | Key Actions |
|---|---|
| **Missing Values** | Diagnosed 30 fully-null rows (systemic data entry failure) → removed via listwise deletion. Remaining `?` values imputed using **grouped mean/mode imputation** (grouped by similar specs) instead of global imputation. |
| **Duplicates** | Verified zero duplicate records post-cleaning. |
| **Outliers** | Used **IQR / Tukey's fences** (not Z-score, due to right-skewed price distribution). 28 high-price outliers identified as legitimate premium laptops and retained to avoid selection bias. |
| **Structural Fixes** | Corrected mislabeled brand (`Vero` → `Acer`), converted cm → inches, stripped unit suffixes (`GB`, `kg`), fixed an erroneous RAM value via external verification. |
| **Feature Decomposition** | Split `ScreenResolution` → resolution + panel type; `Cpu` → brand, model, clock speed; `Memory` → SSD/HDD/Flash storage (GB) via regex parsing. |
| **Encoding** | Hybrid strategy — **Ordinal** (screen quality tiers), **One-Hot** (low-cardinality nominal features), **Label** (high-cardinality features like `Company`, `Gpu`). |
| **Scaling** | **MinMax scaling**, fit on training data only to prevent leakage. |
| **Feature Engineering** | Created `ProcessingPower = RAM × Cpu_speed` composite feature; dropped originals to reduce multicollinearity. |

---

## 📊 Results

**Final dataset:** 1,273 rows × 27 features, zero nulls, zero duplicates, fully numeric.

To validate the pipeline, two baseline regression models were trained on an 80/20 split:

| Model | MSE | R² Score |
|---|---|---|
| Linear Regression | 382,517,897.42 | 0.7363 (~73.6%) |
| **Random Forest Regressor** | **225,761,240.25** | **0.8444 (~84.4%)** |

The strong R² from a *baseline* Random Forest (no hyperparameter tuning) suggests the engineered features carry real predictive signal.

---

## 🗂️ Repository Structure

```
├── data/
│   └── laptop_price_raw.csv        # original uncleaned dataset
├── notebooks/
│   └── preprocessing_pipeline.ipynb
├── report/
│   └── DPP_Assignment1_Report.pdf  # full write-up with justifications
└── README.md
```

*(Adjust to match your actual file layout before pushing.)*

---

## 🧠 Why These Choices? (Highlights)

- **Listwise deletion over imputation for the 30 empty rows** — with every feature (including the target) missing, any imputed value would be entirely synthetic.
- **Retained high-price outliers** — removing them would bias the model toward underpredicting premium laptops.
- **Grouped imputation over global mean/mode** — a 10" netbook and a 17" gaming laptop shouldn't share an imputed value; grouping on similar specs keeps imputation contextually realistic.
- **Decomposed rather than label-encoded compound strings** — collapsing `ScreenResolution` or `Cpu` into arbitrary integers would destroy meaningful sub-features (panel quality, resolution tier, clock speed) that individually drive price.

Full reasoning for every decision is documented in the [assignment report](./report/DPP_Assignment1_Report.pdf).

---

## 🛠️ Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `regex`

---

## 👤 Author

**Arpit Dahyabhai Parmar**
MSc Data Science & Artificial Intelligence

---

⭐ If you found this useful as a reference for handling messy real-world datasets, consider starring the repo!
