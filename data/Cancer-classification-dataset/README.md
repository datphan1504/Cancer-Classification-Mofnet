# Multi-Omics Cancer Dataset

## Overview

This dataset is a curated subset of the MLOmics benchmark database for cancer machine learning research. It contains four cancer types with Top-feature selected multi-omics data, prepared for machine learning and bioinformatics tasks such as cancer subtype classification, clustering, and multi-omics integration.

The dataset includes four omics modalities for each cancer type:

- mRNA expression
- miRNA expression
- DNA Methylation (Methy)
- Copy Number Variation (CNV)

Only the **Top** feature version is included in this dataset.

Top features are identified through ANOVA statistical testing according to p-values, selecting the most significant features among samples. This approach helps:
- reduce noisy and irrelevant features,
- unify feature dimensions across datasets,
- improve computational efficiency,
- and potentially improve machine learning performance.

---

# Included Cancer Types

This dataset contains the following four cancer datasets:

| Cancer Type | Abbreviation | Description |
|---|---|---|
| Breast Invasive Carcinoma | BRCA | Breast cancer dataset |
| Colon Adenocarcinoma | COAD | Colon cancer dataset |
| Glioblastoma Multiforme | GBM | Brain tumor dataset |
| Ovarian Serous Cystadenocarcinoma | OV | Ovarian cancer dataset |

---

# Dataset Structure

```
dataset/
├── BRCA/
│   ├── BRCA_mRNA_top.csv
│   ├── BRCA_miRNA_top.csv
│   ├── BRCA_Methy_top.csv
│   ├── BRCA_CNV_top.csv
│   └── BRCA_label.csv
│
├── COAD/
│   ├── COAD_mRNA_top.csv
│   ├── COAD_miRNA_top.csv
│   ├── COAD_Methy_top.csv
│   ├── COAD_CNV_top.csv
│   └── COAD_label.csv
│
├── GBM/
│   ├── GBM_mRNA_top.csv
│   ├── GBM_miRNA_top.csv
│   ├── GBM_Methy_top.csv
│   ├── GBM_CNV_top.csv
│   └── GBM_label.csv
│
└── OV/
    ├── OV_mRNA_top.csv
    ├── OV_miRNA_top.csv
    ├── OV_Methy_top.csv
    ├── OV_CNV_top.csv
    └── OV_label.csv
```

# Omics Modalities

## 1. mRNA

Messenger RNA expression data representing gene activity levels.

- Rows: mRNA gene features
- Columns: patient samples
- Data type: continuous numerical values

### Example File
`BRCA_mRNA_top.csv`

### Example Features
- `KRT5`
- `KRT6A`
- `CEACAM5`

### Example

| Feature (Gene) | TCGA-IB-AAUW-01 | TCGA-AJ-A3EJ-01 |
|---|---|---|
| KRT5 | 0.565732 | -0.063888 |
| KRT6A | -0.122688 | -0.114109 |

---

## 2. miRNA

MicroRNA expression profiles involved in gene regulation.

- Rows: miRNA features
- Columns: patient samples
- Data type: continuous numerical values

### Example File
`BRCA_miRNA_top.csv`

### Example Features
- `hsa-miR-205-5p`
- `hsa-miR-375`
- `hsa-miR-200c-3p`

### Example

| miRNA Feature | TCGA-IB-AAUW-01 | TCGA-AJ-A3EJ-01 |
|---|---|---|
| hsa-miR-205-5p | 0.384463 | 0.854965 |
| hsa-miR-375 | 1.314672 | 0.004291 |

---

## 3. DNA Methylation (Methy)

Epigenetic methylation features related to gene regulation.

- Rows: methylation features / probes
- Columns: patient samples
- Data type: continuous numerical values

### Example File
`BRCA_Methy_top.csv`

### Example

| Methylation Feature | TCGA-IB-AAUW-01 | TCGA-AJ-A3EJ-01 |
|---|---|---|
| cg00000029 | 0.245 | -0.381 |
| cg00000108 | -0.112 | 0.593 |

---

## 4. Copy Number Variation (CNV)

Genomic structural variation features representing DNA copy number changes.

- Rows: CNV-related genomic features
- Columns: patient samples
- Data type: continuous numerical values

### Example File
`BRCA_CNV_top.csv`

### Example Features
- `EGFR`
- `CCND1`
- `FGF19`

### Example

| CNV Feature | TCGA-IB-AAUW-01 | TCGA-AJ-A3EJ-01 |
|---|---|---|
| EGFR | -0.407662 | -0.036677 |
| CCND1 | -0.244089 | -0.780234 |
