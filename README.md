# Using Kubeflow Notebook

## 📌 Contents
- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Prerequisites Installation](#prerequisites-installation)
- [Usage](#usage)
- [To-Do](#to-do)

---

## 🧩 Introduction
This repository is designed to be used inside a Kubeflow Notebook. Its main purpose is to demonstrate how to use Kubeflow Notebooks and execute basic Kubeflow Pipelines (KFP) within it.

---

## 📁 Repository Structure

```text
git-underwrite-mlflow/
├── requirements.txt
├── README.md
├── config.yaml
├── LICENSE
├── data/
│   └── processed/
│       ├── train/
│       │   └── processed_train_v1.csv
│       └── test/
│           └── processed_test_v1.csv
├── pipeline/
│   ├── compose_pipeline.py
│   ├── credit_underwriting_pipeline.yaml
│   ├── run.py
│   └── components/
│       ├── preprocess_binning.py
│       ├── detect_feature_types.py
│       ├── dataloader.py
│       ├── modeling.py
│       └── settings.py
└── notebook/
    ├── modeling.ipynb
    └── eda.ipynb
```

---

## ⚙️ Prerequisites Installation

Please refer to the Kubeflow platform setup documented in my other repository:  
🔗 [An-end-to-end-MLops-with-Kubeflow](https://github.com/dohuyduc2002/An-end-to-end-MLops-with-Kubeflow)

### ✅ Enable Notebook to Run KFP
When creating a namespace in Kubeflow, check the **"tickbox"** to allow Notebooks to access KFP.

> 📌 *(Insert GIF here demonstrating the tickbox selection)*

### 🛠 Git Configuration in Notebook

By default, Kubeflow Notebooks have Git installed. To enable Git operations, configure your GitHub credentials as follows:

1. **Generate a GitHub personal access token:**  
   🔗 [https://github.com/settings/tokens](https://github.com/settings/tokens)

2. **Configure Git in Kubeflow Notebook terminal:**

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

3. **Set remote with token authentication:**

```bash
git remote set-url origin https://<github_username>:<access_token>@github.com/dohuyduc2002/git-underwrite-mlflow.git
```

> Upon pushing changes, GitHub may prompt for your username and token as a password.

---

## 🚀 Usage

The Notebook image used for this project is:  
**`microwave1005/scipy-img`** — a custom image prebuilt with all required dependencies.

To run the compiled Kubeflow Pipeline via CLI:

```bash
python run.py \
  --model_name lgbm \
  --version v1
```

Additional CLI arguments can be found in `pipeline/run.py`.

---

## 📌 To-Do
- [ ] Ingest secrets in the notebook (e.g., service endpoints, GCP credentials, etc.)
