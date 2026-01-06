# Project Repository

## Environment Setup

Due to library conflicts between **SHAP 0.12.0** and **Captum 0.8.0**, which require completely incompatible versions of **NumPy**, the code for computing **Integrated Gradients (IG)** has been separated from the rest of the project.

As a result, **two different virtual environments are required**:

- **Integrated Gradients environment**
  - Requirements file: `requirements_IG.txt`
- **Main project environment (SHAP, LIME, training, etc.)**
  - Requirements file: `requirements.txt`

Please make sure to activate the appropriate virtual environment before running each script.

---

## Usage
### Hyperparameter Tuning
To perform hyperparameter tuning for the models, run:
```bash

python3 hyperparameter_tunning.py
```

### Explanations
To perform calulations of SHAP and LIME, run:
```bash
python3 explanation.py
```
To perform calulations of IG , run:
```bash
python3 explanationIG.py
```
