# IGNIS: A Robust Neural Network Framework for Constrained Parameter Estimation in Archimedean Copulas

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
[![Copula Modeling](https://img.shields.io/badge/ML–Copula%20Estimation-ff69b4)](https://en.wikipedia.org/wiki/Copula_(probability))  
[![Parameter Estimation](https://img.shields.io/badge/Parameter%20Estimation-%CE%B8%E2%89%A5%201-green)](https://en.wikipedia.org/wiki/Parameter_estimation)

IGNIS (Latin for “fire”) is a unified neural estimation framework that delivers **robust**, **constraint-aware** parameter estimates for Archimedean copulas—even when classical methods fail due to non-monotonic mappings or pathological likelihood surfaces. IGNIS learns a direct mapping from data-driven dependency features to the copula parameter θ, enforcing θ ≥ 1 via a theory-guided softplus+1 output layer.

Code for Simulation Studies for IGNIS can be found in the Code_Sim.py file.

Real World Dataset 1: CDC Diabetes Health Indicators
Source: The dataset is the Diabetes Health Indicators dataset, publicly available from the UCI Machine Learning Repository (ID 891).
Access Method: The data is retrieved programmatically using the Python package ucimlrepo. The specific function call used is fetch_ucirepo(id=891). (Check Code_Real_World_Data.py file)
Variables Used: GenHlth: A self-reported measure of general health on a scale of 1 (excellent) to 5 (poor). PhysHlth: The number of days during the past 30 days that physical health was not good.

Real World Dataset 2: AAPL-MSFT Daily Stock Returns
Source: The data consists of historical daily stock prices from Yahoo! Finance.
Access Method: The data is downloaded programmatically using the Python package yfinance.  (Check Code_Real_World_Data.py file)
Variables Used: The daily Close price for Apple Inc. (AAPL) and Microsoft Corporation (MSFT) is used.

---

## 📦 Requirements

```bash
python >= 3.11
pip install numpy scipy scikit-learn tensorflow>=2.19 matplotlib
```

## 🚀 Getting Started

```bash
git clone https://github.com/agnivibes/IGNIS.git
cd IGNIS
```
## 🔬 Research Paper

Aich, A., Aich, A.B., Wade, B (2025). IGNIS: A Robust Neural Network Framework for Constrained Parameter Estimation in Archimedean Copulas. [Manuscript under review]

## 📊 Citation
If you use this code or method in your own work, please cite:

@article{Aich2025TCP,
  title   = {IGNIS: A Robust Neural Network Framework for Constrained Parameter Estimation in Archimedean Copulas},
  author  = {Aich, Agnideep, Aich, Ashit Baran and Wade, Bruce},
  year    = {2025},
  note    = {Manuscript under review}
  url     = {https://doi.org/10.48550/arXiv.2505.22518 }
}

## 📬 Contact
For questions or collaborations, feel free to contact:

Agnideep Aich,
Department of Mathematics, University of Louisiana at Lafayette
📧 agnideep.aich1@louisiana.edu

## 📝 License

This project is licensed under the [MIT License](LICENSE).
