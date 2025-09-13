CS771 Mini Project – Breaking Multi-Level PUFs & Arbiter PUF Delay Recovery
📌 Project Overview

This project was completed as part of CS771: Introduction to Machine Learning (IIT Kanpur, Spring 2025) under the guidance of Prof. Purushottam Kar.

The mini-project consists of two main tasks:

Breaking Multi-Level PUFs (ML-PUFs) – Designing linear feature maps and learning models that can predict the XOR response of ML-PUFs.

Arbiter PUF Delay Recovery – Recovering non-negative stage delays from a given linear model representation of a 64-bit Arbiter PUF.

Our work demonstrates that even though PUFs are designed to be unclonable, carefully constructed linear models can effectively learn and invert their behavior.

📂 Repository Structure
.
├── CS771_Project.py    # Final code implementation (submit.py equivalent)
├── CS771_Report.pdf    # Detailed report with derivations, results, and experiments
├── CS771_PS.pdf        # Problem statement provided by the course
└── README.md           # Project documentation

⚙️ Implementation
1. Breaking ML-PUFs

Implemented a feature mapping function (my_map) to transform 8-bit challenges into higher-dimensional linear feature space.

Trained a logistic regression model (my_fit) with L1 penalty to achieve high accuracy while ensuring efficiency.

Compared performance with LinearSVC, analyzing the effect of loss functions, regularization (C), tolerance (tol), and penalty types (l1 vs l2) on accuracy and training speed.

Key Function:

def my_fit(X_train, y_train):
    feat_train = my_map(X_train)
    model = LogisticRegression(
        C=1,
        penalty='l1',
        solver='liblinear',
        tol=0.01,
        max_iter=1000,
        fit_intercept=True,
        random_state=42
    )
    model.fit(feat_train, y_train)
    return model.coef_.flatten(), model.intercept_[0]

2. Arbiter PUF Delay Recovery

Formulated the forward Arbiter PUF model as a sparse system of linear equations (A·x = y).

Implemented my_decode() using non-negative least squares via linear regression to recover valid non-negative delays.

Output consists of four 64-dimensional vectors (p, q, r, s).

📊 Results
ML-PUF Classification

Feature Dimension: 256 (quadratic expansion of 16 base features).

Best Model: Logistic Regression with L1 penalty.

Accuracy: >97% on public test set.

Observations:

Lower tol improved accuracy but increased training time.

L1 penalty induced sparsity, improving interpretability.

Logistic Regression outperformed LinearSVC for high C values.

Arbiter PUF Inversion

Method: Non-negative least squares.

Recovered Delays: 256 per model (p, q, r, s).

Performance Metric: Low Euclidean distance between reconstructed and true linear models.

📜 Report

The detailed mathematical derivations, kernel SVM analysis, feature dimensionality proofs, and experimental outcomes are available in CS771_Report.pdf
.

🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/CS771-PUF-Project.git
cd CS771-PUF-Project


Ensure dependencies are installed:

pip install numpy scikit-learn scipy


Import and test functions:

from CS771_Project import my_fit, my_map, my_decode


Run validation using the provided Colab script
.

👨‍💻 Contributors

Aditya Suman (EE, 220081)

Akansha Ratnakar (CHE, 220093)

Priyanshu Gujraniya (CE, 220825)

Aman Kumar Sriwastav (MSE, 220116)

Ashutosh Anand (ECO, 220239)

📖 References

CS771 Course Material, IIT Kanpur

Lecture notes on Arbiter PUFs and ML-PUFs

Scikit-learn documentation for linear models
