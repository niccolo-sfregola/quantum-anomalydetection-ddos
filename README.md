# Quantum-Enhanced DDoS Anomaly Detection

A hybrid quantum-classical framework for supervised and unsupervised Distributed Denial of Service (DDoS) detection using quantum reservoir computing and density-matrix-based anomaly scoring.

Built during the **QCentroid × GSMA Challenge** at the **ETH Quantum Hackathon 2026**, where the project won **1st place**.

---

## Overview

Modern intrusion detection systems increasingly rely on supervised learning pipelines trained on large labeled datasets. In practice, however, real-world network environments rarely provide exhaustive attack labels, and attack distributions evolve continuously over time.

This project explores whether **quantum-inspired representations** can provide useful structure for anomaly detection in low-label or label-free settings.

Instead of replacing the entire machine learning pipeline with a quantum model, the quantum component is applied to a single sub-task where it is naturally suited:

* encoding population-level traffic structure,
* extracting geometric observables from density matrices,
* and introducing adaptive memory through a fixed quantum reservoir.

The resulting framework supports both:

* **Supervised detection** using logistic regression
* **Unsupervised anomaly detection** using only benign traffic during training

The project focuses specifically on DDoS attacks, where malicious behavior often emerges only at the collective traffic level across short temporal windows.

---

## Key Features

* Quantum reservoir enrichment with fixed random quantum circuits
* Density matrix encoding of source-IP distributions
* Adaptive recurrent feedback using previous quantum measurements
* Quantum-native anomaly score via trace distance
* Entanglement entropy as an additional quantum observable
* Fully unsupervised anomaly detection pipeline
* No quantum parameter training
* Lightweight simulation executable on a laptop

---

## Pipeline Architecture

The framework consists of four main stages:

```text
1. Preprocessing
2. Aggregation
3. Quantum reservoir enrichment
4. Detection
```

### Supervised Pipeline

```text
Traffic windows
    ↓
Feature preprocessing
    ↓
Quantum reservoir enrichment
    ↓
Logistic Regression
    ↓
Attack / Benign classification
```

### Unsupervised Pipeline

```text
Traffic windows
    ↓
Feature preprocessing
    ↓
Quantum reservoir enrichment
    ↓
Anomaly scoring
    ├── Trace distance
    ├── Isolation Forest
    └── Hybrid ensemble
```

---

## Quantum Reservoir

The core of the project is a fixed quantum reservoir inspired by recurrent reservoir computing.

### Circuit Design

The reservoir uses:

* Random `RY` and `RZ` rotations
* Brick-wall `CNOT` entangling layers
* Circuit depth = 4
* No trainable quantum parameters

The random angles are sampled once during initialization and remain fixed throughout execution.

### Adaptive Memory

To introduce temporal structure, the expectation values from window `t` are injected into the encoding of window `t+1`.

This acts as a quantum analogue of recurrent hidden-state injection in classical recurrent neural networks.

Without this mechanism, each traffic window would be processed independently and temporal correlations would be lost.

---

## Quantum-Derived Features

Each traffic window produces several quantum observables.

### 1. Von Neumann Entropy

Measures the entropy of the IP-distribution density matrix:


S(rho) = -Tr(rho log rho)


Captures diversity and disorder in the population-level source-IP distribution.

---

### 2. Trace Distance from Benign Baseline

A geometric anomaly score computed between the current density matrix and a benign reference state:

T(rho, sigma) = 1/2 ||rho - sigma||_1

Operationally, the trace distance represents the maximum distinguishability between two quantum states.

This produces a fully unsupervised anomaly score requiring no attack labels.

---

### 3. Entanglement Entropy

The reservoir generates entanglement across qubits.

After tracing out half of the system, the entanglement entropy is computed:

S_E = -Tr(rho_A log rho_A)

This quantity has no compact classical analogue and captures higher-order feature correlations.

---

## Detection Methods

### Method A — Trace Distance

Purely quantum-inspired anomaly scoring.

The anomaly score is the trace distance between the current window density matrix and the benign baseline.

---

### Method B — Isolation Forest

Classical anomaly detection using only benign traffic during fitting.

The model assigns anomaly scores through Isolation Forest decision functions.

---

### Method C — Hybrid Ensemble

Combines normalized scores from:

* Trace distance
* Isolation Forest

The final score is computed as their average.

---

## Datasets

The framework was evaluated on two DDoS families:

### Family A

Synthetic bot-pool attacks:

* `NF-UNSW-NB15-v3`

### Family B

Native DDoS attacks:

* `NF-CSE-CIC-IDS2018`

---

## Results

### Supervised Detection

The supervised pipeline achieved near-perfect classification performance across multiple configurations.

Key observation:

> Quantum-inspired observables consistently receive strong weights in the learned decision boundary.

Even simple linear classifiers extract highly discriminative information from:

* trace distance,
* entropy,
* entanglement,
* and qubit expectation values.

---

### Unsupervised Detection

The unsupervised pipeline was trained exclusively on benign traffic.

Despite never observing attack labels during fitting, the model:

* reliably localized attack bursts,
* produced minimal false alarms,
* and achieved perfect recall in several configurations.

The hybrid ensemble reached:

* `F1 = 1.000`
* `AP = 1.000`
* `ROC-AUC = 1.000`

for most evaluated configurations.

---

## Computational Profile

The framework is intentionally lightweight.

* `0` trainable quantum parameters
* `4–10` qubits
* Quantum feature generation in approximately `~2 seconds`
* No backpropagation
* No barren plateaus
* No quantum hyperparameter optimization

The quantum reservoir acts as a deterministic feature enrichment block executable entirely through statevector simulation.


---


## Technologies Used

* Python
* Qiskit
* NumPy
* SciPy
* scikit-learn
* Matplotlib
* Pandas

---

## Limitations

This work does **not** claim a demonstrated quantum advantage.

The datasets used in evaluation are relatively structured, and several classical methods already achieve extremely strong performance.

Instead, the project investigates whether:

* quantum-inspired observables,
* density-matrix geometry,
* and entanglement-based representations

can provide useful inductive structure for future anomaly detection systems under:

* distribution shift,
* limited labels,
* and evolving attack patterns.

---

## Authors

Developed by **Qool Quids** during the ETH Quantum Hackathon 2026.

Team members:

* Niccolò Sfregola
* David Chudožilov
