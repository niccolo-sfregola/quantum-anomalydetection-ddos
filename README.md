# Quantum-Enhanced DDoS Anomaly Detection

A hybrid quantum-classical framework for supervised and unsupervised Distributed Denial of Service (DDoS) detection using quantum reservoir computing, density-matrix-based anomaly scoring, source-IP attribution, and stealth-attack stress testing.

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
* **Source-IP attribution** for anomalous traffic windows
* **Stealth-attack simulation** for robustness testing

The project focuses specifically on DDoS attacks, where malicious behavior often emerges only at the collective traffic level across short temporal windows. The added attribution stage extends this from window-level detection to identifying the source IPs most associated with the suspicious activity.

---

## Key Features

* Quantum reservoir enrichment with fixed random quantum circuits
* Density matrix encoding of source-IP distributions
* Adaptive recurrent feedback using previous quantum measurements
* Quantum-native anomaly score via trace distance
* Entanglement entropy as an additional quantum observable
* Fully unsupervised anomaly detection pipeline
* Post-detection source-IP attribution from anomalous windows
* Optional stealth-attack dataset generation for stress testing
* No quantum parameter training
* Lightweight simulation executable on a laptop

---

## Pipeline Architecture

The framework consists of four core stages, with optional stealth-generation and attribution stages:

```text
0. Optional stealth attack generation
1. Preprocessing
2. Aggregation
3. Quantum reservoir enrichment
4. Detection
5. Source-IP attribution
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

### Source-IP Attribution Pipeline

```text
Flagged attack windows
    ↓
Per-window source-IP frequency tables
    ↓
Candidate source-IP ranking
    ↓
Malicious source-IP attribution
```

The attribution step is applied after a window has been flagged as anomalous. It reuses the per-window `src_ip` distributions produced during aggregation and ranks candidate IPs by how strongly they contribute to suspicious windows. In labeled experiments, the audit fields can be used only for validation of the attributed IPs, not as model inputs.

### Stealth-Attack Stress-Test Pipeline

```text
Original challenge datasets
    ↓
Stealth attack generation
    ↓
Standard preprocessing, aggregation, enrichment, and detection
    ↓
Robustness evaluation
```

The stealth-generation step creates modified attack datasets before the normal pipeline starts. Normal datasets are copied or sliced unchanged, while seeded attack rows are transformed to make the attack less obvious under simple volumetric rules.

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


S(ρ) = -Tr(ρ log ρ)


Captures diversity and disorder in the population-level source-IP distribution.

---

### 2. Trace Distance from Benign Baseline

A geometric anomaly score computed between the current density matrix and a benign reference state:


T(ρ, σ) = 1/2 ||ρ - σ||_1


Operationally, the trace distance represents the maximum distinguishability between two quantum states.

This produces a fully unsupervised anomaly score requiring no attack labels.

---

### 3. Entanglement Entropy

The reservoir generates entanglement across qubits.

After tracing out half of the system, the entanglement entropy is computed:


S_E = -Tr(ρ_A log ρ_A)


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

### Method D — Source-IP Attribution

Post-detection attribution is used to identify which source IPs are most associated with anomalous windows.

The attribution stage uses the saved `*_ip_distributions.csv` files generated during aggregation. For every window flagged by Method A, B, or C, the pipeline can inspect the source-IP frequency distribution and rank candidate malicious IPs by their contribution to the anomalous traffic population.

This keeps detection and attribution separated:

* detection answers **which windows are anomalous**,
* attribution answers **which source IPs are most suspicious inside those windows**.

The attribution stage is especially useful for DDoS settings because the signal is collective: a single flow is usually insufficient, while a short window of many coordinated source IPs can reveal the attack.

---

## Datasets

The framework was evaluated on two DDoS families:

### Family A

Synthetic bot-pool attacks:

* `NF-UNSW-NB15-v3`

### Family B

Native DDoS attacks:

* `NF-CSE-CIC-IDS2018`

### Stealth Attack Variants

The repository also supports generated stealth variants of the attack datasets.

The stealth generator creates a complete raw input tree that can be fed into the same preprocessing, aggregation, reservoir, detection, and attribution pipeline. In the default stress-test configuration, seeded attack rows are modified through:

* **Volume/rate reduction**, where packet and byte rate features are reduced by a configurable factor
* **Behavioral mimicry**, where selected non-audit numeric attack-flow features are shifted toward the benign reference distribution

For example, the current `stealth_mimic90_vol100x` configuration uses strong behavioral mimicry and a 100x volume reduction to test whether the pipeline still detects attacks when simple high-volume signatures are weakened.

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

### Source-IP Attribution

After anomalous windows are detected, the attribution pipeline can report the source IPs most responsible for the suspicious activity in those windows.

This provides supporting evidence beyond the binary window label and makes the system more actionable for network operators: instead of only reporting that a burst occurred, the pipeline can surface the candidate source IPs associated with that burst.

---

### Stealth-Attack Stress Testing

The stealth-attack pipeline is intended as a robustness test rather than a claim about all real-world adversarial behavior.

It evaluates whether the detector remains useful when attack rows are made less obvious by reducing volume/rate indicators and shifting selected flow-level features toward benign statistics. This is useful for comparing quantum-derived density-matrix features, classical flow aggregates, and hybrid ensembles under more challenging synthetic attack conditions.

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

The source-IP attribution stage adds minimal computational overhead because it operates on frequency tables already produced during aggregation. The stealth-attack generator is also preprocessing-only and does not introduce additional quantum simulation cost.


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

The attribution stage depends on the quality of the detected windows: missed attack windows cannot be attributed, and false-positive windows may produce misleading source-IP rankings. The audit fields are used only for validation and should not be used as model inputs.

The stealth-attack generator is a defensive stress-testing tool. It creates synthetic variants of the challenge data to evaluate robustness, but it does not claim to model every possible real-world stealth DDoS strategy.

---

## Authors

Developed by:

* Niccolò Sfregola
* David Chudožilov
