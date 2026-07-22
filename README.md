# Time-Aware Adaptive Federated–Classical Hybrid Framework for Privacy-Preserving Electricity Theft Detection in Smart Grids

## Overview

Electricity theft remains one of the major challenges faced by modern smart grids, resulting in significant financial losses and reduced operational efficiency. Traditional centralized machine learning approaches require collecting consumer data at a central server, raising serious privacy concerns.

This project presents a **Time-Aware Adaptive Federated–Classical Hybrid Framework (TAAFCH)** that combines **Federated Learning**, **Gated Recurrent Units (GRU)**, and **Random Forest** to accurately detect multiple electricity theft scenarios while preserving consumer privacy. The proposed framework adaptively switches between global federated intelligence and local historical knowledge using a confidence-aware decision mechanism.

---

## Key Features

- Privacy-preserving Federated Learning architecture
- Time-series electricity consumption analysis using GRU
- Client-specific Random Forest classifier
- Adaptive confidence-aware prediction mechanism
- Multi-class electricity theft detection
- Secure distributed model training without sharing raw data
- Performance evaluation using multiple classification metrics
- Comparative analysis with conventional machine learning models

---

## Proposed Architecture
Electricity Consumption Dataset
│
▼
Data Preprocessing
│
▼
Feature Engineering
│
▼
Client Data Partitioning
│
┌────────┴────────┐
▼ ▼
Federated GRU Random Forest
(Local Training) (Client Model)
│ │
└────────┬────────┘
▼
Confidence-Aware Switching

## Methodology

The proposed framework consists of the following stages:

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Time-Series Sequence Generation
5. Federated Learning using GRU
6. Local Random Forest Training
7. Federated Averaging (FedAvg)
8. Adaptive Confidence-Based Prediction
9. Performance Evaluation

---

## Dataset

The framework is evaluated using the **TDD2022 Electricity Theft Detection Dataset**, containing electricity consumption records with multiple theft categories.

### Dataset Characteristics

- Hourly electricity consumption
- Multi-class theft labels
- Consumer energy usage patterns
- Temporal information
- Smart meter readings

---

## Technologies Used

### Programming Language

- Python

### Libraries

- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Flower (Federated Learning)
- TensorFlow Federated (optional)

---

## Project Structure
TAAFCH/
│
├── dataset/
│
├── preprocessing/
│
├── feature_engineering/
│
├── federated_learning/
│
├── random_forest/
│
├── hybrid_framework/
│
├── evaluation/
│
├── results/
│
├── models/
│
├── figures/
│
├── notebooks/
│
├── main.py
│
├── requirements.txt
│
└── README.md
## Performance Metrics

The proposed framework is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Training Time
- Communication Rounds
- Computational Complexity

---

## Workflow

1. Load electricity consumption dataset.
2. Perform preprocessing and feature engineering.
3. Generate temporal sequences.
4. Partition data among federated clients.
5. Train local GRU models.
6. Aggregate models using FedAvg.
7. Train client-specific Random Forest models.
8. Apply confidence-aware adaptive prediction.
9. Evaluate and compare results.

---

## Advantages

- Preserves consumer privacy.
- Eliminates centralized data collection.
- Learns temporal consumption behavior.
- Improves detection accuracy.
- Handles heterogeneous client data.
- Reduces communication overhead.
- Combines global and local intelligence.

---

## Applications

- Smart Grid Monitoring
- Electricity Theft Detection
- Utility Companies
- Smart Meter Analytics
- Distributed Energy Management
- Privacy-Preserving AI Systems

---

## Future Scope

- Real-time smart grid deployment
- Transformer-based temporal models
- Explainable AI (SHAP/LIME)
- Blockchain-enabled federated learning
- Edge AI implementation
- Federated optimization algorithms
- Large-scale distributed deployment

---

## Research Contribution

This work proposes a novel hybrid framework that integrates:

- Federated Learning
- Time-Series Deep Learning
- Random Forest
- Confidence-Aware Adaptive Decision Making

The framework achieves improved electricity theft detection while preserving user privacy by ensuring that raw consumer data never leaves local clients.

---

## Citation

If you use this work in your research, please cite:

**Prakki Krishna Karthik**, et al.

*Time-Aware Adaptive Federated–Classical Hybrid Framework for Privacy-Preserving Electricity Theft Detection in Smart Grids.*

---

## Author

**Prakki Krishna Karthik**

B.Tech Computer Science and Engineering

VIT Chennai

---

## License

This project is intended for academic and research purposes.
│
▼
Final Theft Prediction
