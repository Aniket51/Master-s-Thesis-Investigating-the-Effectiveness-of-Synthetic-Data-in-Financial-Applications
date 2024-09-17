# Master's Thesis: Investigating the Effectiveness of Synthetic Data in Financial Applications

## Overview

This repository contains the implementation and results of my Master's thesis titled **"Investigating the Effectiveness of Synthetic Data in Financial Applications."** The study focuses on generating synthetic time-series data, particularly stock market data, using various machine learning and statistical models. The primary objective is to assess the effectiveness of synthetic data in replicating real financial data for applications like risk management, algorithmic trading, and fraud detection.

---

## Thesis Summary

The thesis investigates the potential of synthetic data in the financial domain, especially for sensitive and scarce datasets. Advanced methodologies like **TimeGAN**, **Long Short-Term Memory (LSTM)** networks, and **Block Bootstrapping** are employed to generate synthetic financial datasets.

The synthetic data is compared against real stock price data (from 2014-2024) using various metrics:
- **Statistical Similarity**: How closely the synthetic data resembles the real data in terms of statistical properties.
- **Predictive Accuracy**: The effectiveness of synthetic data in predictive financial models.
- **Visual Congruence**: A comparison of visual trends and patterns in real vs. synthetic datasets.

---

## Algorithms Used

- **Long Short-Term Memory (LSTM)**: A recurrent neural network model to generate synthetic time-series data by learning long-term dependencies.
- **Time-Series GAN (TimeGAN)**: Combines GAN and RNN to handle the temporal dynamics of time series, producing high-fidelity synthetic data.
- **Block Bootstrapping**: A statistical resampling technique that preserves the autocorrelation structure of time-series data.

The implementations of these models are provided in the `Algorithms/` directory. Each folder includes code, data preprocessing steps, and evaluations.

---

## Repository Structure
├── Algorithms/ │ ├── LSTM/ │ │ ├── lstm_model.py │ │ ├── data_preprocessing.py │ │ └── evaluation_metrics.py │ ├── TimeGAN/ │ │ ├── timegan_model.py │ │ ├── timegan_training.py │ │ └── evaluation_metrics.py │ └── Block_Bootstrapping/ │ ├── block_bootstrap.py │ └── evaluation.py ├── Paper/ │ └── Aniket_Dere_Masters_Thesis.pdf

