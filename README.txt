# Synthetic Data Generation and Anomaly Detection Framework

This project implements a comprehensive framework for **synthetic data generation** and **anomaly detection** using advanced machine learning techniques, including **Autoencoders**, **Variational Autoencoders (VAEs)**, and **TimeGAN**. It supports synthetic data generation for both general and anomalous data distributions, anomaly detection, and visualization.

---

## Features

### 1. **Anomaly Detection Using Autoencoder**
- Autoencoder-based anomaly detection using reconstruction errors.
- Visualizes reconstruction errors and highlights anomalies.
- Detects anomalies by setting a configurable percentile-based threshold.

### 2. **Synthetic Data Generation Using SDV and CTGAN**
- Generates realistic synthetic data using **SDV**’s CTGANSynthesizer.
- Produces both general synthetic data and anomaly-specific synthetic data.
- Combines original and synthetic data for extended datasets.

### 3. **TimeGAN for Time-Series Data**
- Implements TimeGAN for generating synthetic time-series data.
- Preserves temporal dynamics and feature distributions.
- Includes tools for preprocessing, training, and visualizing synthetic data.
- Provides side-by-side histogram comparisons of original and synthetic data.

---

## Requirements

Install the following libraries before running the scripts:
- `numpy`
- `pandas`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `sdv`

Install via pip:
```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn sdv
```

---

## Workflow Overview

### Step 1: Anomaly Detection
- Preprocesses the dataset by scaling and splitting it into training and test sets.
- Trains an Autoencoder to reconstruct the input data.
- Calculates reconstruction errors and detects anomalies based on a percentile threshold.

### Step 2: Synthetic Data Generation
- Generates general synthetic data using **CTGAN** for the entire dataset.
- Creates anomaly-specific synthetic data by focusing on rows labeled as anomalies.
- Combines original and synthetic data for enhanced datasets.

### Step 3: Time-Series Data Generation with TimeGAN
- Trains a TimeGAN model on time-series data.
- Generates synthetic sequences while preserving temporal dependencies.
- Visualizes and compares the distributions of original and synthetic datasets.

---

## Key Files

### Scripts
- **Autoencoder Anomaly Detection**:
  - `AutoencoderAnomalyDetector`: Encapsulates preprocessing, training, and anomaly detection.
- **Synthetic Data Generation with CTGAN**:
  - `DataSynthesizer`: Automates metadata creation, model training, and synthetic data generation.
- **Time-Series Generation with TimeGAN**:
  - `TimeGAN`: Generates realistic synthetic sequences with temporal patterns.

### Outputs
1. **Anomaly Detection Results**:
   - Reconstruction error plots.
   - Bar plots showing the number of normal and anomalous samples.
2. **Synthetic Data**:
   - General synthetic data: `anomaly_detection.csv`.
   - Anomaly-specific synthetic data: `anomaly_separate_dataset.csv`.
   - Combined dataset: `anomaly_combined_dataset.csv`.
3. **TimeGAN Outputs**:
   - Generated synthetic time-series data: `synthetic_data.csv`.
   - Side-by-side histogram comparisons for each feature.

---

## How to Run

### Step 1: Anomaly Detection with Autoencoder
1. Load the dataset:
   ```python
   data = pd.read_csv('network_dataset_labeled.csv')
   ```
2. Train the Autoencoder:
   ```python
   anomaly_detector = AutoencoderAnomalyDetector(encoding_dim=10, epochs=50, batch_size=64)
   anomaly_detector.train_autoencoder()
   ```
3. Detect anomalies:
   ```python
   mse = anomaly_detector.predict_reconstruction_error()
   anomalies = anomaly_detector.detect_anomalies(mse, percentile=95)
   ```
4. Visualize results:
   ```python
   anomaly_detector.visualize_anomalies(mse)
   ```

### Step 2: Synthetic Data Generation with CTGAN
1. Initialize and train the synthesizer:
   ```python
   synthesizer = DataSynthesizer(data=df1)
   synthesizer.fit()
   ```
2. Generate synthetic data:
   ```python
   synthetic_data = synthesizer.generate_synthetic_data()
   synthetic_data.to_csv('anomaly_detection.csv', index=False)
   ```

### Step 3: Time-Series Data Generation with TimeGAN
1. Preprocess the dataset:
   ```python
   data_normalized = (data - data.mean()) / data.std()
   ```
2. Train TimeGAN:
   ```python
   gan = TimeGAN(seq_length=30, num_features=data_normalized.shape[1])
   gan.compile_models()
   gan.train(real_data=time_series_data, epochs=1000, batch_size=64)
   ```
3. Generate synthetic time-series data:
   ```python
   synthetic_data = gan.generate_synthetic_data(num_samples=1000)
   synthetic_data.to_csv('synthetic_data.csv', index=False)
   ```

4. Visualize the data:
   ```python
   plot_side_by_side_histograms(original_data, synthetic_data, selected_features)
   ```

---

## Example Visualizations

1. **Reconstruction Error Histogram**:
   ![Histogram](path/to/histogram.png)

2. **Side-by-Side Histogram**:
   ![Side-by-Side Histogram](path/to/side_by_side_histogram.png)

3. **Time-Series Comparison**:
   ![Time-Series Plot](path/to/time_series_plot.png)

---

## Future Enhancements

- Integrate additional models (e.g., VAEs, GANs for non-tabular data).
- Expand support for irregularly sampled time-series data.
- Incorporate privacy-preserving techniques like differential privacy.

---
