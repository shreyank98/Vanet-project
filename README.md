# AI-based Intrusion Detection System for VANET

## Project Overview

This project implements a **Deep Learning-based Intrusion Detection System (IDS)** for **Vehicular Ad-hoc Networks (VANET)** with a focus on detecting three critical attack types:
- **DoS (Denial of Service) Attacks**
- **Sybil Attacks**
- **Wormhole Attacks**

The system uses state-of-the-art deep learning models including CNN-LSTM hybrid networks and Autoencoders for real-time attack detection and classification.

---

## 🎯 Key Features

- **Multi-Model Architecture**: CNN-LSTM hybrid for classification + Autoencoder for anomaly detection
- **Real-time Detection**: Fast inference for real-time VANET security
- **Synthetic Attack Generation**: Generate synthetic VANET attack scenarios using statistical methods
- **Comprehensive Evaluation**: Detailed metrics including precision, recall, F1-score, and confusion matrices
- **Attack-Specific Detectors**: Specialized modules for DoS, Sybil, and Wormhole attacks
- **Extensible Design**: Easy to add new attack types and models

---

## 📊 Dataset Information

### Primary Dataset: CICIDS2017
- **Source**: Canadian Institute for Cybersecurity
- **Contains**: DoS, DDoS, Port Scan, and other network attacks
- **Size**: ~2.5GB (DoS/DDoS subset)
- **Download**: [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)

### Secondary Dataset: Network Dataset (Included)
- **File**: `network_dataset_labeled.csv`
- **Records**: 1,002 samples
- **Features**: 19 (throughput, congestion, packet_loss, latency, jitter, etc.)
- **Labels**: Binary anomaly labels

### Synthetic Data Generation
The project includes a synthetic data generator that creates realistic VANET attack scenarios based on known attack characteristics.

---

## 🏗️ Project Structure

```
Vanet-project/
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Preprocessed data
│   └── synthetic/                 # Synthetic attack data
│
├── models/                        # Deep learning models
│   ├── cnn_lstm_ids.py           # CNN-LSTM hybrid model
│   ├── autoencoder_ids.py        # Autoencoder for anomaly detection
│   └── __init__.py
│
├── detection/                     # Attack-specific detectors
│   ├── dos_detector.py           # DoS attack detector
│   ├── sybil_detector.py         # Sybil attack detector
│   ├── wormhole_detector.py      # Wormhole attack detector
│   └── __init__.py
│
├── preprocessing/                 # Data preprocessing
│   ├── data_loader.py            # Dataset loading and preprocessing
│   ├── synthetic_generator.py    # Synthetic attack generation
│   └── __init__.py
│
├── evaluation/                    # Evaluation metrics
│   ├── metrics.py                # Performance metrics
│   └── __init__.py
│
├── utils/                         # Utilities
│   ├── config.py                 # Configuration management
│   ├── logger.py                 # Logging utilities
│   └── __init__.py
│
├── saved_models/                  # Trained model checkpoints
├── results/                       # Evaluation results
├── logs/                          # Training logs
│
├── train_model.py                # Main training script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PROJECT_PLAN.md               # Detailed project plan
└── .gitignore                    # Git ignore file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/shreyank98/Vanet-project.git
cd Vanet-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create data directories**
```bash
mkdir -p data/raw data/processed data/synthetic saved_models results logs
```

4. **Place datasets**
- Copy `network_dataset_labeled.csv` to `data/raw/`
- (Optional) Download CICIDS2017 to `data/raw/CICIDS2017/`

---

## 📖 Usage

### Quick Start: Train with Synthetic Data

The easiest way to get started is to train on synthetic data:

```bash
python train_model.py --generate_synthetic --samples_per_class 1000 --model both
```

This will:
- Generate 4,000 synthetic samples (1,000 per class: NORMAL, DOS, SYBIL, WORMHOLE)
- Train both CNN-LSTM and Autoencoder models
- Evaluate and save the models

### Training Options

#### Train CNN-LSTM model only
```bash
python train_model.py --model cnn_lstm --generate_synthetic
```

#### Train Autoencoder only
```bash
python train_model.py --model autoencoder --generate_synthetic
```

#### Custom training parameters
```bash
python train_model.py \
    --generate_synthetic \
    --samples_per_class 2000 \
    --epochs 50 \
    --batch_size 32 \
    --sequence_length 10 \
    --model both
```

#### Train with CICIDS2017 (if downloaded)
```bash
python train_model.py --use_cicids --model cnn_lstm
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | 'cnn_lstm' | Model to train: 'cnn_lstm', 'autoencoder', or 'both' |
| `--use_cicids` | flag | False | Use CICIDS2017 dataset |
| `--generate_synthetic` | flag | False | Generate synthetic attack data |
| `--samples_per_class` | int | 1000 | Number of synthetic samples per class |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 64 | Batch size for training |
| `--sequence_length` | int | 10 | Sequence length for LSTM |

---

## 🔬 Model Architectures

### 1. CNN-LSTM Hybrid Model

**Purpose**: Multi-class attack classification

**Architecture**:
- **CNN Layers**: 3 layers (64, 128, 256 filters) for spatial feature extraction
- **LSTM Layers**: 2 layers (128, 64 units) for temporal pattern recognition
- **Dense Layers**: 2 layers (128, 64 units) for classification
- **Output**: Softmax (4 classes: NORMAL, DOS, SYBIL, WORMHOLE)

**Input Shape**: (sequence_length, num_features)
**Output**: Class probabilities

### 2. Autoencoder Model

**Purpose**: Anomaly detection

**Architecture**:
- **Encoder**: [input_dim → 128 → 64 → 32]
- **Decoder**: [32 → 64 → 128 → input_dim]
- **Latent Dimension**: 32
- **Loss**: Mean Squared Error (MSE)

**Training**: Trained only on normal traffic
**Detection**: Reconstruction error threshold-based

---

## 📈 Performance Metrics

The system evaluates performance using:

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **ROC-AUC**: Area under ROC curve
- **Detection Rate**: Per-attack detection success rate
- **False Positive Rate**: Critical for IDS systems

---

## 🎯 Attack Detection

### DoS Attack Detection

**Indicators**:
- High packet loss (>50%)
- Extreme latency (>1000ms)
- High congestion (>0.8)
- Low throughput (<1.5)

**Module**: `detection/dos_detector.py`

### Sybil Attack Detection

**Indicators**:
- Abnormally high router count (>5)
- Multiple identities from single source
- Rapid identity changes
- Normal traffic with suspicious identity patterns

**Module**: `detection/sybil_detector.py`

### Wormhole Attack Detection

**Indicators**:
- Artificially low latency (<50ms)
- Low router count (<3)
- High throughput with low latency
- Route anomalies (planned vs actual)

**Module**: `detection/wormhole_detector.py`

---

## 📊 Example Results

Example output from training:

```
VANET IDS PERFORMANCE SUMMARY
============================================================
Overall Performance:
  Accuracy:        0.9524
  Precision (avg): 0.9531
  Recall (avg):    0.9524
  F1-Score (avg):  0.9525

Per-Attack Performance:
  NORMAL       - Precision: 0.9800, Recall: 0.9600, F1: 0.9699
  DOS          - Precision: 0.9500, Recall: 0.9700, F1: 0.9599
  SYBIL        - Precision: 0.9300, Recall: 0.9200, F1: 0.9250
  WORMHOLE     - Precision: 0.9400, Recall: 0.9600, F1: 0.9499
============================================================
```

---

## 🔧 Configuration

Modify `utils/config.py` to customize:

- Model hyperparameters (filters, units, dropout)
- Training parameters (epochs, batch size, learning rate)
- Data paths and directories
- Attack type definitions
- Detection thresholds

---

## 📝 Synthetic Data Generation

Generate synthetic VANET attack data:

```python
from preprocessing.synthetic_generator import VANETAttackGenerator

generator = VANETAttackGenerator()

# Generate balanced dataset
dataset = generator.generate_balanced_dataset(samples_per_class=1000)

# Generate specific attack types
dos_data = generator.generate_dos_attack(n_samples=500)
sybil_data = generator.generate_sybil_attack(n_samples=500)
wormhole_data = generator.generate_wormhole_attack(n_samples=500)

# Save to file
generator.save_synthetic_data(dataset, 'my_dataset.csv')
```

---

## 🧪 Testing and Evaluation

After training, models are automatically evaluated on the test set. To manually evaluate:

```python
from models.cnn_lstm_ids import CNNLSTM_IDS
from evaluation.metrics import IDSMetrics

# Load trained model
model = CNNLSTM_IDS(input_shape=(10, 17), num_classes=4)
model.load_model()

# Make predictions
y_pred = model.predict_classes(X_test)

# Calculate metrics
metrics = IDSMetrics()
results = metrics.calculate_all_metrics(y_test, y_pred)
metrics.summarize_performance(results)
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional attack types (Blackhole, Greyhole, etc.)
- Real VANET dataset integration
- Model optimization and compression
- Real-time detection interface
- Visualization dashboard

---

## 📚 References

### Datasets
1. **CICIDS2017**: [Canadian Institute for Cybersecurity IDS Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
2. **CY0P5_ML_Datasets**: [Comprehensive ML Security Datasets](https://github.com/ctinnil/CY0P5_ML_Datasets)

### Research Papers
1. Hasrouny et al. (2017). "VANET security challenges and solutions: A survey"
2. Sedjelmaci et al. (2018). "An accurate and efficient collaborative intrusion detection framework for VANET"
3. Bangui et al. (2019). "Recent advances in machine-learning driven intrusion detection in VANET"

### Technologies
- **TensorFlow/Keras**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **pandas/numpy**: Data processing
- **SDV**: Synthetic data generation

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👤 Author

**Project**: VANET Security - AI-based Intrusion Detection System  
**Topic**: Security, Privacy, and Trust Management  
**Focus**: Deep Learning for DoS, Sybil, and Wormhole Attack Detection

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: TensorFlow/CUDA errors
```bash
# Use CPU version if GPU unavailable
pip install tensorflow-cpu
```

**Issue**: Out of memory during training
```bash
# Reduce batch size
python train_model.py --batch_size 16
```

**Issue**: Slow training
```bash
# Reduce epochs or use smaller dataset
python train_model.py --epochs 20 --samples_per_class 500
```

---

## 📞 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check PROJECT_PLAN.md for detailed methodology
- Review existing issues for solutions

---

## 🎓 Academic Use

If you use this project in your research, please consider citing:

```
@misc{vanet_ids_2024,
  title={AI-based Intrusion Detection System for VANET},
  author={VANET Project Team},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/shreyank98/Vanet-project}}
}
```

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Status**: Active Development
