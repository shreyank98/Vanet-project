# AI-based Intrusion Detection System for VANET
## Project Topic: Security, Privacy, and Trust Management
## Subtopic: AI-based IDS - Deep Learning for Real-time Detection of DoS, Sybil, and Wormhole Attacks

---

## 1. REPOSITORY ANALYSIS

### 1.1 Current Repository (shreyank98/Vanet-project)
**Existing Capabilities:**
- **Synthetic Data Generation**: CTGAN and TimeGAN implementations
- **Anomaly Detection**: Autoencoder-based anomaly detection system
- **Dataset**: network_dataset_labeled.csv (1002 records)
  - Features: timestamp, throughput, congestion, packet_loss, latency, jitter, routers, routes, network targets, video metrics
  - Contains labeled anomalies (19 features including anomaly flags)
- **Technologies**: TensorFlow/Keras, pandas, scikit-learn, SDV

**Limitations:**
- Not specifically designed for VANET security attacks
- No specific DoS, Sybil, or Wormhole attack detection
- Limited real-world attack scenarios
- No multi-class classification for specific attack types

### 1.2 CY0P5_ML_Datasets Repository Analysis
**Available Datasets Relevant to VANET Security:**

| Dataset | Attack Types | Suitability | Pros | Cons |
|---------|-------------|--------------|------|------|
| **CICIDS2017** | DoS, DDoS, Port Scan, Brute Force | **HIGH** | Comprehensive, Well-labeled, Modern attacks | Large size, No VANET-specific |
| **NSL-KDD** | DoS, Probe, R2L, U2R | **MEDIUM** | Balanced, Classic benchmark | Older, Network-centric |
| **CSE-CIC-IDS2018** | DoS, DDoS, Botnet, Web attacks | **HIGH** | Recent, Diverse attacks | Very large, General network |
| **UNSW-NB15** | DoS, Exploits, Generic, Fuzzers | **MEDIUM** | Modern, Detailed features | Complex preprocessing |

**Key Finding:** No VANET-specific datasets in CY0P5_ML_Datasets repository. Best approach is to:
1. Use CICIDS2017 as base dataset (has comprehensive DoS attacks)
2. Augment with existing network_dataset_labeled.csv 
3. Create synthetic VANET-specific scenarios using existing GAN capabilities

---

## 2. RECOMMENDED DATASET SELECTION

### 2.1 Primary Dataset: **CICIDS2017**
**Justification:**
- Contains **comprehensive DoS/DDoS attacks** (DoS Hulk, DoS GoldenEye, DoS Slowloris, DDoS)
- Modern attack scenarios (2017)
- 80+ features including flow-based network metrics
- Well-documented and widely used in research
- Available in manageable CSV format

**Relevant Attack Types for VANET:**
1. **DoS/DDoS** - Direct mapping to VANET DoS attacks
2. **Port Scan** - Can simulate reconnaissance in VANET
3. **Botnet** - Similar to compromised vehicle nodes

**Dataset Structure:**
- Training files: ~2.5GB total
- Features: Source/Dest IP, Port, Protocol, Flow Duration, Packets, Bytes, Flags, etc.
- Labels: BENIGN, DoS GoldenEye, DoS Hulk, DoS Slowloris, DDoS, etc.

### 2.2 Secondary Dataset: **Existing network_dataset_labeled.csv**
**Use Case:**
- Represents VANET-like network metrics (throughput, congestion, latency, jitter)
- Can be used to create VANET-specific features
- Augment with synthetic data generation (TimeGAN/CTGAN)

### 2.3 Synthetic VANET Attack Generation Strategy
Since true VANET attack datasets are scarce, we'll create synthetic scenarios:

**DoS Attack Simulation:**
- High packet loss (>50%)
- Extreme latency spikes (>1000ms)
- Network congestion (>0.8)
- Throughput degradation

**Sybil Attack Simulation:**
- Multiple identities from same source
- Rapid identity changes
- Abnormal routing patterns
- Duplicate message patterns

**Wormhole Attack Simulation:**
- Artificially reduced latency between distant nodes
- Abnormal route hops
- Geographic impossibilities in timing
- Tunnel detection patterns

---

## 3. PROPOSED ARCHITECTURE

### 3.1 Multi-Model Deep Learning Architecture

```
┌─────────────────────────────────────────────────────┐
│           Data Preprocessing Pipeline                │
├─────────────────────────────────────────────────────┤
│  • Feature extraction from CICIDS2017                │
│  • VANET feature mapping                             │
│  • Normalization & encoding                          │
│  • Temporal sequence creation                        │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│         Deep Learning Models (Ensemble)              │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────────┐  ┌──────────────────┐        │
│  │   CNN-LSTM       │  │   Autoencoder    │        │
│  │   (Sequential)   │  │   (Anomaly Det)  │        │
│  └──────────────────┘  └──────────────────┘        │
│           │                      │                   │
│           │    ┌──────────────────┐                │
│           └───▶│  Attention      │                 │
│                │  Mechanism      │                 │
│                └──────────────────┘                │
│                         │                           │
└─────────────────────────┼───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│            Classification Layer                      │
├─────────────────────────────────────────────────────┤
│  • DoS Attack Detection                              │
│  • Sybil Attack Detection                            │
│  • Wormhole Attack Detection                         │
│  • Normal Traffic                                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 Model Components

**1. CNN-LSTM Hybrid**
- **CNN layers**: Extract spatial features from network traffic patterns
- **LSTM layers**: Capture temporal dependencies in attack sequences
- **Use case**: Primary classifier for attack type detection

**2. Autoencoder**
- **Architecture**: Deep autoencoder with bottleneck layer
- **Use case**: Anomaly detection and feature learning
- **Output**: Reconstruction error as anomaly score

**3. Attention Mechanism**
- **Type**: Self-attention layer
- **Use case**: Focus on most relevant features for attack detection
- **Benefit**: Interpretability of model decisions

**4. Ensemble Decision**
- Weighted voting from all models
- Confidence scoring
- Real-time prediction

---

## 4. IMPLEMENTATION PLAN

### Phase 1: Data Preparation (Priority: HIGH)
**Tasks:**
1. Download CICIDS2017 dataset (DoS/DDoS components)
2. Create data loader for CICIDS2017
3. Map CICIDS2017 features to VANET context
4. Augment with existing network_dataset_labeled.csv
5. Generate synthetic Sybil and Wormhole attack samples using TimeGAN
6. Create train/validation/test splits (70/15/15)

**Deliverables:**
- `data_loader.py` - Dataset loading and preprocessing
- `feature_engineering.py` - VANET feature mapping
- `synthetic_generator.py` - Attack scenario generation
- Processed datasets in `/data/` directory

### Phase 2: Model Development (Priority: HIGH)
**Tasks:**
1. Implement CNN-LSTM hybrid model
2. Implement Autoencoder for anomaly detection
3. Add attention mechanism
4. Create ensemble framework
5. Implement training pipeline with early stopping
6. Add model checkpointing

**Deliverables:**
- `models/cnn_lstm_ids.py` - Main classification model
- `models/autoencoder_ids.py` - Anomaly detection model
- `models/attention_layer.py` - Attention mechanism
- `models/ensemble_ids.py` - Ensemble framework
- `train_model.py` - Training script

### Phase 3: Attack-Specific Detection (Priority: HIGH)
**Tasks:**
1. Implement DoS detection module
2. Implement Sybil detection module
3. Implement Wormhole detection module
4. Create attack-specific feature extractors
5. Fine-tune models for each attack type

**Deliverables:**
- `detection/dos_detector.py`
- `detection/sybil_detector.py`
- `detection/wormhole_detector.py`
- Attack-specific metrics and evaluation

### Phase 4: Evaluation & Testing (Priority: MEDIUM)
**Tasks:**
1. Implement comprehensive metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
2. Create confusion matrix visualization
3. Perform cross-validation
4. Test real-time detection performance
5. Generate evaluation reports

**Deliverables:**
- `evaluation/metrics.py` - Evaluation metrics
- `evaluation/visualizations.py` - Result visualization
- `evaluate_model.py` - Evaluation script
- Performance reports and graphs

### Phase 5: Documentation & Deployment (Priority: MEDIUM)
**Tasks:**
1. Create comprehensive README.md
2. Add inline code documentation
3. Create usage examples
4. Write research methodology documentation
5. Create requirements.txt

**Deliverables:**
- Updated `README.md`
- `METHODOLOGY.md` - Research approach
- `examples/` - Usage examples
- `requirements.txt`
- API documentation

---

## 5. EXPECTED PROJECT STRUCTURE

```
Vanet-project/
├── data/
│   ├── raw/
│   │   ├── CICIDS2017/           # Downloaded CICIDS2017 files
│   │   └── network_dataset_labeled.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── synthetic/
│       ├── dos_synthetic.csv
│       ├── sybil_synthetic.csv
│       └── wormhole_synthetic.csv
│
├── models/
│   ├── __init__.py
│   ├── cnn_lstm_ids.py           # CNN-LSTM hybrid model
│   ├── autoencoder_ids.py        # Autoencoder for anomaly detection
│   ├── attention_layer.py        # Attention mechanism
│   └── ensemble_ids.py           # Ensemble framework
│
├── detection/
│   ├── __init__.py
│   ├── dos_detector.py           # DoS attack detector
│   ├── sybil_detector.py         # Sybil attack detector
│   └── wormhole_detector.py      # Wormhole attack detector
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading
│   ├── feature_engineering.py   # Feature extraction
│   └── synthetic_generator.py    # Synthetic data generation
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Evaluation metrics
│   └── visualizations.py         # Result visualization
│
├── utils/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   └── logger.py                 # Logging utilities
│
├── examples/
│   ├── train_example.py
│   ├── evaluate_example.py
│   └── real_time_detection.py
│
├── saved_models/                 # Trained model checkpoints
├── results/                      # Evaluation results
├── logs/                         # Training logs
│
├── train_model.py                # Main training script
├── evaluate_model.py             # Main evaluation script
├── predict.py                    # Real-time prediction
│
├── requirements.txt
├── README.md
├── METHODOLOGY.md
├── PROJECT_PLAN.md              # This file
└── .gitignore
```

---

## 6. TECHNICAL SPECIFICATIONS

### 6.1 Feature Engineering for VANET Context

**Network Metrics Mapping (CICIDS2017 → VANET):**
- Flow Duration → Session Duration
- Packet Length Stats → Message Size
- Flow Bytes/s → Throughput
- Flow Packets/s → Message Rate
- Idle Time → Channel Idle Time
- Active Time → Channel Active Time

**VANET-Specific Features (from network_dataset_labeled.csv):**
- Vehicle density (inferred from routers)
- Route stability (planned route vs actual)
- Congestion level
- Packet loss rate
- Latency & Jitter

**Synthetic Features for Attacks:**
- Identity entropy (Sybil detection)
- Route hop count anomaly (Wormhole)
- Resource consumption rate (DoS)

### 6.2 Model Hyperparameters

**CNN-LSTM Model:**
- CNN layers: 3 (filters: 64, 128, 256)
- LSTM layers: 2 (units: 128, 64)
- Dropout: 0.3
- Activation: ReLU (CNN), tanh (LSTM)
- Output: Softmax (4 classes)

**Autoencoder:**
- Encoder: [input_dim → 128 → 64 → 32]
- Decoder: [32 → 64 → 128 → input_dim]
- Latent dimension: 32
- Activation: ReLU
- Loss: MSE

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 100 (with early stopping)
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Precision, Recall, F1-Score

---

## 7. EXPECTED OUTCOMES

### 7.1 Performance Targets
- **Overall Accuracy**: >95%
- **DoS Detection**: Precision >96%, Recall >94%
- **Sybil Detection**: Precision >90%, Recall >88%
- **Wormhole Detection**: Precision >88%, Recall >85%
- **False Positive Rate**: <5%
- **Real-time Detection**: <100ms latency

### 7.2 Research Contributions
1. Novel adaptation of CICIDS2017 for VANET security research
2. Synthetic VANET attack generation methodology
3. Multi-model ensemble approach for VANET IDS
4. Attention-based interpretable attack detection
5. Open-source implementation for VANET security research

---

## 8. TIMELINE ESTIMATION

| Phase | Duration | Effort |
|-------|----------|--------|
| Phase 1: Data Preparation | 2-3 days | High |
| Phase 2: Model Development | 3-4 days | High |
| Phase 3: Attack Detection | 2-3 days | High |
| Phase 4: Evaluation | 1-2 days | Medium |
| Phase 5: Documentation | 1 day | Medium |
| **Total** | **9-13 days** | **High** |

---

## 9. RISKS AND MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| CICIDS2017 download size (2.5GB+) | High | Download only DoS/DDoS subsets |
| No real VANET attack data | High | Use synthetic generation + mapping |
| Model overfitting | Medium | Cross-validation, regularization |
| Real-time performance | Medium | Model optimization, pruning |
| Feature mismatch | Medium | Careful feature engineering |

---

## 10. CONCLUSION

This plan proposes a comprehensive AI-based Intrusion Detection System for VANET using:
- **CICIDS2017** as the primary dataset (DoS/DDoS attacks)
- **Existing network_dataset_labeled.csv** for VANET metrics
- **Synthetic data generation** for Sybil and Wormhole attacks
- **Deep learning ensemble** (CNN-LSTM + Autoencoder + Attention)

The approach balances:
✅ Using established cybersecurity datasets (CICIDS2017)
✅ Leveraging existing VANET project capabilities (GANs, Autoencoders)
✅ Creating VANET-specific attack scenarios synthetically
✅ Implementing state-of-the-art deep learning techniques
✅ Maintaining research reproducibility and extensibility

**Next Steps:**
1. Download CICIDS2017 dataset (DoS/DDoS components)
2. Implement data preprocessing pipeline
3. Develop deep learning models
4. Integrate attack detection modules
5. Evaluate and document results
