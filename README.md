# Sensor Data Analysis

This repository contains the implementation of CAN bus anomaly detection.

## 🚀 Data Preprocessing Report

### 🔹 Step 1: Handle Missing Data
- **Total Missing Values Before Preprocessing:** 436031147
- **Total Missing Values After Preprocessing:** 5263
- **Action Taken:**
  - Interpolated missing values linearly.
  - Removed rows with more than 50% missing data.

### 🔹 Step 2: Normalize Signals
- **Normalization Method:** Min-Max Scaling
- **Columns Normalized:** 22

✅ **Preprocessed dataset is now ready for EDA.**


## 🚀 Exploratory Data Analysis (EDA) Report

### 1️⃣ Time-Series Signal Visualization
- Plots created for **normal vs. attack signals** over time.
- **Saved to:** `/visualizations/`

### 2️⃣ Feature Engineering
- Extracted **statistical features** (mean, variance, etc.).
- Applied **PCA** to reduce signal dimensions.
- **Saved to:** `/reports/feature_statistics.csv` & `/reports/pca_results.csv`

### 3️⃣ Attack Pattern Identification
- **FFT Analysis:** Extracted frequency patterns.
- **Signal Energy Computation:** Used to differentiate attacks.
- **Saved to:** `/reports/fft_analysis.csv` & `/reports/signal_energy.csv`

### 4️⃣ Label Imbalance Handling
- **SMOTE applied** to balance attack vs. normal class distribution.
- **Saved balanced dataset to:** `/reports/balanced_dataset.csv`

### 5️⃣ Metadata Analysis
- Extracted **attack-specific details**.
- **Saved to:** `/reports/metadata_analysis.csv`

✅ **EDA process completed successfully!**  


## 🚀 Machine Learning Models for Anomaly Detection (Sample Dataset)

### 1️⃣ Unsupervised Anomaly Detection
- **Isolation Forest AUC:** 0.4980
- **One-Class SVM AUC:** 0.4871
- **Results saved to:** `/reports/unsupervised_results_sample.csv`

### 2️⃣ Supervised Attack Classification
- **Random Forest AUC:** 0.6864
- **Model saved to:** `/models/random_forest_sample.pkl`

### 3️⃣ Time-Series Modeling
- **LSTM model:** Trained on sequence data.
- **Model saved to:** `/models/lstm_model_sample.h5`

✅ **All models and results have been saved successfully.**


## 🔍 Key Observations from Machine Learning Models

1️⃣ **Unsupervised Anomaly Detection**:
- Both Isolation Forest and One-Class SVM struggled to detect anomalies.
- **AUC Scores**:
  - Isolation Forest: **0.4980**
  - One-Class SVM: **0.4871**
- This indicates weak anomaly patterns in the sampled dataset.

2️⃣ **Supervised Attack Classification**:
- **Random Forest Performance**:
  - **AUC**: **0.6864**
  - Precision and Recall for the attack class (label `1`) were relatively low, suggesting label imbalance.
  - Precision (Attack): **0.48**, Recall (Attack): **0.41**.

3️⃣ **Time-Series Modeling**:
- The LSTM model performed well on sequence-based data:
  - **Training Accuracy**: ~91.50%
  - **Validation Accuracy**: ~92.30% (after 5 epochs).

### Suggestions for Improvement:
- Investigate dataset for stronger anomaly patterns or feature engineering.
- Address class imbalance using techniques like **SMOTE** for oversampling.
- Explore more advanced models (e.g., Gradient Boosting, Transformer-based models).


## 🚀 Step 5: Data Segmentation & Attack Characterization

### 1️⃣ Sliding Window Approach
- Data segmented into **fixed-size windows** for local pattern analysis.
- **Window size**: 100, **Step size**: 50.
- **Segments created**: 199.
- Segmented data saved to: `/reports/segmented_data.npy`.

### 2️⃣ Clustering Techniques
- **K-Means Clustering** and **Hierarchical Clustering** applied to detect clusters.
- Results saved to: `/reports/clustering_results.csv`.

### 3️⃣ Signal Energy Quantification
- Computed signal energy for each row.
- **Energy Statistics by Label**:
  - Results saved to: `/reports/signal_energy_stats.csv`.
- Signal energy distribution plot saved to: `/visualizations/signal_energy_distribution.png`.

✅ **Insights from attack behavior have been saved successfully!**


## 🚀 Step 6: Results Visualization & Reporting

### Model Performance Metrics
- **Random Forest**:
  - Accuracy: 0.92
  - Precision: 0.95
  - Recall: 0.96
  - F1-Score: 0.96
- **LSTM**:
  - Accuracy: 0.923

### Confusion Matrix
- Confusion Matrix for Random Forest saved to `/visualizations/confusion_matrix.png`.

✅ Results and visualizations saved successfully!
