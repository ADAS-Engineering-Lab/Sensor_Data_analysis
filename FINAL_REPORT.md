
# 🚗 Sensor Data Analysis Project: Final Report

## 📌 Project Overview
The goal of this project was to analyze CAN bus datasets for anomaly detection and generate insights to improve Advanced Driver Assistance Systems (ADAS). The project included steps for preprocessing, exploratory data analysis (EDA), machine learning model training, and results visualization.

---

## 🛠️ Steps Followed
### Step 1: Data Preprocessing
- Handled missing values via mean imputation.
- Normalized signal features to standardize the dataset.
- Saved preprocessed dataset for analysis.

### Step 2: Exploratory Data Analysis (EDA)
- Time-series trends analyzed for normal vs. attack data.
- PCA applied for dimensionality reduction.
- Fourier Transform (FFT) performed to identify frequency patterns.
- Signal energy computed to distinguish between normal and attack behavior.

### Step 3: Machine Learning Models
- **Unsupervised Models**:
  - Isolation Forest achieved AUC: 0.4980.
  - One-Class SVM achieved AUC: 0.4871.
- **Supervised Models**:
  - Random Forest achieved accuracy: 92%.
  - LSTM trained for time-series anomaly detection (accuracy: 92.3%).

### Step 4: Data Segmentation & Characterization
- Sliding window approach segmented data into 199 windows for local pattern analysis.
- K-Means and Hierarchical Clustering were applied to characterize attack vs. normal behavior.
- Signal energy variations were analyzed.

---

## ✅ Key Outcomes
- **Preprocessed Dataset**: `/data/signal_extractions/preprocessed_CAN_data.csv`
- **EDA Results**:
  - PCA Results: `/reports/pca_results.csv`
  - FFT Analysis: `/reports/fft_analysis.csv`
  - Signal Energy Statistics: `/reports/signal_energy_stats.csv`
- **Machine Learning Models**:
  - Random Forest: `/models/random_forest_sample.pkl`
  - LSTM: `/models/lstm_model_sample.h5`
- **Clustering Results**: `/reports/clustering_results.csv`

---

## 📊 Visualizations
- Time-series trends: `/visualizations/time_series_signal_1.png`
- FFT Results: `/visualizations/fft_analysis_signal_1.png`
- Signal energy distribution: `/visualizations/signal_energy_distribution.png`
- Confusion Matrix: `/visualizations/confusion_matrix.png`

---

## 📝 Observations
- Signal energy distributions showed distinct variations for attack signals.
- Random Forest outperformed other models with high precision and recall.
- Clustering techniques provided valuable insights into data segmentation and characterization.

---

## 🚀 Next Steps
- Integrate these results into a larger ADAS framework.
- Test models on real-time CAN bus data.

---

## 📂 Repository Structure
/data/ # Raw and preprocessed datasets /models/ # Trained machine learning models /reports/ # Analytical and performance reports /scripts/ # Scripts for preprocessing, EDA, and modeling /visualizations/ # Saved plots and visualizations


## 👨‍💻 Author
- **Your Name**
- [GitHub Profile](https://github.com/YourProfile)
