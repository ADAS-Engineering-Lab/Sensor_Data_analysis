
# Exploratory Data Analysis (EDA) Script
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
data_path = '/content/drive/My Drive/ADAS-Engineering-Lab/Sensor_Data_Analysis_Repo/data/signal_extractions/preprocessed_CAN_data.csv'
df = pd.read_csv(data_path)

# Time-Series Visualization
def plot_time_series(data, signal_column, label_column="Label", save_path=None):
    plt.figure(figsize=(12, 5))
    normal_data = data[data[label_column] == 0]
    attack_data = data[data[label_column] == 1]
    plt.plot(normal_data["Time"], normal_data[signal_column], label="Normal", alpha=0.5)
    plt.plot(attack_data["Time"], attack_data[signal_column], label="Attack", alpha=0.8, linestyle="--")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# PCA for Dimensionality Reduction
pca = PCA(n_components=2)
signal_features = [col for col in df.columns if "Signal" in col]
pca_results = pca.fit_transform(df[signal_features])
print(f"✅ PCA results computed.")
