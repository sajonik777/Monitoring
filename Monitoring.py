import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load data
df = pd.read_csv('/checkpoint-107-2.txt', delimiter='\t')
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

# Preprocessing
df_numeric = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df_numeric)

# Feature Importance
pca = PCA()
df_pca = pca.fit_transform(df_numeric_scaled)
explained_variance = pca.explained_variance_ratio_

# Anomaly Detection
model = IsolationForest(contamination=0.01)
model.fit(df_numeric_scaled)
df['anomaly'] = model.predict(df_numeric_scaled)

# Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='server name', y='status', hue='anomaly')
plt.title('Anomalies in Status')
plt.show()

# Severity Classification
df['severity'] = df['anomaly'].apply(lambda x: 'High' if x == -1 else 'Low')

# Display anomalies
print(df[df['anomaly'] == -1])
