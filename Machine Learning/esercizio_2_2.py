# Import librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Caricamento dataset
df = pd.read_csv("Wholesale customers data.csv")

# Teniamo solo le colonne di spesa (escludiamo Channel e Region)
X = df.drop(columns=["Channel", "Region"])

print("Prime righe del dataset:")
print(X.head())


# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-distance plot (per stimare eps)
from sklearn.neighbors import NearestNeighbors

# Lista di valori di min_samples che voglio testare
min_samples_list = [3, 5, 8]

for ms in min_samples_list:
    neighbors = NearestNeighbors(n_neighbors=ms)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    # ordino le distanze del k-esimo vicino
    distances = np.sort(distances[:, -1])

    # grafico
    plt.figure(figsize=(8, 5))
    plt.plot(distances)
    plt.title(f"K-distance plot (k={ms})")
    plt.xlabel("Punti ordinati")
    plt.ylabel("Distanza al k-esimo vicino")
    plt.grid(True)
    plt.show()

#testo diversi valori di eps e min_samples
eps_values = [0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]
min_samples_values = [3, 5, 8]

for ms in min_samples_values:
    print(f"\n--- min_samples = {ms} ---")
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=ms)
        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = list(labels).count(-1)

        print(f"eps={eps} -> cluster: {n_clusters}, outlier: {n_outliers}")

#scelgo eps pari a 0.5 e min_samples pari a 10
eps=1.5
min_samples=3

# Alleno DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

df["Cluster"] = labels  # -1 = outlier


# Valutazione
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)

print(f"\nNumero di cluster trovati: {n_clusters}")
print(f"Numero di outlier: {n_outliers}")

if n_clusters > 1 and n_clusters < len(X):
    score = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {score:.3f}")
else:
    print("Silhouette Score non calcolabile (serve almeno 2 cluster validi).")


# Visualizzazione (riduzione in 2D con PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels,
    palette="tab10",
    s=60,
    alpha=0.8
)
plt.title(f"DBSCAN clustering (eps={eps}, min_samples={min_samples})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster", loc="best")
plt.show()


# Interpretazione 
for c in set(labels):
    subset = df[df["Cluster"] == c]
    if c == -1:
        print(f"\nCluster {c} → OUTLIER (N={len(subset)})")
    else:
        print(f"\nCluster {c} - N clienti: {len(subset)}")
        print(subset.drop(columns=["Cluster"]).mean())


#####################################################################
# Cluster 0:
# circa 413 clienti
# spesa bilanciata in tutte le categorie (Fresh ~11k, Milk ~4.7k, Grocery ~6.6k)

# Cluster 1:
# solo 3 clienti
# spesa altissima su Milk (~22.8k), Grocery (~22.4k) e Detergents (~8.2k) 

# Cluster 2:
# 3 clienti
# spesa elevata su Delicatessen (~4.3k vs media ~1.2k)

# Cluster -1:
# 21 clienti atipici (outlier), profili estremi (molto grandi o molto piccoli)
# DBSCAN li considera “rumore”, ma possono rappresentare casi speciali da analizzare a parte 
####################################################################
