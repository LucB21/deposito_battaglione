# Import librerie
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# Caricamento dataset
df = pd.read_csv("Mall_Customers.csv")

# Teniamo solo Annual Income e Spending Score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print("Prime righe del dataset:")
print(X.head())

#  Standardizzazione delle features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Scelta k con metodo elbow 
inertias = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=inertias, marker="o")
plt.title("Metodo del gomito (Elbow)")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# Scelta k con Silhouette score
silhouette_scores = []
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=silhouette_scores, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Stampa silhouette score per scegliere il k
for k, score in zip(K, silhouette_scores):
    print(f"k={k}: silhouette={score:.3f}")

# Scelta del best k=5 
best_k = 5


# Addestramento KMeans 
kmeans_final = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)

df["Cluster"] = labels

# Calcolo centroidi in scala originale
centroids = scaler.inverse_transform(kmeans_final.cluster_centers_)

# Visualizzazione cluster con centroidi
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Annual Income (k$)", 
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    data=df,
    s=60,
    alpha=0.8
)

# Aggiunta centroidi marcati 
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=250, c="red", marker="X", label="Centroidi"
)

plt.title(f"Clustering clienti con KMeans (k={best_k})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()

# Interpretazione
for c in range(best_k):
    subset = df[df["Cluster"] == c]
    print(f"\nCluster {c} - N clienti: {len(subset)}")
    print(subset[["Annual Income (k$)", "Spending Score (1-100)"]].mean())


#########################################################################
#cluster 0: Medio reddito, medio spending → clienti medi

#Cluster 1: Alto reddito, alto spending → clienti da fidelizzare

#Cluster 2: Basso reddito, alto spending → clienti che spendono molto più di quanto guadagnao (prendono in prestito)

#Cluster 3: Alto reddito, basso spending → clienti risparmiatori

#Cluster 4: Basso  reddito, basso spending → clienti poco rilevanti
##########################################################################
