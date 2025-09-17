# Import librerie
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Caricamento dataset
df = pd.read_csv("train.csv")

# X = tutti i pixel, y = label
X = df.drop("label", axis=1)
y = df["label"]

print("Shape X:", X.shape)
print("Distribuzione classi:\n", y.value_counts())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 
pca = PCA().fit(X_scaled)

# Calcolo varianza cumulativa
varianza_cum = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(varianza_cum)
plt.axhline(y=0.95, color='r', linestyle='--', label="95% soglia")
plt.xlabel("Numero componenti PCA")
plt.ylabel("Varianza cumulativa")
plt.title("Varianza spiegata dalle componenti PCA")
plt.legend()
plt.grid(True)
plt.show()

# Numero di componenti per spiegare il 95%
n_components = np.argmax(varianza_cum >= 0.95) + 1
print(f"Numero di componenti per spiegare il 95% della varianza: {n_components}")

# Trasformazione PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)


# Train/test split 
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, stratify=y, random_state=42
)

X_train_raw, X_test_raw, _, _ = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)


# Decision Tree con PCA
dt_pca = DecisionTreeClassifier(random_state=42)
dt_pca.fit(X_train_pca, y_train)

y_pred_pca = dt_pca.predict(X_test_pca)

acc_pca = accuracy_score(y_test, y_pred_pca)
print(f"\nAccuracy con PCA: {acc_pca:.4f}")

cm_pca = confusion_matrix(y_test, y_pred_pca)
ConfusionMatrixDisplay(cm_pca, display_labels=range(10)).plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Matrice di confusione - PCA + Decision Tree")
plt.show()

# Decision Tree senza PCA
dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train_raw, y_train)

y_pred_raw = dt_raw.predict(X_test_raw)

acc_raw = accuracy_score(y_test, y_pred_raw)
print(f"\nAccuracy senza PCA: {acc_raw:.4f}")

cm_raw = confusion_matrix(y_test, y_pred_raw)
ConfusionMatrixDisplay(cm_raw, display_labels=range(10)).plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Matrice di confusione - Decision Tree (senza PCA)")
plt.show()

# Analisi finale (confronto con e senza PCA)
print("\n=== Analisi ===")
print(f"Accuracy con PCA: {acc_pca:.4f}")
print(f"Accuracy senza PCA: {acc_raw:.4f}")

##################################################################################
# Con la riduzione dimensionale tramite PCA (per il 95% varianza sono circa 320 componenti) 
# il modello ottiene un’accuracy di circa 0.80, inferiore a quella del Decision Tree allenato sui dati originali (0.85). 
# Questo è un risultato ci mostra come la PCA riduce la dimensionalità comprimendo l’informazione, 
# e quindi può sacrificare parte della capacità predittiva.

# Senza PCA, l’albero sfrutta tutte le 784 feature e raggiunge una accuracy leggermente migliore, 
# ma a costo di un modello molto più complesso, con rischio di overfitting sui dati di training.

# Il modello impiega più tempo ad allenarsi nel caso con PCA, 
# questo perchè con PCA si riduce la dimensione ma non la qualità dell’informazione. 
# La qualità del dataset è talmente ampia da non risentire molto della riduzione dimensionale.
#################################################################################

#EXTRA: (grafico 3D PCA)
from mpl_toolkits.mplot3d import Axes3D 

# PCA con 3 componenti
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

print("Varianza spiegata dalle 3 componenti:", pca_3d.explained_variance_ratio_)
print("Totale varianza spiegata:", pca_3d.explained_variance_ratio_.sum())

# Grafico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=y,  # le etichette delle cifre (0-9) per colorare i punti
    cmap="tab10",
    s=10,
    alpha=0.6
)

ax.set_title("MNIST - PCA 3D (prime 3 componenti)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

# legenda con le cifre
legend1 = ax.legend(*scatter.legend_elements(), title="Cifre")
ax.add_artist(legend1)

plt.show()
