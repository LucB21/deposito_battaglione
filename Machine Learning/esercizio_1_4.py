# Import librerie
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Caricamento dataset
df = pd.read_csv("iris.csv")

# Prime 5 righe
print(df.head())

# Separazione feature e target
X = df.drop("target", axis=1)   # tutte le colonne tranne 'target'
y = df["target"]

# K-Fold Stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Modello Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)

# Cross-validation (Accuracy)
scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

print("Risultati Cross-Validation")
print(f"Accuracy media: {scores.mean():.3f}")
print(f"Deviazione standard: {scores.std():.3f}")

# Report finale su tutto il dataset
model.fit(X, y)
y_pred = model.predict(X)
print("Classification Report")
print(classification_report(y, y_pred))
