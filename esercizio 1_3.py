# Import librerie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Caricamento dataset
df = pd.read_csv("iris.csv")

# Prime 5 righe
print(df.head())

# Separazione feature e target
X = df.drop("target", axis=1)   # tutte le colonne tranne 'target'
y = df["target"]


# Divisione in Train, Validation e Test

# Prima separiamo il test set (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Dal resto, estraiamo il validation (15% del totale ≈ 0.176 del residuo)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print("Dimensioni train:", X_train.shape)
print("Dimensioni validation:", X_val.shape)
print("Dimensioni test:", X_test.shape)


# Training del modello Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


# Valutazione su Validation set
y_val_pred = model.predict(X_val)
print("VALIDATION")
print("Accuracy (val):", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# Valutazione finale su Test set
y_test_pred = model.predict(X_test)
print("TEST")
print("Accuracy (test):", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
