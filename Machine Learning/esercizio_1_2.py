# Import librerie
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Caricamento dataset
df = pd.read_csv("creditcard.csv")

# visualizzazione prime righe del dataset
print(df.head())

# Separazione feature e target
X = df.drop("Class", axis=1)   # tutte le colonne tranne "Class"
y = df["Class"]                # target = Class

# Divisione in training e test set (80% - 20%) 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

print("Distribuzione classi nel train:", Counter(y_train))
print("Distribuzione classi nel test :", Counter(y_test))


# Decision Tree
dt = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
print("Decision Tree")
print(classification_report(y_test, y_pred_dt, digits=4))


# Random Forest
rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest")
print(classification_report(y_test, y_pred_rf, digits=4))


# SMOTE sul train
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Distribuzione classi post-SMOTE:", Counter(y_train_sm))


# Decision Tree con SMOTE
dt_sm = DecisionTreeClassifier(class_weight="balanced", random_state=42)
dt_sm.fit(X_train_sm, y_train_sm)

y_pred_dt_sm = dt_sm.predict(X_test)
print("Decision Tree con SMOTE")
print(classification_report(y_test, y_pred_dt_sm, digits=4))


# Random Forest con SMOTE
rf_sm = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
rf_sm.fit(X_train_sm, y_train_sm)

y_pred_rf_sm = rf_sm.predict(X_test)
print("Random Forestcon SMOTE")
print(classification_report(y_test, y_pred_rf_sm, digits=4))
