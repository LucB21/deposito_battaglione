#Import librerie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#caricamento dataset
df = pd.read_csv('iris.csv')

#print prime 5 righe
print(df.head())

#separazione feature e target
X = df.drop('target', axis=1) #prende tutte le colonne tranne 'target'
y = df['target']

#divisione in training e test set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

#fit del modello Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

#predizione e valutazione
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Report delle performance
print(classification_report(y_test, y_pred))
