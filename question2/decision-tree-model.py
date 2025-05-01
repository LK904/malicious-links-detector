import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from extract import extract_features


df = pd.read_csv("../data/malicious_phish.csv")
df = df[df["type"] != "benign"]

label_encoder = LabelEncoder()
df["type"] = label_encoder.fit_transform(df["type"])


feature_list = [extract_features(u) for u in df['url']]

X = pd.DataFrame(feature_list)
y = df["type"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)

target_names = label_encoder.classes_
print(classification_report(y_test, y_pred, target_names=target_names))
