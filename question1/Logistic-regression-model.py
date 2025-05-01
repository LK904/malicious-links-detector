import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sklearn.preprocessing import StandardScaler
from extract import extract_features



df = pd.read_csv("../data/malicious_phish.csv")
assert 'url' in df.columns and 'type' in df.columns, "CSV must have 'url' and 'label' columns"

feature_list = [extract_features(u) for u in df['url']]
features_df = pd.DataFrame(feature_list)

#encode type to binary classification
df['type'] = df['type'].apply(lambda x: 'benign' if x.lower() == 'benign' else 'malicious')

labels = df['type'].apply(lambda x: 1 if str(x).lower() in ['malicious', '1'] else 0)

#split data
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()

#scale model
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train model
model = LogisticRegression( max_iter=2000, class_weight= 'balanced')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
