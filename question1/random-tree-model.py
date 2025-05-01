import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from extract import extract_features




# 2. Load and preprocess your data
df = pd.read_csv("../data/malicious_phish.csv")  # Replace with your actual filename

# Binary classification: label 1 for malicious, 0 for benign
df["type"] = df["type"].apply(lambda x: 0 if x.lower() == "benign" else 1)

# Extract features
feature_rows = [extract_features(url) for url in df["url"]]
X = pd.DataFrame(feature_rows)
y = df["type"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# 4. Train the random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))