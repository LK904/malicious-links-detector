import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from extract import extract_features
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class URLClassifier(nn.Module):
    def __init__(self, input_dim):
        super(URLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 3)  # 3 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

df = pd.read_csv("../data/malicious_phish.csv")
df = df[df["type"] != "benign"]

label_encoder = LabelEncoder()
df["type"] = label_encoder.fit_transform(df["type"])


feature_list = [extract_features(u) for u in df['url']]
features_df = pd.DataFrame(feature_list)

X = np.array(features_df).astype(float)
y = df["type"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = URLDataset(X_train, y_train)
test_ds = URLDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = URLClassifier(X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


print(torch.cuda.is_available())
epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")


y_true = []
y_pred = []
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        logits = model(batch_X)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch_y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)


print(f"Test Accuracy: {correct / total:.4f}")
print(classification_report(y_pred, y_test, target_names=label_encoder.classes_))
