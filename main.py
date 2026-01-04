import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# =============================== Args ===============================
embedding_dim = int(sys.argv[1])

# =============================== Load data ===============================
data = pd.read_csv("recommend.csv")
data.columns = ["uid", "vid", "rate", "time"]
data["uid"] = data["uid"].astype("category").cat.codes
data["vid"] = data["vid"].astype("category").cat.codes

num_users = data["uid"].nunique()
num_items = data["vid"].nunique()

user_item_matrix = np.zeros((num_users, num_items), dtype=np.float32)
for _, row in data.iterrows():
    user_item_matrix[row.uid, row.vid] = row.rate

user_item_matrix /= 5.0

# =============================== Train / Val split ===============================
train_users, val_users = train_test_split(
    np.arange(num_users), test_size=0.2, random_state=42
)

X_train = user_item_matrix[train_users]
X_val = user_item_matrix[val_users]

train_tensor = torch.FloatTensor(X_train)
val_tensor = torch.FloatTensor(X_val)

train_loader = DataLoader(
    TensorDataset(train_tensor, train_tensor),
    batch_size=64,
    shuffle=True
)

# =============================== Model ===============================
class UserAutoEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_items)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

model = UserAutoEncoder(num_items, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# =============================== Loss ===============================
def weighted_mse(pred, target, alpha=2.0):
    weight = torch.ones_like(target)
    weight[target > 0] = alpha
    return torch.mean(weight * (pred - target) ** 2)

# =============================== Training w/ Early Stopping ===============================
epochs = 50
patience = 5
best_val_loss = float("inf")
patience_ctr = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = weighted_mse(recon, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        recon_val, _ = model(val_tensor)
        val_loss = weighted_mse(recon_val, val_tensor).item()

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_ctr = 0
        best_state = model.state_dict()
    else:
        patience_ctr += 1

    if patience_ctr >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)

# =============================== Extract embeddings ===============================
model.eval()
with torch.no_grad():
    _, user_embeddings = model(torch.FloatTensor(user_item_matrix))

user_embeddings = nn.functional.normalize(user_embeddings, dim=1).cpu().numpy()

# =============================== Item → liked users ===============================
RATING_THRESHOLD = 3.0
item_to_liked_users = (
    data[data.rate >= RATING_THRESHOLD]
    .groupby("vid")["uid"]
    .apply(np.array)
    .to_dict()
)

# =============================== Save EVERYTHING ===============================
save_payload = {
    "model_state": model.state_dict(),
    "user_embeddings": user_embeddings,
    "item_to_liked_users": item_to_liked_users,
    "num_items": num_items,
    "embedding_dim": embedding_dim,
    "best_val_loss": best_val_loss
}

joblib.dump(
    save_payload,
    f"recommendation_model_{embedding_dim}.pkl"
)

print(f"Saved recommendation_model_{embedding_dim}.pkl")

