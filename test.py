import numpy as np
import pandas as pd
import torch.nn as nn
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import sys
# ================= Load saved model and artifacts =================
model_name = sys.argv[1]
saved_data = joblib.load(model_name)
model_state = saved_data['model_state']
user_embeddings = saved_data['user_embeddings']
item_to_liked_users = saved_data['item_to_liked_users']
num_items = saved_data['num_items']
embedding_dim = saved_data['embedding_dim']

# ================= Reconstruct model architecture =================
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
model.load_state_dict(model_state)
model.eval()

# ================= Collaborative filtering function =================
def predict_user_item_score(user_id, item_id, rating_threshold=3.0):
    liked_users = item_to_liked_users.get(item_id, [])
    liked_users_filtered = [uid for uid in liked_users if user_item_matrix[uid, item_id] > rating_threshold]
    if len(liked_users_filtered) == 0:
        return 0.0
    target = user_embeddings[user_id].reshape(1, -1)
    others = user_embeddings[liked_users_filtered]
    return float(cosine_similarity(target, others).mean())

def recommend_for_user(user_id, user_item_matrix, K=10):
    user_ratings = user_item_matrix[user_id]
    unrated_items = np.where(user_ratings == 0)[0]
    scores = []
    for item_id in unrated_items:
        score = predict_user_item_score(user_id, item_id)
        scores.append((item_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:K]

# ================= Load data and build user-item matrix =================
data = pd.read_csv("recommend.csv")
data.columns = ['uid', 'vid', 'rate', 'time']
data['uid'] = data['uid'].astype('category').cat.codes
data['vid'] = data['vid'].astype('category').cat.codes
num_users = data['uid'].nunique()
num_items = data['vid'].nunique()

user_item_matrix = np.zeros((num_users, num_items), dtype=np.float32)
for _, row in data.iterrows():
    user_item_matrix[row.uid, row.vid] = row.rate

# ================= Mask one positive item per user and compare =================
fixed_users = [3, 7, 12, 18, 25, 31, 44, 52, 60, 71]

masked_pairs = []

for u in fixed_users:
    positive_items = np.where(user_item_matrix[u] >= 3)[0]
    if len(positive_items) == 0:
        continue
    hidden_item = positive_items[0]
    masked_pairs.append((u, hidden_item))

corrects = 0
total = len(masked_pairs)

print("Fixed masked evaluation:\n")

for user_id, hidden_item in masked_pairs:
    actual_rating = user_item_matrix[user_id, hidden_item]
    predicted_score = predict_user_item_score(user_id, hidden_item)

    print(f"User {user_id}, Item {hidden_item}:")
    print(f"  Actual rating: {actual_rating}")
    print(f"  Predicted similarity score: {predicted_score:.4f}")

    corrects += (actual_rating >= 3 and predicted_score > 0.50)

print(f"\nAccuracy proxy: {corrects}/{total}")

