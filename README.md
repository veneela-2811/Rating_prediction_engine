# Supermarket Recommendation System                                                                                           

A deep learning–based recommendation engine built using **PyTorch**, designed to predict user preferences in a supermarket environment based on their historical purchase behavior.

---

## Project Overview

This project aims to simulate a supermarket’s personalized recommendation engine by analyzing user–item interaction data.  
Using embedding-based neural networks, the system learns **latent representations of products** and **captures user preferences** to predict how likely a user is to engage with similar products in the future.

---

##  Key Features

- **Embedding-based model** to represent items in dense vector space.  
- **User preference embeddings** computed by averaging item vectors.  
- **PyTorch-based training pipeline** with modular design for scalability.  
- **Train–validation–test split (40-40-20)** for balanced model evaluation.  
- **Evaluation metrics:** MSE Loss and Accuracy.  
- Achieved **~91.5% test accuracy** on 24,421 real-world purchase interactions.

---

##  Workflow

1. **Data Preprocessing:**  
   - Encoded user and item IDs using categorical encoding.  
   - Built user–item interaction matrices from purchase data.  

2. **Data Splitting:**  
   - Split data into **Train (40%)**, **Validation (40%)**, and **Test (20%)** using a custom splitter class.

3. **Model Architecture:**  
   - Used **`nn.Embedding`** to learn dense representations for each item.  
   - Averaged embeddings of items purchased by a user to form a **user vector**.  
   - Passed through a **Linear Layer** to predict user preference score.  

4. **Training:**  
   - Used **Mean Squared Error (MSE)** loss to minimize prediction error.  
   - Optimized with **Adam** optimizer (learning rate = 0.001).  
   - Trained over multiple epochs and tracked validation loss & accuracy.

5. **Evaluation:**  
   - Computed **Test Loss** and **Test Accuracy** using unseen data.  
   - Visualized performance metrics for analysis.

---

##  Technologies Used

| Category | Libraries |
|-----------|------------|
| **Core ML Framework** | PyTorch |
| **Data Handling** | pandas, numpy |
| **Sparse Matrices** | scipy |
| **Model Training** | torch.nn, torch.optim |
| **Data Splitting** | scikit-learn |
| **Utilities** | logging |

---

