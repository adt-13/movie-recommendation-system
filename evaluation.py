# evaluation.py or in a Jupyter Notebook

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Load data
ratings_df = pd.read_csv('C:/Users/HP/Desktop/python_work/academic_garbage/project/movie_recommender_system/ml-latest-small/ratings.csv')

# Prepare data for Surprise
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Train the SVD model (your CF component)
cf_model = SVD(n_factors=100, n_epochs=20, random_state=42)
cf_model.fit(trainset)

# Get predictions on the test set
predictions = cf_model.test(testset)

# --- Metric 1: RMSE & MAE ---
# This is your "Accuracy/Loss" for the CF model
print("CF Model Performance:")
accuracy.rmse(predictions)
accuracy.mae(predictions)

# --- Metric 2 & 3: Confusion Matrix & ROC/AUC ---
# Define "liked" as rating > 3.5
like_threshold = 3.5
y_true = [1 if rating > like_threshold else 0 for _, _, rating in testset]
y_pred_scores = [pred.est for pred in predictions]
y_pred_class = [1 if score > like_threshold else 0 for score in y_pred_scores]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_class)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Disliked', 'Liked'], yticklabels=['Disliked', 'Liked'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# ROC Curve and AUC Score
auc_score = roc_auc_score(y_true, y_pred_scores)
print(f"\nROC AUC Score: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()