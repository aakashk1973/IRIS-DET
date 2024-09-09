# train_model.py
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load a sample dataset (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an SVM model
svm = SVC(random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model's accuracy
svm_predictions = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'Model Accuracy: {svm_accuracy * 100:.2f}%')

# Save the trained model to a pickle file
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm, file)

print("Model saved as svm_model.pkl")
