import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('Iris.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
     

probabilities = model.predict_proba(X_test)
     

for i in range(5):
    print(f"Sample {i + 1} - True class: {y_test.iloc[i]}, Predicted class: {y_pred[i]}")
    print("Predicted Probabilities:")
    for j, class_name in enumerate(model.classes_):
        print(f"{class_name}: {probabilities[i, j]:.4f}")
    print("\n")

    
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
     

print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)