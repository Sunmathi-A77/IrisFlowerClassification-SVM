# app.py
import pickle
import numpy as np

# Load the saved model
with open('linear_svm_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
le = data['label_encoder']

# Get user input
print("Enter iris flower measurements:")
sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

# Prepare input array
new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Scale input
new_flower_scaled = scaler.transform(new_flower)

# Predict
pred_class_encoded = model.predict(new_flower_scaled)
pred_class = le.inverse_transform(pred_class_encoded)

print(f"\nPredicted Iris Species: {pred_class[0]}")
