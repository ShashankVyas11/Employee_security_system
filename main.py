import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated accelerometer data (replace with real data)
# Features: [mean_x, mean_y, mean_z, std_x, std_y, std_z]
data = np.random.rand(100, 6)
labels = np.random.choice([0, 1], size=100)  # 0 for non-employee, 1 for employee

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Simulated real-time data (replace with actual smartphone data)
real_time_data = np.random.rand(1, 6)

# Predict if the person is an employee
prediction = model.predict(real_time_data)
if prediction[0] == 1:
    print("Employee detected. Opening doors...")
else:
    print("Non-employee detected. Access denied.")
