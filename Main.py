import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
df = pd.read_csv("Dataset/RT_IOT2022 (1).csv")

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Drop columns
features = df.drop(columns=['Attack_type', 'id.orig_p'])

# One-hot encode categorical features
x = pd.get_dummies(features)

# Standardize features
scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

# One-hot encode target
y = pd.get_dummies(df['Attack_type'])
num_classes = y.shape[1]

# Print class distribution
print("Class distribution:")
print(y.sum().sort_values(ascending=False) / len(y))

# Split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Compute class weights
labels = df['Attack_type']
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_labels = {label: i for i, label in enumerate(y.columns)}
weights_dict = {class_labels[label]: weight for label, weight in zip(np.unique(labels), class_weights)}

# Build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_dim=len(x_train.columns)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(x_test, y_test),
    class_weight=weights_dict
)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
