import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (same as before)
df = pd.read_csv("Dataset/RT_IOT2022 (1).csv")

# Shuffle the dataset and reset to default numerical index
df = df.sample(frac=1).reset_index(drop=True)

# Drop the 'id.orig_p' column from the feature set
features = df.drop(columns=['Attack_type', 'id.orig_p'])

# One-hot encode categorical features
x = pd.get_dummies(features)

# One-hot encode the target
y = pd.get_dummies(df['Attack_type'])
num_classes = y.shape[1]

# Train-test split (same split as before)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Load the pre-trained model
model = tf.keras.models.load_model('Models/100epoc.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')

# Make predictions
y_pred = model.predict(x_test)

# Convert predictions to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test.values, axis=1)

# Randomly select a few comparisons (let's say 10 random rows)
random_indices = np.random.choice(len(x_test), size=10, replace=False)

# Create a DataFrame to show actual vs guessed labels
comparison_table = pd.DataFrame({
    'Index': random_indices,
    'Actual Classifier': y_true_labels[random_indices],
    'Guessed Classifier': y_pred_labels[random_indices]
})

# Print the comparison table
print("\nRandom Sample of Predictions vs Actual Labels:")
print(comparison_table)

# Plotting model accuracy (if you still want to plot based on your test set)
plt.plot([test_accuracy] * 30, label='Test Accuracy')  # Just show a flat line for test accuracy
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
