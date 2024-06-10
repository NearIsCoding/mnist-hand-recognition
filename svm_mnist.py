import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Loading data...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Data succesfully loaded!!!")

#Reshape the data to a 2D array
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

#Normalize the data to [0, 1]
x_train = x_train_flat.astype('float32') / 255.0
x_test = x_test_flat.astype('float32') / 255.0

#binarize the data 0 for black and 1 for white
threshold = 0.5
x_train_binarized = (x_train_flat > threshold).astype(np.float32)
x_test_binarized = (x_test_flat > threshold).astype(np.float32)

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the model
print("Training SVM model...")
svm_model.fit(x_train_binarized, y_train)
print("Model succesfully trained!!!")

print("Testing SVM model...")
y_pred = svm_model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

