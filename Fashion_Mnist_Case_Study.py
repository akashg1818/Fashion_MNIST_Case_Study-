###################################################################################################
# Project Title : Fashion-MNIST Image Classification Case Study
# Description   : End-to-end deep learning pipeline using CNN along with
#                 classical machine learning baselines for Fashion-MNIST dataset.
# Author        : Akash Subhash Guldagad
# Date          : 22/09/2025
###################################################################################################

# ==============================
# Import Required Libraries
# ==============================

import os                          # For directory and file operations
import argparse                    # For command-line argument parsing
import numpy as np                 # Numerical computations
import matplotlib.pyplot as plt    # Plotting graphs and images

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.random import set_seed

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==============================
# Global Constants
# ==============================

SEED = 42                          # Seed for reproducibility

ARTIFACT_DIR = "artifacts"         # Directory to store outputs
BEST_MODEL = os.path.join(ARTIFACT_DIR, "best_model.keras")
FINAL_MODEL = os.path.join(ARTIFACT_DIR, "final_model.keras")

# Fashion-MNIST class labels
FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


###################################################################################################
# Function    : ensure_dir
# Inputs      : path (str) – Directory path
# Outputs     : None
# Description : Creates directory if it does not exist.
# Author      : Akash Subhash Guldagad
# Date        : 22/09/2025
###################################################################################################
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


###################################################################################################
# Function    : load_data
# Inputs      : val_split (float) – Validation split ratio
# Outputs     : Train, Validation, Test dataset tuples
# Description : Loads Fashion-MNIST dataset, normalizes pixel values,
#               reshapes data, and splits training data into train and validation sets.
# Author      : Akash Subhash Guldagad
# Date        : 22/09/2025
###################################################################################################
def load_data(val_split=0.1):
    # Load dataset from Keras
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Split training data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=val_split,
        random_state=SEED,
        stratify=y_train
    )

    # Normalize pixel values and add channel dimension
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_val   = (x_val   / 255.0).astype("float32")[..., None]
    x_test  = (x_test  / 255.0).astype("float32")[..., None]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


###################################################################################################
# Function    : load_flattened
# Inputs      : None
# Outputs     : Flattened training and testing datasets
# Description : Loads Fashion-MNIST and flattens images for classical ML models.
# Author      : Akash Subhash Guldagad
# Date        : 22/09/2025
###################################################################################################
def load_flattened():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Flatten 28x28 images into 784-dimensional vectors
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0

    return x_train, y_train, x_test, y_test


###################################################################################################
# Function    : build_cnn
# Inputs      : lr (float) – Learning rate
# Outputs     : Compiled CNN model
# Description : Builds and compiles a Convolutional Neural Network for
#               Fashion-MNIST classification.
# Author      : Akash Subhash Guldagad
# Date        : 22/09/2025
###################################################################################################
def build_cnn(lr=1e-3):

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Data augmentation for better generalization
    x = layers.RandomRotation(0.05)(inputs)

    # First convolution block
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Second convolution block
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    # Output layer
    outputs = layers.Dense(10, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


###################################################################################################
# Function    : run_baselines
# Inputs      : None
# Outputs     : Prints accuracy of baseline models
# Description : Trains and evaluates Logistic Regression,
#               Linear SVM, and Random Forest models.
# Author      : Akash Subhash Guldagad
# Date        : 22/09/2025
###################################################################################################
def run_baselines():
    x_train, y_train, x_test, y_test = load_flattened()

    # Logistic Regression
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300))
    ])
    logreg.fit(x_train, y_train)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg.predict(x_test)))

    # Linear SVM
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(max_iter=5000))
    ])
    svm.fit(x_train, y_train)
    print("Linear SVM Accuracy:", accuracy_score(y_test, svm.predict(x_test)))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
    rf.fit(x_train, y_train)
    print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(x_test)))


###################################################################################################
# Function    : plot_training_curves
# Inputs      : history, out_dir
# Outputs     : Saves accuracy plot
# Description : Plots training and validation accuracy curves.
###################################################################################################
def plot_training_curves(history, out_dir):
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve.png"))
    plt.close()


###################################################################################################
# Function    : train_and_evaluate
# Inputs      : batch_size, epochs, lr
# Outputs     : Trained model and saved artifacts
# Description : Trains CNN, evaluates on test data,
#               and saves model and reports.
###################################################################################################
def train_and_evaluate(batch_size=128, epochs=15, lr=1e-3):

    ensure_dir(ARTIFACT_DIR)

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

    # Build model
    model = build_cnn(lr)

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2
    )

    # Save final model
    model.save(FINAL_MODEL)

    # Plot training curves
    plot_training_curves(history, ARTIFACT_DIR)

    # Predictions
    y_pred = model.predict(x_test).argmax(axis=1)

    # Save classification report
    report = classification_report(y_test, y_pred, target_names=FASHION_CLASSES)
    with open(os.path.join(ARTIFACT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)


###################################################################################################
# Function    : parse_args
# Inputs      : None
# Outputs     : Parsed CLI arguments
# Description : Defines command-line arguments for training and baselines.
###################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Case Study")
    parser.add_argument("--train", action="store_true", help="Train CNN model")
    parser.add_argument("--baseline", action="store_true", help="Run baseline ML models")
    return parser.parse_args()


###################################################################################################
# Function    : main
# Inputs      : None
# Outputs     : None
# Description : Program entry point. Executes training or baselines based on CLI.
###################################################################################################
def main():
    set_seed(SEED)
    args = parse_args()

    if args.train:
        train_and_evaluate()
    elif args.baseline:
        run_baselines()
    else:
        print("Use --train or --baseline")


###################################################################################################
# Program Execution Starts Here
###################################################################################################
if __name__ == "__main__":
    main()
