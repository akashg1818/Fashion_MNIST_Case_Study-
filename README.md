# Fashion-MNIST Image Classification – Mini Project

## 📌 Project Overview

This mini-project demonstrates an **end-to-end image classification system** using the **Fashion-MNIST dataset**. The project combines **Deep Learning (CNN)** with **classical Machine Learning baselines** to perform a comparative performance study.

The system is designed following **engineering mini-project standards**, with modular code structure, detailed documentation, reproducibility, and command-line execution support.

---

## 🎯 Objectives

* To understand image classification using Convolutional Neural Networks (CNN)
* To preprocess and normalize image datasets
* To compare Deep Learning models with classical ML algorithms
* To evaluate model performance using accuracy and confusion matrix
* To follow professional engineering coding and documentation practices

---

## 📂 Dataset Description

* **Dataset Name:** Fashion-MNIST
* **Source:** Keras built-in dataset
* **Total Samples:** 70,000 grayscale images
* **Image Size:** 28 × 28 pixels
* **Classes:** 10 clothing categories

### Class Labels

1. T-shirt / Top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle Boot

---

## 🧠 System Architecture

### 1. Data Preprocessing

* Normalization of pixel values (0–255 → 0–1)
* Reshaping images for CNN input
* Train / Validation / Test split

### 2. Deep Learning Model (CNN)

* Convolutional Layers
* Batch Normalization
* Max Pooling
* Dropout for regularization
* Fully Connected Dense Layers
* Softmax output layer

### 3. Classical ML Baselines

* Logistic Regression
* Linear Support Vector Machine (SVM)
* Random Forest Classifier

---

## ⚙️ Technologies Used

| Technology         | Purpose                 |
| ------------------ | ----------------------- |
| Python 3.x         | Programming Language    |
| TensorFlow / Keras | Deep Learning Framework |
| NumPy              | Numerical Computation   |
| Matplotlib         | Visualization           |
| Scikit-learn       | Classical ML Models     |

---

## 📁 Project Structure

```
Fashion-MNIST-MiniProject/
│
├── fashion_mnist_case_study.py
├── artifacts/
│   ├── best_model.keras
│   ├── final_model.keras
│   ├── training_curve.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 2️⃣ Train CNN Model

```bash
python fashion_mnist_case_study.py --train
```

### 3️⃣ Run Baseline Models

```bash
python fashion_mnist_case_study.py --baseline
```

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## 📈 Results Summary

* CNN achieves **higher accuracy** compared to classical ML models
* Data augmentation improves generalization
* Dropout reduces overfitting

*(Exact results may vary due to training randomness)*

---

## ✅ Key Features

* Modular and well-documented code
* Engineering mini-project standard formatting
* CLI-based execution
* Reproducible experiments using fixed random seed
* Comparative study between CNN and ML models

---

## 🎓 Academic Relevance

This project satisfies requirements for:

* Engineering Mini Project
* Machine Learning / Deep Learning Lab
* Academic Demonstration and Viva

---

## 👨‍💻 Author Information

**Name:** Akash Subhash Guldagad
**Course:** Engineering (Mini Project)
**Academic Year:** 2025–26

---

## 📜 License

This project is developed for **educational purposes only**.

---

## ⭐ GitHub Note

If you find this project useful, feel free to ⭐ the repository.
