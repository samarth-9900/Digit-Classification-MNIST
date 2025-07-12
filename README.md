# MNIST Digit Classification — Dense vs CNN

This project demonstrates handwritten digit classification using the **MNIST dataset**, built and trained in **Google Colab** using **TensorFlow/Keras**.

We compare two architectures:
- A basic **Dense Neural Network (DNN)**
- A **Convolutional Neural Network (CNN)**

---

## Open in Google Colab

Click below to open and run the notebook in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/samarth-9900/Digit-Classification-MNIST/blob/main/dnn_cnn_models.ipynb)

---

## Project Structure

| File                      | Description                                           |
|---------------------------|-------------------------------------------------------|
| `dnn_cnn_models.ipynb`    | Main notebook with model code and training pipeline   |
| `README.md`               | Project documentation                                 |
| `results.pdf`             | Training plots, confusion matrices,classification report  |

---

## Dataset

- **Name**: MNIST
- **Source**: `tensorflow.keras.datasets.mnist`
- **Size**: 60,000 training + 10,000 testing images (28×28 grayscale)
- **Labels**: Digits from 0 to 9

---

## Model Architectures

### Dense Neural Network (DNN)
- Flatten input
- Dense(64) + ReLU
- Dense(32) + ReLU
- Dense(10) + linear
- used logits

> A basic feedforward model treating the image as a flat vector.

---

### Convolutional Neural Network (CNN)
- Conv2D(32) + ReLU + MaxPooling
- Conv2D(64) + ReLU + MaxPooling
- Dropout(0.25)
- Flatten
- Dense(64) + ReLU
- Dense(10) + linear
- used logits

> Learns spatial patterns, improving performance on image tasks.

---

## Compile
- loss = SparseCategoricalCrossentropy
- optimizer = Adam Optimizer

---

## Evaluation Metrics

- Accuracy on test set
- Confusion matrix
- Classification report (precision, recall, F1)
- Training/validation loss & accuracy curves

---

## Results

| Model     | Accuracy | Comments                             |
|-----------|----------|--------------------------------------|
| DNN       | ~97%     | Simple, quick, but lacks spatial awareness |
| CNN       | ~99%     | More powerful,Time-Taking, captures image structure |

---

