# 🌸 Image Classifier Project - Machine Learning with TensorFlow Nanodegree

## Overview
This project is part of the **Machine Learning with TensorFlow Nanodegree** and focuses on building a **Convolutional Neural Network (CNN)**-based image classifier. The goal is to classify images of flowers into 102 different categories from the [Oxford Flowers 102 dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/).

The project demonstrates the full workflow for training a deep learning model: data preprocessing, model building, training, evaluation, and saving the model for future use.

---

## Project Objectives
- Load and preprocess the Oxford Flowers 102 dataset.
- Build a CNN-based image classifier using **TensorFlow** and **TensorFlow Hub**.
- Train the model on the training set and validate its performance.
- Evaluate the trained model on unseen test data.
- Save the trained model for inference.

---

## Dataset
- **Name:** Oxford Flowers 102
- **Categories:** 102 flower species
- **Size:**
  - Training set: 1020 images
  - Validation set: 1020 images
  - Test set: 6149 images
- **Image Size:** Resized to 224x224 pixels for CNN input
- **Source:** [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

---

## Technologies Used
- **Languages & Libraries:** Python, TensorFlow, TensorFlow Hub, Keras, NumPy, Matplotlib
- **Concepts:** 
  - Convolutional Neural Networks (CNN)
  - Transfer Learning using MobileNet V3
  - Image preprocessing and augmentation
  - Model training and evaluation
  - Saving and loading Keras models

---

##Key Results
- Training Accuracy: >99%
- Validation Accuracy: ~87%
- Test Accuracy: ~83%
- The CNN-based classifier successfully learned to distinguish between 102 flower species using transfer learning.

---
##Conclusion

This project serves as a foundation for building AI applications that leverage image classification. The trained CNN can be integrated into mobile apps, web services, or any AI-driven system requiring image recognition.
