### GitHub Readme (Plain Text)

# Fruit 360 Classification Project

This repository contains the implementation of **Detection and Classification of Fruits and Vegetables Using Machine Learning and Deep Learning** techniques based on the Fruits-360 dataset. The project compares traditional machine learning methods with deep learning architectures and achieves high classification performance.

---

## Table of Contents
- **Introduction**
- **Dataset Structure**
- **Preprocessing**
- **Models**
  - Traditional Machine Learning
  - Artificial Neural Network (ANN)
  - Convolutional Neural Network (CNN)
- **Results and Performance**
- **Conclusion**
- **References**

---

## Introduction
The goal of this project is to classify various fruits and vegetables using the **Fruits-360** dataset. The approaches used include:
- Dimensionality Reduction using PCA.
- Traditional Machine Learning Algorithms (Random Forest, SVM, XGBoost, LightGBM).
- Deep Learning Architectures (ANN and CNN).

---

## Dataset Structure
The dataset is organized into the following folders:
- **`fruits-360_dataset_100x100`**: Resized images (100x100 resolution).
- **`fruits-360_dataset_original-size`**: Raw images in original resolution.
- **`Training`**: Training images.
- **`Validation`**: Validation images for hyperparameter tuning.
- **`Test`**: Test images for evaluation.

---

## Preprocessing
Steps involved in data preprocessing:
1. **Normalization**: Pixel values scaled to the range [0, 1].
2. **Image Resizing**: Consistent dimension (100x100) across all images.
3. **Label Encoding**: Converting labels into numerical values.
4. **Dataset Splitting**: Dividing into training, validation, and test sets.

---

## Models

### Traditional Machine Learning
Implemented models:
- **Random Forest**: Number of estimators = 100, Random state = 42.
- **Support Vector Machine (SVM)**: RBF kernel with hyperparameter tuning using GridSearchCV.
- **XGBoost**: Random state = 42.
- **LightGBM**: Random state = 42.

### Artificial Neural Network (ANN)
Architecture:
- Input Layer: Number of PCA components.
- Hidden Layers: Two fully connected layers with ReLU activation.
- Output Layer: Softmax activation for multi-class classification.
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Regularization: Dropout and L2 weight regularization.  
- Epochs: 50, Batch Size: 64  

### Convolutional Neural Network (CNN)
Custom architecture:
- Convolutional Layers: 3x3 filters for feature extraction.
- Pooling Layers: MaxPooling for spatial dimension reduction.
- Dropout Layers: Dropout rate of 0.25.
- Fully Connected Layers: Dense layers with Softmax activation.
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Epochs: 20, Batch Size: 8  

---

## Results and Performance

### Key Highlights
- **CNN** achieved the best performance with **99.9% accuracy**.
- **PCA** significantly reduced computational complexity.
- Among traditional models, **Random Forest** had the highest accuracy.
- Regularization and data augmentation techniques helped mitigate overfitting in ANN and CNN.

### Performance Metrics
- Precision, Recall, F1-score, and Accuracy were calculated for each model.
- Confusion matrices were used to analyze misclassifications.

---

## Conclusion
This project successfully classified fruits and vegetables using various machine learning and deep learning methods. The custom CNN model demonstrated exceptional performance, achieving **99.9% accuracy**, making it the optimal approach for this classification task.

---

## References
1. Fruits-360 Dataset: [https://www.kaggle.com/datasets/moltean/fruits/data](https://www.kaggle.com/datasets/moltean/fruits/data)
2. PCA Implementation: [https://www.kaggle.com/code/navabhaarathi20003/fruit-classification-pca-svm-knn-decisiontree](https://www.kaggle.com/code/navabhaarathi20003/fruit-classification-pca-svm-knn-decisiontree)
3. ANN Implementation: [https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-andclassification-of-the-fruits360-image-3c56affa4491](https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-andclassification-of-the-fruits360-image-3c56affa4491)
4. CNN Implementation: [https://www.kaggle.com/code/endofnight17j03/fruit-classification-cnn](https://www.kaggle.com/code/endofnight17j03/fruit-classification-cnn)
5. IEEE Documentation: [https://ieeexplore.ieee.org/document/10601062](https://ieeexplore.ieee.org/document/10601062)
