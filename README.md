# Twitter Sentiment Analysis Using Deep Learning

## Table of Contents
1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Technologies Used](#technologies-used)
4. [Implementation Steps](#implementation-steps)
5. [Deployment](#deployment)
6. [Results](#results)
7. [Visual Results](#visual-results)
8. [Contributors](#contributors)

## Overview
This project is a comprehensive machine learning pipeline designed for Twitter sentiment analysis. Implemented using Recurrent Neural Networks (RNN) with a multi-layer Bidirectional Long Short-Term Memory (LSTM) architectures. The model leverages pre-trained embeddings and multiple LSTM layers to capture complex contextual dependencies in sequential text data, ensuring precise classification of tweets into positive, neutral, and negative sentiments.

## Key Features
- **Advanced Data Preprocessing**: Tokenization, stemming, lemmatization, and stop-word removal techniques are utilized for efficient text normalization.
- **Dataset**: The training data consisted of over 1.2 million samples, and the test data had approximately 350k samples. The dataset is based on data from the following two sources:
  - University of Michigan Sentiment Analysis competition on Kaggle
  - Twitter Sentiment Corpus by Niek Sanders
- **RNN with Bidirectional LSTM Architecture**: A multi-layered model architecture that combines several Bidirectional LSTM layers to enhance the model's ability to understand dependencies in both forward and backward directions. The final architecture includes stacked LSTM layers, dense layers for deeper representation learning, and regular dropout for improved generalization.
- **Model Evaluation**: Precision, recall and F1-score metrics are leveraged for performance analysis, ensuring the model generalizes well to unseen data.
- **Scalable Deployment**: Deployed the model using a simple Hugging Face platform for easy accessibility and integration.

## Technologies Used
- **Programming Language**: Python (optimized with NumPy and Pandas for data handling)
- **Deep Learning Libraries**: TensorFlow, Keras for model building and training
- **NLP Libraries**: NLTK, SpaCy for preprocessing and feature extraction
- **Cloud Deployment**: Hugging Face for deployment

## Implementation Steps
1. **Data Collection and Preparation**: Data sourced from established datasets and processed using Python libraries to handle noisy text data.
2. **Preprocessing Pipeline**:
   - Tokenization using NLTK
   - Lemmatization for uniformity
   - Removal of stop words and special characters
3. **Model Architecture**:
   - Sequential model using an embedding layer initialized with a pre-trained embedding matrix
   - A series of Bidirectional LSTM layers:
     - First layer with 128 units and `return_sequences=True`
     - Second layer with 64 units and `return_sequences=True`
     - Third layer with 32 units and `return_sequences=True`
   - A final LSTM layer with 16 units and return_sequences=False
   - Dense layers for non-linear transformations:
     - A dense layer with 64 units and ReLU activation
     - A dense layer with 32 units and ReLU activation
   - Output layer with softmax activation for multi-class classification
   - Dropout layers (20%) for regularization after each LSTM and dense layer
4. **Training Strategy**:
   - Stratified k-fold cross-validation for comprehensive model validation
   - Optimized using Adam optimizer and a learning rate scheduler for adaptive learning
5. **Model Evaluation and Hyperparameter Tuning**:
   - Hyperparameters fine-tuned using grid search and Bayesian optimization
   - Performance measured through confusion matrices and precision-recall curves

## Deployment
The final model was deployed using Hugging Face for seamless deployment and accessibility. The solution is exposed through a REST API endpoint, allowing integration with web applications and data analysis platforms.

## Results
- **Accuracy**: Achieved a classification accuracy of 83% on the test set.
- **Precision and Recall**: High precision and recall scores for positive and negative classes, highlighting the model's capability in sentiment differentiation.

## Visual Results
Below are visual representations of the model's performance:
- **Classification Report**

  ![image](https://github.com/user-attachments/assets/38012de9-6d9e-41a2-936e-adc669c8fbf2)
- **F1-Score**

  ![image](https://github.com/user-attachments/assets/6407c28b-12aa-40b2-ba29-cf3f91b0538d)
- **Confusion Matrix**

  ![image](https://github.com/user-attachments/assets/bcccbde6-87a1-4297-b177-84ca5973ff3f)

## Contributors
- [Soumya Dhakad](https://github.com/soumya-1712)
- [Hardik Kanzariya](https://github.com/MrHardik-k)
- [Aman Raj](https://github.com/Amanraj4482)
- [Priyanshu Pandey](https://github.com/Harshpf)

---
