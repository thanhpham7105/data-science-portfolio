# Deep Learning: CNN & NLP

## Overview
Two deep learning projects: a convolutional neural network for image classification and an embedding-based neural network for text sentiment classification.

## 1. CNN -- image classification
Convolutional neural network to classify plant-seedling images by species at an early growth stage: data exploration, normalization/augmentation, model construction (convolutional/pooling/dense layers), training, and evaluation (accuracy, loss curves, confusion matrix).
- `cnn_image_classification.py` -- the model-building script
- `cnn_seedling_classification.ipynb` -- the full notebook workflow (data loading, augmentation, model definition, training, evaluation)
## 2. Sentiment analysis -- neural text classification
Embedding-based neural network (Embedding layer -> pooling -> dense layers) to classify short customer-review sentences as positive or negative, using the public **UCI Sentiment Labelled Sentences** dataset (Amazon, IMDB, and Yelp reviews).
- `sentiment_analysis_nn.py`, `sentiment_analysis_nn.ipynb`
- `data/amazon_cells_labelled.txt`, `data/imdb_labelled.txt`, `data/yelp_labelled.txt`

**Dataset citation:** Kotzias, D., Denil, M., De Freitas, N., & Smyth, P. (2015). From Group to Individual Labels using Deep Features. *KDD 2015*. Dataset available via the UCI Machine Learning Repository.

## Skills demonstrated
CNN architecture design, image augmentation/preprocessing, NLP preprocessing (tokenization, padding, embeddings), neural network training/evaluation with TensorFlow/Keras, model interpretation for business use.
