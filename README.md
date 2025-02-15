# Crop Disease Detection Using Machine Learning

## About the Model
This machine learning project focuses on binary classification to determine whether an image depicts a disease-free or diseased condition. The dataset undergoes preprocessing using OpenCV, which includes resizing, RGB conversion, and normalisation. Labels are encoded using LabelEncoder, and the dataset is shuffled before being split into 80% training and 20% testing sets. A Support Vector Machine (SVM) classifier, implemented via Scikit-Learn, is trained on the data. The model is then saved using Pickle for future predictions. Performance is evaluated using accuracy scores, and the system is capable of classifying new images by loading the saved model and predicting their disease status.

## Model Performance
The model achieved a prediction accuracy of 76%. To validate its performance, two sample images were tested. The relatively lower accuracy is primarily attributed to challenges such as shadows in the images and irregular cropping, which hinder effective feature extraction and classification. These factors contribute to misclassifications, thereby impacting the overall performance of the model. Future improvements could involve enhanced image preprocessing techniques, such as adaptive thresholding, improved cropping methods, and data augmentation, to reduce noise and increase accuracy.
