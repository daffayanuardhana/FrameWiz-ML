![image](https://github.com/daffayanuardhana/FrameWiz-ML/assets/95835149/f82db00b-818b-4c36-92bd-5b9d22e59c42)# Glasses Frame Classification with CNN
This project aims to classify glasses frame images based on face shape using a Convolutional Neural Network (CNN). The provided code demonstrates the implementation of the CNN model using TensorFlow and Keras.

## Dataset

![image](https://github.com/daffayanuardhana/FrameWiz-ML/assets/95835149/f35930a0-3a97-4268-84c2-d5c77ae76d76)

The dataset used in this project consists of face images categorized by face shape. It is assumed that the dataset is structured in separate directories for each class (face shape) within the train and test directories. link to the dataset: https://www.kaggle.com/datasets/zeyadkhalid/faceshape-processed

## Model Architecture
![image](https://github.com/daffayanuardhana/FrameWiz-ML/assets/95835149/d374cf2c-7e5c-4409-8087-c3ed0ea382b1)

The CNN model architecture is defined as follows:

Input layer: Accepts grayscale images of size (img_width, img_height, 1).
Convolutional layers: Two convolutional layers with SELU activation and different kernel sizes.
MaxPooling layers: Two max pooling layers with different pool sizes and strides.
Flatten layer: Flattens the output from the previous layers.
Dense layers: Two fully connected (dense) layers with SELU activation.
Output layer: Dense layer with softmax activation, assuming 5 face shape classes.
Training
The model is trained using the model.fit() function with the following configurations:

Training set: Generated using the train_datagen.flow_from_directory() function with the specified target size, batch size, grayscale color mode, and categorical class mode.
Testing set: Generated using the test_datagen.flow_from_directory() function with similar configurations as the training set.
Loss function: Categorical cross-entropy.
Optimizer: Adam optimizer.
Metrics: Accuracy.
The model is trained for 100 epochs with a batch size of 32. The training and testing sets are shuffled during training.
Model used on app: https://drive.google.com/drive/u/2/folders/1tk_eG0Ce4R4fb4mWcFPqDMAVTS9IhVAR

## Results
The model achieved an accuracy of 94% on the testing set, indicating its effectiveness in classifying glasses frame images based on face shape.
![image](https://github.com/daffayanuardhana/FrameWiz-ML/assets/95835149/f0218511-54e9-4879-929d-30b4ebc25db7)

## Usage
To use this code for your own glasses frame classification task, follow these steps:

Prepare your dataset with face images categorized by face shape.
Adjust the img_width and img_height variables to match the size of your input images.
Ensure that the dataset is structured in separate directories for each class within the train and test directories.
Run the code and observe the training process and accuracy results.
Note: It is recommended to further fine-tune the model, adjust hyperparameters, and evaluate performance metrics based on your specific dataset and requirements.
