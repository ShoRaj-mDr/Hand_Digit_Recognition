# Hand Digit Recognition
Classify hand digits from 0-5 using Keras convolutional neural networks.  


## Requirements
Python3, Keras, TensorFlow, sklearn, OpenCV, Mathplotlib 


## Dataset Description
The images in the dataset are either showing the left hand or right hand, of which the hand may be holding up 0 to 5 finger digits. Each image is 128x128 pixels and is centered by the center of mass.

![Fingers Dataset](https://github.com/ShoRaj-mDr/Hand_Digit_Recognition/blob/master/images/Fingers%20Dataset.png)

Dataset contains total of 21,600 images
- Training set: 18000 images
- Testing set: 3600 images

### Label 
Labels are in two last characters of filename 
- L/R indicates left/right hand
- 0, 1, 2, 3, 4, 5 indicates number of fingers 
- With 0-5 digits and L/R hand, there are 12 total classes 

__Link:__   (http://kaggle.com/koryakinp/fingers)


## Simple Convolutional Neural Networks (CNNs)
The CNN model has 2 Convolutional Layers, with 32 filters followed by 64, 3x3 Kernel Size, and 2x2 Stride. The images then go through a MaxPooling layer of size 2x2, followed by ReLu as an activation function. The data is then flattened to convert into a 1-dimensional array which then goes through two Dense layers of sizes 32 and 6 (number of classes). The CNN model has two dropout rate of 0.25 and 0.5. The loss function is the categorical cross-entropy which just computes the cross-entropy loss between the labels and predictions with a learning rate of 0.005. 

__Model Pattern:__ 
Conv2d -> ReLU -> Conv2d -> ReLU -> MaxPooling -> Dropout -> Dense ->  Dropout -> Dense -> Softmax

```
model = Sequential()
model.add(Conv2D(32, (3, 3), strides = (2,2), padding='same',
            input_shape = train_shape))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides = (1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes))
model.add(Activation('softmax'))
return model
```


## Result
#### Confusion Matrix
![confusion_matrix](https://github.com/ShoRaj-mDr/Hand_Digit_Recognition/blob/master/images/confusion_matrix.png)

#### Accuracy
![Accuracy](https://github.com/ShoRaj-mDr/Hand_Digit_Recognition/blob/master/images/accuracy.png)

#### Loss
![Loss](https://github.com/ShoRaj-mDr/Hand_Digit_Recognition/blob/master/images/loss.png)

#### Prediction
![Prediction](https://github.com/ShoRaj-mDr/Hand_Digit_Recognition/blob/master/images/prediction.png)
