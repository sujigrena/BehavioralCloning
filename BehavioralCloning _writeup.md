# **Behavioral Cloning** 



The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image7]: ./examples/placeholder_small.png "Cropped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a 5 convolution layers with 5x5 and 3x3  filter sizes and depths between 3 and 64 followed by 5 fully connected layers (model.py lines 85-98) 

The model includes RELU layers in between the convolutional layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 86). 

_________
Layer (type) |                   Output Shape    |      Param #  |   Connected to                     
|--------------:|:-------------:|:--------------------:|:----------:|
|lambda_1 (Lambda)          |      (None, 160, 320, 3)|   0          |lambda_input_1[0][0] |            
 |cropping2d_1 (Cropping2D)      |   (None, 65, 320, 3)   |  0      |      lambda_1[0][0]                   
convolution2d_1 (Convolution2D)   |(None, 61, 316, 3)  |   228  |        cropping2d_1[0][0]     |           
convolution2d_2 (Convolution2D) |  (None, 29, 156, 24)  |  1824 |        convolution2d_1[0][0]    |         
convolution2d_3 (Convolution2D)  | (None, 13, 76, 36)    | 21636 |       convolution2d_2[0][0]     |        
convolution2d_4 (Convolution2D)  | (None, 6, 37, 48)   |   15600 |       convolution2d_3[0][0]   |          
convolution2d_5 (Convolution2D)   |(None, 2, 18, 64) |     27712 |       convolution2d_4[0][0]    |         
flatten_1 (Flatten)        |       (None, 2304)   |        0    |        convolution2d_5[0][0]   |          
dense_1 (Dense)                 |  (None, 1166)    |       2687630    |  flatten_1[0][0]      |             
dense_2 (Dense)        |           (None, 100)   |         116700 |      dense_1[0][0]    |                 
dense_3 (Dense)           |        (None, 50)      |       5050   |      dense_2[0][0]   |                  
dense_4 (Dense)       |            (None, 10)    |         510 |         dense_3[0][0]     |                
dense_5 (Dense)       |            (None, 1)       |       11     |      dense_4[0][0]           |          

____________________________________________________________________________________________________

Total params: 2,876,901
Trainable params: 2,876,901
Non-trainable params: 0
____________________________________________________________________________________________________


#### 2. Attempts to reduce overfitting in the model

I've not used any dropout layers in the model contains because using them seems to chop of important features of the road which ultimately leads to an inefficient model.

Rather I have used augmented data and trained for lesser number of epochs to avoid overfitting.

The model was also trained and validated on different data sets to ensure that the model was not overfitting (code line 44-74). The model was tested by running it through the simulator multiple times and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99), since we are creating a regression model, the appropriate error function chosen was mse.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. As we are aware there are 3 cameras mounted in the car and hence we get the outputs of 3 camera images. But the steering angle is specific to the center camera images. I used a combination of center lane driving by, recovering from the left and right sides of the road and driving the road in the clockwise direction and I made use of all the 3 camera images and just to make the right and left camera images appropriate to the steering angle, I have used a correction factor of 0.2. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make use of convolutional neural networks followed by dense layers to get one output which is the steering angle of the car.

My first step was to use a convolution neural network model similar to the nvidia architecture though not exact same one.I thought this model might be appropriate because it had appropriate number of filters to extract the features of the road and depth of the output layers fairly enough to capture the minute curvatures of the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set which is 20% of the training data. I found that my first model had high mean squared error on the training set as well as a high mean squared error on the validation set. This implied that the model was not designed right. I increased the number of epochs then the mean squared error started decreasing gradually and hence the mean squared error on the validation set. After certain number of epochs the training mean squared error was still less but the validation mean squared loss started increasing. This implied overfitting. 

To combat the overfitting, I ran the training for lesser number of epochs.

Then I tried altering the depth of the output feature maps to get a yet more accurate model with very less mean squared error for both training and validation data set. After so many attempts of trial and error I finally got the mentioned model which had the least error with less training data and epochs. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or didn't take the proper turns and to improve the driving behavior in these cases gave augmented data which gave the model more clarity on the features of the road and also added additional cropping layer to remove the unwanted features of the road. There was a spot where my model failed terribly even with the shortest mean squared error values. Then I added little more data for the exact same spot and retrained the model. Now the model has adopted to the deepest turn possible along with the minor turns and features of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-98)  consisted of a convolution neural network with the following layers and layer sizes:

_________
Layer (type) |                   Output Shape    |      Param #  |   Connected to                     
|--------------:|:-------------:|:--------------------:|:----------:|
|lambda_1 (Lambda)          |      (None, 160, 320, 3)|   0          |lambda_input_1[0][0] |            
 |cropping2d_1 (Cropping2D)      |   (None, 65, 320, 3)   |  0      |      lambda_1[0][0]                   
convolution2d_1 (Convolution2D)   |(None, 61, 316, 3)  |   228  |        cropping2d_1[0][0]     |           
convolution2d_2 (Convolution2D) |  (None, 29, 156, 24)  |  1824 |        convolution2d_1[0][0]    |         
convolution2d_3 (Convolution2D)  | (None, 13, 76, 36)    | 21636 |       convolution2d_2[0][0]     |        
convolution2d_4 (Convolution2D)  | (None, 6, 37, 48)   |   15600 |       convolution2d_3[0][0]   |          
convolution2d_5 (Convolution2D)   |(None, 2, 18, 64) |     27712 |       convolution2d_4[0][0]    |         
flatten_1 (Flatten)        |       (None, 2304)   |        0    |        convolution2d_5[0][0]   |          
dense_1 (Dense)                 |  (None, 1166)    |       2687630    |  flatten_1[0][0]      |             
dense_2 (Dense)        |           (None, 100)   |         116700 |      dense_1[0][0]    |                 
dense_3 (Dense)           |        (None, 50)      |       5050   |      dense_2[0][0]   |                  
dense_4 (Dense)       |            (None, 10)    |         510 |         dense_3[0][0]     |                
dense_5 (Dense)       |            (None, 1)       |       11     |      dense_4[0][0]           |          

____________________________________________________________________________________________________

Total params: 2,876,901
Trainable params: 2,876,901
Non-trainable params: 0
____________________________________________________________________________________________________


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and changed the angles accordingly thinking that this would help the model get trained better on the curves. Then I also used the images of all the three cameras mounted on the car and also modified the steering angle to fit the different cameras by adding a correction factor to the steering angles.

![alt text][image6]
![alt text][image7]

Then added additional data to make my model get trained to the deepest curve in the road.

After the collection process, I had 20,000 data points. I then preprocessed this data by passing it through the normalization layer followed by the cropping layer to chop off the unwanted part in the top and bottom of the image.

![alt text][image8]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by constantly reducing and converged mean squared error in training data set and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
