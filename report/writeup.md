#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/dataset_sample.png "Dataset Sample"
[image2]: ./images/label_count.png "Label Count"
[image3]: ./images/sorted_label_count.png "Sorted Label Count"
[image4]: ./images/sample_rotation.png "Sample Rotation"
[image5]: ./images/c1.png "Predication Comparison1"
[image6]: ./images/c2.png "Predication Comparison2"
[image7]: ./images/c3.png "Predication Comparison3"
[image8]: ./images/c4.png "Predication Comparison4"
[image9]: ./images/c5.png "Predication Comparison5"
[image10]: ./images/featuremap.png "Feature map"
[image12]: ./images/feature100.png "Feature 100"
[image13]: ./images/original100.png "original 100"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

[German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb) is a multi-class, single-image dataset consists of 39,209 32×32 px color images  f

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the code cell[30] of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12360
* The shape of a traffic sign image is 32 x 32
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code [31] [60] [58] of the IPython notebook.


![alt text][image1]

Below is bar chart showing number of samples for each label

![alt text][image2]

Number of samples for each label in ascending order

![alt text][image3]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

I decided to convert the images to grayscale,
according to [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), grayscaling training images will improve model performance. We will come back to this point later.

The code for this step is contained in the code cell[17] of the IPython notebook.

Before grayscaling, other image/data translation were used for augmenting data.

For example, image rotation

![Sample image rotation in 90 degree][image4]

The code for splitting the data into training and validation sets is in the cell[37]. We use original provided data set segements.

In the future, given enough time and resource, we will look into how different size and segementation of training and validation sets

###### Normalization

1. Improve numerical stability

2. Pose problem in a well conditioned way in order to enable optimizer (e.g. SGD or auto tuned Adam optimizer) to proceeed 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)
To cross validate model, there is a validation set provided which has 4410 data points

Here is an example of an original image and an augmented image:

![Sample image rotation in 90 degree][image4]

The difference between the original data set and the augmented data set is the following ... 

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

LeNet-5 model architecture could be found in the code cell [22]

Both grayscaled and 3 channels (RGB) models was put under tests. Surprisingly, transforming image to grayscale does`t significantly improve overall accuracy

My final model consisted of the following layers:

When using RGB channel = 3

When using grayscale channel = 1

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32xCHANNEL RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 5x5xCHANNELx6 	|
| RELU					|												|
| Dropout				| rate = 0.5            |
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, output 5x5x6x16|
| RELU					|												|
| Dropout				| rate = 0.5            |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		    |                       |
| Dropout				| rate = 0.5            |
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout				| rate = 0.5            |
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout				| rate = 0.5            |
| Fully connected		| outputs 43        									|
| Softmax				|        									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### Architecture

LeNet-5 model architecture is decipted in the code cell [22]

Dropout layer is adopted to avoid potential overfitting

Keep probability is 0.5

With predefined: Hyperparameters
EPOCH = 50
BATCH_SIZE = 128

Model training and evaluation is contained in the code cell[25, 26, 27] of the IPython notebook.  

One hot encoding was used for classifier.

###### Optimizer
AdamOptimizer with start learning rate 0.0005

###### Cost function
Cross Entropy loss with softmax (probablity activation layer)



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

Calculation test and validation accuracy of the mode is located in the cell[28] of the Ipython notebook

My final model results were:
* training set accuracy of 87.0%
* validation set accuracy of 94.2%
* test set accuracy of 94.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet 5 without dropout was chosen as intial architecture

* What were some problems with the initial architecture?

Potential issue was overfitting. Before introducing drop out layer, validation accuracy was around 90.5 percent. Adding drop out layer slightly improve validation accuracy close to 95 percent

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


If a well known architecture was chosen:
* What architecture was chosen?

LeNet 5
* Why did you believe it would be relevant to the traffic sign application?

LeNet 5 performs well in text, hand recognization.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
In the original paper [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), reported training and test accuracy is around 97 to 98 percent, here, trained model accuracy is around 90 percent. 
Not ideal, as good as described in paper

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[curve](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/pack/curve.jpg)

[50 Km limit](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/pack/50.jpg)

[Keep Right](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/pack/keep_right.jpg)

[Stop_sign](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/pack/stop.jpg)

[Yield](https://github.com/IanBoyanZhang/CarND-Traffic-Sign-Classifier-Project/blob/master/pack/yield.jpg)


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

The code for making predictions on my final model is located in the cell[29] of the Ipython notebook.

Model network constantly mistake 30km speed limit sign to 50km speed limit sign. As we can see the speed limits signs all have round shape

General caution and curve image have same triangular shape, and internal stroke, the model mistake one for another

Model correctly predict stop sign and keep right, even the labeled images are dark

Priority road was mistaken by model for yield sign. My assumption is they all have trangular features. I will try extract feature maps
to verify the assuption



Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50Km Speed limit     		| 30Km speed limit   									| 
| Curve     			| General Caution 										|
| Keep Right					| Keep Right										|
| Stop      		| Stop					 				|
| Yield			| Priority Road      							|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. Comparing to test set result, this test sample 
is extremely small. Accuracy is also significantly lower. But model is able to capture important features from images.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell[24, 43] of the Ipython notebook.

For the first image, the model is really confident (more than 99.8 percent sure) that this is a 30Km/h speed limit sign


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998       			| 30Km/h Speed Limit | 
| .001     				| 20Km/h Speed Limit										|
|  <.001				| 50Km/h Speed Limit|
|  <.001	      			| 80Km/h Speed Limit					 				|
|  <.001				    | Other      							|


Dangerouse Curve has a relative sufficient training sets. However, as we can see, many training images are dark.
Potential improvement idea: increase image contrast or brightness. The model also seems confused red trangular board with 
Speed limit round boarder. Is this the sign of over-fitting/training?

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.476958 | General caution|
| 0.306895 | Speed limit (70km/h)|
| 0.181009 | Speed limit (20km/h)|
| 0.0344641| Speed limit (30km/h)|
| < 0.001  | Speed limit (120km/h)|


Keep right and ahead only also share similar features, such sign shape, background and arrows

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.55838 | Keep right|
| 0.264957 | Ahead only|
| 0.0802027| Road work|
| 0.0486142| Turn left ahead |
|0.0251512 | Turn right ahead|


Model nails it.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0| Stop|
|2.31413e-13| No entry|
|2.79405e-15|Speed limit (80km/h)|
|2.22512e-20|Speed limit (30km/h)|
|2.47789e-22|No passing for vehicles over 3.5 metric tons|


The model is really confident (more than 99.9 percent sure) that this is priority road. Image constrast issue again?

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.999734|priority road|
|0.000193087 | Dangerous curve to the right|
|2.31002e-05 | Slippery road|
|2.29349e-05 | Speed limit (60km/h)|
| 1.49192e-05 | No passing for vehicles over 3.5 metric tons|

##### Feature maps

Outputing feature map for "conv1" showing first hidden layer output after 2d convolution

Input layer with 5x5 filter size

![alt text][image10]


Input layer feature output

![alt text][image12] 

Original image

![alt text][image13]

#### Future improvements
Currently, we don't do extensive runs to optimize hyperparameters. There are a couple of popular hyperparameter optimization strategies, such as 
grid search, bayesian optimization, random search and gradient based optimization. Hopefully, in the near future, I would have enough bandwidth to explore these techniques in this or other ML problems.

Model improvements: would love to try modified LeNet architecture as seen in other papers. Such as connect multiple pooling layer together.

#### Appendix

To run jupyter solution book on CarND EC2 clusters.

You would need to manually download training set.

Install cv2: pip install opencv-python

Install tensorflow: pip install tensorflow

Ideally, tensorflow gpu

pip install tensorflow-gpu
