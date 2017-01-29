#Behavioral Cloning Project

This Project is the third task of the Udacity Self-Driving Car Nanodegree program. The project is about building a Driver Behavioral Cloning using convolutional neural networks.

##Idea
The main idea was not only create an artificial neural network (ANN), able to drive a virtual car on a track in Simulator, but mimic human style of turning. In other words, the ANN should not drive along the track in the middle in a very unnatural way, but turnini in so-called classic racing line. The line is the path taken by a driver through a corner with the goal of smoothing out corners in the most efficient way which in case of real self-driving car can increase comfort and speed, prolong the life of tyres. Details on the classic racing line may be found on [http://www.drivingfast.net](http://www.drivingfast.net/racing-line/)

##Architecture and Approach
Driving is a dynamic process. That is why, from my point of view, it is very important to take into account not only current view from windshield, but also local motion direction and speed. In order to have such information, the net use two images: current and a previous image, shifted back in time on a constant value. For the training the previous image was 5 frames behind the current one. Current and previous images were concatenated along the color axis (so, input was X*Y*6). Generally speaking, the "Early Fusion" approach was realized. The approach was inspired by a [paper][1]

##Data and Data Augmentation
The training dataset was collected during 5 laps on the training track in the Simulator with a joystick. Desired behavior (driving style in turns) was used during the data collection process.

In fact, only augmented images were involved in training. Generators were used to create augmented batches for training and validation.I observed that in my case generators increase training time because of weak cpu: the most of time the powerful gpu was waiting for new input. However, using of generators is mandatory according to the project specification. Undoubtedly, generators are very useful in case of large datasets and more balanced hardware setup.


## References:
[1]: Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, Li Fei-Fei. Large-scale Video Classification with Convolutional Neural Networks // Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference. DOI [10.1109/CVPR.2014.223](https://doi.org/10.1109/CVPR.2014.223)
