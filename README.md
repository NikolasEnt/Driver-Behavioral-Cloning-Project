# Behavioral Cloning Project

This Project is the third task of the Udacity Self-Driving Car Nanodegree program. The project is about building a Driver Behavioral Cloning using convolutional neural networks.

## Idea
The main idea was not only create an artificial neural network (ANN), able to drive a virtual car on a track in the Simulator (provided by Udacity) using only images from dashboard camera, but mimic human style of turning. In other words, the ANN should not drive along the track in the middle in a very unnatural way, but turning in so-called classic racing line. The line is the path taken by a driver through a corner with the goal of smoothing out corners in the most efficient way which in case of real self-driving car can increase comfort and speed, prolong the life of tyres. Details on the classic racing line may be found on [http://www.drivingfast.net](http://www.drivingfast.net/racing-line/).

## Data structure
`model.py` - The script used to create and train the model.

`drive.py` - The script to drive the car.

`img_tools.py` - The useful image processing functions based on OpenCV.

`model.json` - The model architecture.

`model.h5` - The model weights.

## How to run the model

In order to train the model, use:

`python model.py`

You have to provide images (training dataset) and driving log `driving_log.csv` obtained from the Simulator in training mode.

To evaluate performance of the model, run predictions while the Simulator is in autonomous mode:

`python drive.py model.json`

## Architecture and Approach
Driving is a dynamic process. That is why, from my point of view, it is very important to take into account not only current view from windshield, but also local motion direction and speed. In order to have such information, the ANN use two images: current and a previous image, shifted back in time on a constant value. For the training the previous image was 5 frames behind the current one. Current and previous images were concatenated along the color axis (so, input was XxYx6). Generally speaking, the "Early Fusion" approach was realized. The approach was inspired by a paper [1].

It was shown, that such problem could be solved by Convolutional Neural Networks (CNN) [2], so, convolutional architecture was used. The final architecture, which is the result of try and error process, consists of 4 convolutional layers followed by 2 fully connected layers. It was advised [1] to use bigger convolutional kernel  on the first convolutional layer for such CNNs. In the case kernel 7x7 was applied. The second convolutional layer involved (2,2) stride to reduce number of elements. For the same purpose two max pooling layers were used as well. Dropout layers were added to prevent overfitting. ReLU was used as activation. Architecture with details on output sizes of each layers is depicted on the figure below. 

![Architecture](/images/model.png)

## Data and Data Augmentation
The training dataset was collected during 5 laps on the training track in the Simulator with a joystick. Desired behavior (driving style in turns) was used during the data collection process. Example of raw image from the training dataset is presented below:

![Original image](/images/or.jpg)


Images were preprocessed: histogram equalization, gaussian blurring with kernel (5,5) to eliminate unnecessary details and contrast increase were applied. Original images  were 320x160 px, but top ~50 px contain sky and unnecessary details, so, they were cropped. Images were augmented by random brightness variation, vertical shifts and random upper boundary of the region of interest shifts. The last operation was used to obtain images which emulate sloppy road after image scaling to the same resolution (128x64 px). The training track has only one right corner and a lot of left corners. Some images were flipped vertically (with sign changing for the corresponding steering angle) in order to equalize number of left and right corners. Images from left and right cameras were used as well. An constant angle was added to the corresponding left camera and subtract from the right camera in order to simulate effect of center camera shift to left or right. Actually, finetuning of the constant angle `SHIFT_CUR` was the most time consuming part of the project as the final model is quite sensitive to it.

Some examples of prepared and augmented images, produced out of raw image above, are presented below.

![Augmented image 1](/images/ag1.jpg) ![Augmented image 2](/images/ag2.jpg)

![Augmented image 3](/images/ag3.jpg) ![Augmented image 4](/images/ag4.jpg)

In fact, only augmented images were involved in training. Generators were used to create augmented batches for training and validation. I observed that in my case generators increase training time because of weak cpu: the most of time the powerful gpu was waiting for new input. However, using of generators is mandatory according to the project specification. Undoubtedly, generators are very useful in case of large datasets and more balanced hardware setup.

For validation and online prediction images were preprocessed, cropped and scaled in the same way.

## Model training
The model was trained with Adam optimizer. The whole training process were for 6 epochs, 32768 images in each. Training dataset was organized in minibatches of 256 pairs of images each. Number of epochs was selected by experiments and correspond to the minimum of the validation loss.

## Results and discussion
Two concatenated images were used for prediction as well: the current and the previous one (from memory). Time shift was different from one, used for training. It may be explained by the fact that during training data collecting human used different point of view and has different angle of view (see screenshot below and compare with raw training image above). It may resulted in different necessary reaction time on different driving  situations, that is why optimal time shifts between images in a pair in case of training and actual prediction may be different.

![Simulator screenshot](/images/screen.png)

It was found out that optimal time difference between current and previous frames is 3 (instead of 5 during training). Predictions were produced not on every available frame, but using only every 3rd frame, because, as it was observed from the training drive log, human driver unable to update stearing angle so fast and, consequently, it is useless to spend extra computational power on it.

The model is quite sensitive to input data, so, the training dataset should be constructed wisely. 

The final model is able to drive around the training track and mimic desired driving pattern to some extend. Unfortunately, it is impossible to ride on the second (challenge track) because the used driving style uses whole road width, but there a lot of streetlights on the road sides. So, further tuning is needed to pass the second track.

## References:
[1]: Andrej Karpathy, George Toderici, Sanketh Shetty, Thomas Leung, Rahul Sukthankar, Li Fei-Fei. Large-scale Video Classification with Convolutional Neural Networks // Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference. DOI: [10.1109/CVPR.2014.223](https://doi.org/10.1109/CVPR.2014.223)

[2]: Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon Goyal, Lawrence D. Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, Xin Zhang, Jake Zhao, Karol Zieba. End to End Learning for Self-Driving Cars // [arXiv:1604.07316](https://arxiv.org/abs/1604.07316)
