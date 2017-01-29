#Behavioral Cloning Project

This Project is the thirt task of the Udacity Self-Driving Car Nanodegree program.

##Idea

##Architecture of CNN


##Data and Data Augmentation
The training dataset was collected during 5 laps on the training track in the Simulator with a joystick. Desired behavior (driving style in turns) was used during the data collection process.

Generators were used to create augmented batches for training and validation.I observed that in my case generators increase training time because of weak cpu: the most of time the powerful gpu was waiting for new input. However, using of generators is mandatory according to the project specification. Undoubtedly, generators are very useful in case of large datasets and more balanced hardware setup. 