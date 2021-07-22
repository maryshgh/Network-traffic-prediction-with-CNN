# Network-traffic-prediction-with-CNN

In this project I used the mobile network traffic of the city Milano for predicting the future mobile traffic. In order to incorporate both spatial and temporal correlation a CNN is used in which the height and width of images are corresponding to horizental and vertical neighborhood distance of a target location and number of channels or depth of images (f) is corresponding to the last f timesteps of network traffic. The data of past two hours (12 timestpes) is used to predict the data of the next timestep.


