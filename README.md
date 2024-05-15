Wine-Quality-Predicition-AIML

The data file winequality-red.csv contains a dataset of 1599 data points with 1 1 features and a label which ranges from 3-
9.The aim of the project is to predict the quality of the wine by training the data on the given data points and then
creating a separate csv file for testing and feedinf the predicted value of quality in a new column and marking it as
good, average or bad. Any value of quality predicted which is less than 6 is marked as 'bad', equal to 6 marked as
•average' and greater than 6 is marked as'good'.Out of 1599 data points, 1000 are used to train the model and the
rest are used for testing.
Technology used- Python3, 

DESCRIPTION

fixed.acidity

volatile.acidity	

citric.acid	

residual.sugar	

chlorides	

free.sulfur.dioxide	

total.sulfur.dioxide	

density	

pH	

sulphates	

alcohol	

quality (label)



The final prediction is based on Random Forest. Train accuracy is around 0.93 which shows that it has low bias.
