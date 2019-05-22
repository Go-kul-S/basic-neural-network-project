# banking -- customer relations
This is a project demonstrating an Artificial Neural Network which can be trained on bank information data
and predict whether a new customer will stay with the bank or not.

This is an example for building a simple Artificial Neural Network from scratch. 
The file Churn_Modelling.csv has 10000 records which contain information on customers from the bank.
The neural network has to be trained on this data and should be able to predict whether a new customer will stay in the bank or not.

The data from the csv file cannot directly be fed as inputs to the neural network and hence we need to prepare the data.
This is done using scikit learn libraries. 

Different layers of the ANN has been added using the Sequential Function from keras library.

At the end, there are different sections of code for training the network and predicting for a single customer information. 
