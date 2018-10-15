# Taxi Demand-Supply Gap Prediction #

## Description ##
Both dedicated taxi services and ride-sharing (e.g., Uber) companies find it cruical to forecast demand of trips for a particular location. Demand-supply prediction enables these companies to maximize driver utilization while ensuring that riders have service available in times of need. 

I hope to build a demand-supply gap prediction that allows such a service to guide drivers to specific, granular geographic areas of a city. This solution enables taxi drivers to identify areas that will likely require taxi service in the immediate future, e.g., the next 30-minute time window. 

This repository hosts sample code that trains a supervised predictive model based on past demand/supply patterns and real-time taxi locations. The outcome is a live city heatmap that shows the predicted demand-supply gap for the next 30-minute time window.

Data used in this project is provided by Kaggle (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). The dataset describes a complete year of the paths for taxis running in the city of Porto, Portugal. Particularly it contains GPS coordinates of the paths and starting time of each trip. 

## Method ##
The dataset only has starting time of each trip available. 
I trained a gradient boosting decision tree 




## Results ##
![Alt text](results_actual.png?raw=true)

![Alt text](results_pred.png?raw=true)


