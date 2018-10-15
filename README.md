# Taxi Demand-Supply Gap Prediction #

## Description ##
Both dedicated taxi services and ride-sharing (e.g., Uber) companies find it critical to forecast demand of trips for a particular location. Demand-supply prediction enables these companies to maximize driver utilization while ensuring that riders have service available in times of need. 

We hope to build a demand-supply gap prediction that allows such a service to guide drivers to specific, granular geographic areas of a city (approximately one square mile or one square kilometer). This solution enables taxi drivers to identify areas that will likely require taxi service in the immediate future, e.g., the next 30-minute time window. 

This repository hosts sample code that trains a supervised predictive model based on past demand/supply patterns and real-time taxi locations. The outcome is a live city heatmap that shows the predicted demand-supply gap for the next 30-minute time window.

Toy data used in this demo is provided by [Kaggle Taxi Trajectory Prediction Challenge](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). The dataset contains GPS coordinates of the paths and starting time of each taxi trip in the city of Porto, Portugal, from July 1st, 2013 to June 30th, 2014. 

## Method ##
We treat the starting location and time of a taxi trip as demand, as they correspond to a rider request. The end location and time of a trip are treated as supply as it represents an available taxi.

The city map of Porto is divided into a grid, where grid size is 0.84 km by 1.1 km (1/100th degree longitude by 1/100th degree latitude). Each day is divided into 30-minute time windows. Then for each geospatial cell and and time window, we calculate rider requests (demand) and available taxis (supply). For demonstration purpose, in this repo we show a static demand-supply gap heatmap for a certain time window. If productionalized, the live heatmap would be expected to refresh every 30 minutes. Both the grid size and the duration of the time window would be configurable in production, where a higher refresh frequency is desirable. 

We train a gradient boosting decision tree for this task. When exploring, we also built a multi-layer perceptron neural network model; the performace gain did not justify its significantly longer training time. The gradient boosting decision tree algorithm is not compatible with online learning. However, if used in a real business environment, the first opportunity to increase prediction accuracy is to utilize an online learning algorithm to dynamically adapt to new demand and supply patterns.

Model performance is evaluated with Mean Absolute Error (MAE). We believe that an MAE below 0.5 would have business value. In a real-world situation, we will also consider root mean square error (RMSE) adapting for the financial cost of large prediction errors. 

## Results ##
MAE in the hold-out dataset is 0.19, suggesting that the average difference between the actual gap and predicted gap is 0.19. Below is a snapshot of the live heatmap. 

![Alt text](results_actual.png?raw=true)

![Alt text](results_pred.png?raw=true)


