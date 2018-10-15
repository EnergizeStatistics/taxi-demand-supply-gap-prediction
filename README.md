# Taxi Demand-Supply Gap Prediction #

## Description ##
Both dedicated taxi services and ride-sharing (e.g., Uber) companies find it cruical to forecast demand of trips for a particular location. Demand-supply prediction enables these companies to maximize driver utilization while ensuring that riders have service available in times of need. 

I hope to build a demand-supply gap prediction that allows such a service to guide drivers to specific, granular geographic areas. I have seen many similar analyses that use time series algorithms, such as RNN and ARIMA. Such algorithms are applicable when the number of trips is relatively high at all time units, e.g., 10 trips on day 1, then 20 trips on day 2, then 15 trips on day 3, and so on. In contrary, my prediction looks at the demand and supply in a small geographic area (grid size is approximately 1.1 km * 0.84 km) in a short moving time window (30 minutes). In most of these time windows, both the demand and supply are zero. For this consideration I opt for supervised learning algorithms. 

Data used in this project is provided by Kaggle (https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i). The dataset describes a complete year of the paths for taxis running in the city of Porto, Portugal. Particularly it contains GPS coordinates of the paths and starting time of each trip. 

## Method ##
The dataset only has starting time of each trip available. 
I trained a gradient boosting decision tree 


We intent to build Go-Search, a live heatmap which shows live prediction of supply-demand gap across area in the city. By viewing the heatmap, drivers can search which area that will need their service the most. Therefore, they can make decision to reposition themselves before the demands arises and maximize their revenues. In this way, the supply will be ready for the demand in advance so the waiting time can be minimized and the surge pricing never even has to happen.

In order to achieve this, we attempt build a predictive model powered by machine learning techniques that analyze historical demand pattern and current drivers position to predict the the supply and demand gap, 1 hour in advance. This model will serve as a backend to a city heatmap showing a supply-demand gap prediction for the next hour.


