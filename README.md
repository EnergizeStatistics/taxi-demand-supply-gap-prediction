# Taxi Demand-Supply Gap Prediction #

## Description ##
Both dedicated taxi services and ride-sharing (e.g., Uber) companies find it cruical to forecast demand for user trips for a particular location. Demand-supply prediction enables these companies to maximize driver utilization. On the other hand, the companies also hope to ensure that riders have service available in times of need. 

I hope to build a demand-supply gap prediction that allows us to guide drivers to specific, granular geographic areas. I have seen many predictions that use time series alrightms, such as RNN and ARIMA. Such algorithms are applicable when the number of trips is relatively high at all time units, e.g., 10 Uber trips on day 1, then 20 Uber trips on day 2, then 15 trips on day 3, and so on. In contrary, this prediction looks at the demand and supply in small geographic areas () in a short moving time window (10 minutes) 


In this notebook we look at a classical algorithm(ARIMA) which can be used to predict the demand for user trips in the upcoming week for a particular location. Particularly, we will be utilizing Uber's 2014 user trips data of New York city, to accomplish the same.

The dataset can be found on kaggle.com(https://www.kaggle.com/fivethirtyeight/uber-pickups-in-new-york-city)



Disclaimer: This data set has been compiled from online available data source, a big thanks to its contributor.
https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i