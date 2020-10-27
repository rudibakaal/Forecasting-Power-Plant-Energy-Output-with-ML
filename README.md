# Forecasting-Power-Plant-Energy-Output-with-ML

## Motivation
Machine learning regression algorithm used to predict power plant electrical energy output based on sensor readings.

The dataset contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant[1][2].

Features were standardised with sklearn's StandardScaler class. 

## Neural Network Topology and Results Summary

The root mean squared error (RMSE) loss function was leveraged along with the rmsprop optimizer for this regression problem.

![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

After 50 epochs, the training and validation set regressors reach a RMSE of 4.72 and 4.84 respectively, in predicting the electrical output of the plant. 

![power](https://user-images.githubusercontent.com/48378196/97298557-b201ce80-18a7-11eb-8899-6e5945711919.png)

## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1]  Pınar Tüfekci, Prediction of full load electrical power output of a base load operated combined cycle power plant using machine learning methods, International Journal of Electrical Power & Energy Systems, Volume 60, September 2014, Pages 126-140, ISSN 0142-0615 [2] Heysem Kaya, Pınar Tüfekci , Sadık Fikret Gürgen: Local and Global Learning Methods for Predicting Power of a Combined Gas & Steam Turbine, Proceedings of the International Conference on Emerging Trends in Computer and Electronics Engineering ICETCEE 2012, pp. 13-18 (Mar. 2012, Dubai)
