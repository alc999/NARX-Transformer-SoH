import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error

def report_metrics(x,  x_hat):
    MSE = mean_squared_error(x, x_hat)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(x, x_hat)
    MAX_ERR = max_error(x, x_hat)
    return MSE, RMSE, MAE, MAX_ERR
