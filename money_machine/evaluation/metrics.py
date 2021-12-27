from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def calculate_all_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mape, mse
