import matplotlib.pyplot as plt


def plot_stock(y_test, y_pred, dates, n_ahead, n_days=None):
    plt.figure(figsize=(16, 9))
    plt.title(f"Stock Prediction for {n_ahead} day(s) ahead", size=22)
    plt.ylabel("Stock Close Price", size=16)
    plt.xlabel("Date", size=16)
    plt.plot(dates, y_test[:n_days], c="blue", label="ground truth")
    plt.plot(dates, y_pred[:n_days], c="orange", label="predicted")
    plt.legend(fontsize=14)
    plt.show()
