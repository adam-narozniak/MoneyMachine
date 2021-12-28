import matplotlib.pyplot as plt


def plot_cumulative_value(broker_account):
    plt.figure(figsize=(16, 9))
    plt.title("Cumulative value history", size=22)
    plt.xlabel("Date", size=16)
    plt.ylabel("Dollars", size=16)
    plt.plot(broker_account.stock.dates,
             broker_account.cumulative_val_hist,
             color="blue",
             label="total value")
    plt.axhline(broker_account.initial_amount,
                color="red",
                label="initial value",
                linestyle="--")
    plt.legend(fontsize=14)
    plt.show()
