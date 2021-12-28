"""
Module to represent trading strategies.

Conventions:
    *Functions return True => buy (or keep); False => sell

"""


def baseline_strategy(current_price, prediction):
    """
    Buy if the prediction is predicted to be bigger, sell if the prediction is lower than current stock value.

    Works with 1-day ahead prediction. In this case you go all in.

    Returns:
        True if it you should buy (hold), False if you should sell.

    """
    if prediction > current_price:
        return True
    else:
        return False


def buy_on_min_roi(current_price, prediction, roi=0.1):
    """Buy a stock only if the potential roi is `roi`."""
    if prediction >= current_price * (1 + roi):
        return True
    else:
        return False
