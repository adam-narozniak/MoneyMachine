from money_machine.trading.strategies import buy_on_min_roi


def simulate_trading(buying_rule, broker_account, stock, stock_pred, **kwargs):
    days = stock.dates
    for day in days[:-1]:
        current_price = stock.get_prices_for(day)
        prediction = stock_pred.get_prices_for_next(day)
        if buying_rule == buy_on_min_roi:
            decision = buying_rule(current_price, prediction, kwargs["roi"])
        else:
            decision = buying_rule(current_price, prediction)
        if decision:  # buy
            broker_account.buy_for_all_or_hold(day)
        else:  # sell
            broker_account.sell_all(day)
    broker_account.sell_all(days[-1])
