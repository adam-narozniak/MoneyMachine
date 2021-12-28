class BrokerAccount:

    def __init__(self, stock, initial_amount=10):
        self.stock = stock
        self.initial_amount = initial_amount
        self.account_balance = initial_amount
        self.stock_amount = 0

        self.account_hist = []
        self.stock_hist = []
        self.cumulative_val_hist = []

    def add_history(self, current_day):
        self.account_hist.append(self.account_balance)
        self.stock_hist.append(self.stock_amount)
        self.cumulative_val_hist.append(
            self.account_hist[-1] + self.stock.get_prices_for(current_day) * self.stock_hist[-1])

    def buy_for_x(self, x, current_day):
        if self.account_balance < x:
            raise Exception(f"The amount of money is not sufficient")
        self.stock_amount += x / self.stock.get_prices_for(current_day)
        self.account_balance -= x
        self.add_history(current_day)

    def buy_for_x_or_hold(self, x, current_day):
        if self.account_balance < x:
            self.add_history(current_day)
        else:
            self.buy_for_x(x, current_day)

    def buy_for_all_or_hold(self, current_day):
        self.buy_for_x_or_hold(self.account_balance, current_day)

    def buy_x_stocks(self, x, current_day):
        price_to_pay = x * self.stock.get_prices_for(current_day)
        if self.account_balance < price_to_pay:
            raise Exception(f"The amount of money is not sufficient")
        self.stock_amount += x
        self.account_balance -= price_to_pay
        self.add_history(current_day)

    def sell_all(self, current_day):
        money = self.stock.get_prices_for(current_day) * self.stock_amount
        self.account_balance += money
        self.stock_amount = 0
        self.add_history(current_day)

    def _calculate_roi(self):
        return (self.account_balance - self.initial_amount) / self.initial_amount * 100

    def print_results(self):
        print(f"Account balance: {self.account_balance:.2f} dollars")
        print(f"ROI: {self._calculate_roi():.2f} %")

