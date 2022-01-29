from money_machine.data.fetcher import Fetcher


class PytrendsFetcher(Fetcher):
    def fetch_data(self, kw_list, timeframe):
        self.requester.build_payload(kw_list, timeframe=timeframe)
        return self.requester.interest_over_time()
