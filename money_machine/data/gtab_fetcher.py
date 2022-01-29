from money_machine.data.fetcher import Fetcher


class GtabFetcher(Fetcher):
    def fetch_data(self, kw_list, timeframe):
        self.requester.set_options(pytrends_config={"timeframe": timeframe})
        self.requester.create_anchorbank()
        return self.requester.new_query(kw_list[0])
