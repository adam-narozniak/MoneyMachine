from abc import ABC, abstractmethod


class Fetcher(ABC):
    def __init__(self, requester):
        self.requester = requester

    @abstractmethod
    def fetch_data(self, keyword, timeframe):
        raise NotImplementedError
