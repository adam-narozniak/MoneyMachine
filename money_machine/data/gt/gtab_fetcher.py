import math
import re

from money_machine.data.gt.fetcher import Fetcher
import pathlib


class GtabFetcher(Fetcher):
    def __init__(self, requester, setup_path):
        """
        Args:
            requester:
            setup_path: The path where the directories will be (or are) created  for GTAB to run. It also saves there
                the anchorbanks.
        """
        if requester.dir_path != setup_path:
            raise Exception("The path of GTAB and setup path have to be the same")
        super().__init__(requester)
        self._setup_path = pathlib.Path(setup_path)
        self._anchor_bank_path = self._setup_path / "output/google_anchorbanks"

    def fetch_data(self, kw_list, timeframe, reuse_anchor=True, anchorbank_id=None):
        """
        If there is no anchorbank for given period fetch for it. Otherwise
        Args:
            anchorbank_id: is None the first anchrobank with the smallest id will be used
            kw_list:
            timeframe:
            reuse_anchor: if True then if the file is save with corresponing name

        Returns:

        """
        id_to_use = -1
        if reuse_anchor:
            if anchorbank_id is None:
                anchorbank_id = self.find_min_anchor_id(timeframe)
            if anchorbank_id == math.inf:
                print("no saved anchorbank found")
                reuse_anchor = False
            else:
                # set the avtive anchrobank to the found one
                id_to_use = anchorbank_id

        if reuse_anchor is False:
            max_id = int(self.find_max_anchor_id(timeframe))
            id_to_use = max_id + 1
            self.create_anchorbank(timeframe, id_to_use)

        anchor_name = f"google_anchorbank_geo=_timeframe={timeframe}_id={id_to_use}.tsv"
        self.requester.set_active_gtab(anchor_name)

        return self.requester.new_query(kw_list[0])

    def find_max_anchor_id(self, timeframe):
        """
        Each anchorbank is saved with the _id=x postfix. It's done in order to be able to have many anchors for the
            same time (google give only sample of the whole population).
        Args:
            timeframe:

        Returns:

        """

        search_pattern = f"google_anchorbank_geo=_timeframe={timeframe}_id=([0-9]*).tsv"
        idx_max = -1
        # find other anchorbank for that period
        anchorbanks = [file for file in self._anchor_bank_path.iterdir()]
        for anchorbank in anchorbanks:
            name = anchorbank.name
            match = re.search(search_pattern, name)
            if match is not None:
                found_id = int(match.group(1))
                idx_max = max(idx_max, found_id)
        return idx_max

    def find_min_anchor_id(self, timeframe):
        """

        Args:
            timeframe:

        Returns:

        """

        search_pattern = f"google_anchorbank_geo=_timeframe={timeframe}_id=([0-9]*).tsv"
        idx_min = math.inf
        # find other anchorbank for that period
        anchorbanks = [file for file in self._anchor_bank_path.iterdir()]
        for anchorbank in anchorbanks:
            name = anchorbank.name
            match = re.search(search_pattern, name)
            if match is not None:
                found_id = int(match.group(1))
                idx_min = min(idx_min, found_id)
        return idx_min

    def create_anchorbank(self, timeframe, idx=None):
        """
        Anchorbank is created even if there is already on present for that period.
        In order to avoid overwriting the achorbanks after the creation the anchorbank is reanmed, the id is added at the end.
        Args:
            timeframe:
            idx: is the id that will be appended to the saved name of anchorbank
        """
        if idx is None:
            idx = self.find_max_anchor_id(timeframe) + 1
        original_anchor_name = f"google_anchorbank_geo=_timeframe={timeframe}.tsv"
        # idx_max = self.find_max_anchor_id(timeframe)
        self.requester.set_options(pytrends_config={"timeframe": timeframe})
        self.requester.create_anchorbank()
        anchor_path = self._anchor_bank_path / original_anchor_name
        anchor_path.rename(pathlib.Path(anchor_path.parent, anchor_path.stem + f"_id={idx}" + ".tsv"))


if __name__ == "__main__":
    import gtab
    import datetime as dt
    from money_machine.data.gt.gt_data import create_timeframe_from_datetime

    setup_path = "./data/gtab/bitcoin"
    t = gtab.GTAB()
    gtab_fetcher = GtabFetcher(t, setup_path)
    current_end_date = dt.date.today()
    current_start_date = current_end_date - dt.timedelta(100)
    timeframe = create_timeframe_from_datetime(current_start_date, current_end_date)
    idx = 4
    gtab_fetcher.create_anchorbank(timeframe, idx)
