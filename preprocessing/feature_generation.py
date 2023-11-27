import math
from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """
    Abstract class for feature generators.
    """

    def __init__(self, name, col_alias_mapping):
        self.name = name
        self.col_alias_mapping = col_alias_mapping

    def generate(self, data):
        if self.has_required_cols(data):
            self._generate(data)
        else:
            raise Exception("Missing required columns: %s" % self.get_required_cols())

    @abstractmethod
    def _generate(self, data):
        pass

    @abstractmethod
    def get_required_cols(self):
        pass

    def has_required_cols(self, data):
        cols = self.get_required_cols()
        for col in cols:
            if col in data.columns:
                continue
            aliases = self.col_alias_mapping[col]
            # return True if any alias is in data.columns
            if any(alias in data.columns for alias in aliases):
                continue
            return False
        return True

    def get_col_alias(self, data, col):
        if col in data.columns:
            return col
        if col not in self.col_alias_mapping:
            raise Exception("Column %s not found in col_alias_mapping" % col)
        aliases = self.col_alias_mapping[col]
        for alias in aliases:
            if alias in data.columns:
                return alias
        raise Exception("Column %s not found in data" % col)


class MidPriceGenerator(FeatureGenerator):
    def __init__(self, col_alias_mapping):
        super().__init__("mid_price", col_alias_mapping)

    def _generate(self, data):
        bid_price = self.get_col_alias(data, "bid_price")
        ask_price = self.get_col_alias(data, "ask_price")
        bid_volume = self.get_col_alias(data, "bid_volume")
        ask_volume = self.get_col_alias(data, "ask_volume")
        data[self.name] = (data[bid_price] * data[ask_volume] + data[ask_price] * data[bid_volume]) / (data[bid_volume] + data[ask_volume])
        data[self.name] = data[self.name].round(2)

    def get_required_cols(self):
        return ["bid_price", "ask_price", "bid_volume", "ask_volume"]


class SpreadGenerator(FeatureGenerator):
    def __init__(self, col_alias_mapping):
        super().__init__("spread", col_alias_mapping)

    def _generate(self, data):
        bid_price = self.get_col_alias(data, "bid_price")
        ask_price = self.get_col_alias(data, "ask_price")
        data[self.name] = data[ask_price] - data[bid_price]
        data[self.name] = data[self.name].round(2)

    def get_required_cols(self):
        return ["bid_price", "ask_price"]


class LogReturnGenerator(FeatureGenerator):
    def __init__(self, col_alias_mapping):
        super().__init__("log_return", col_alias_mapping)

    def _generate(self, data):
        mid_price = self.get_col_alias(data, "mid_price")
        data[self.name] = data[mid_price].apply(lambda x: math.log(x)) - data[mid_price].shift(1).apply(lambda x: math.log(x))
        data[self.name] = data[self.name].round(6).apply(lambda x: 0.0 if x == -0.0 else x)

    def get_required_cols(self):
        return ["mid_price"]


class FeatureFactory:
    def __init__(self, col_alias_mapping):
        self.col_alias_mapping = col_alias_mapping
        self.generators = {
            "mid_price": MidPriceGenerator(col_alias_mapping),
            "spread": SpreadGenerator(col_alias_mapping),
            "log_return": LogReturnGenerator(col_alias_mapping)
        }

    def __has_feature(self, data, col):
        if col in data.columns:
            return True
        for alias in self.col_alias_mapping[col]:
            if alias in data.columns:
                return True
        return False

    def __get_generator(self, name):
        if name not in self.generators:
            raise Exception("Feature generator %s not found" % name)
        return self.generators[name]

    def apply_feature(self, data, name):
        stack = [name]
        while stack:
            name = stack.pop()
            if self.__has_feature(data, name):
                continue
            generator = self.__get_generator(name)
            if generator.has_required_cols(data):
                generator.generate(data)
            else:
                required_cols = generator.get_required_cols()
                stack.append(name)
                for col in required_cols:
                    stack.append(col)

    def apply_features(self, data, names):
        for name in names:
            self.apply_feature(data, name)
            print("Applied feature %s" % name)


# main
if __name__ == "__main__":
    # load col_aliases.json as a dict
    import json
    with open("col_aliases.json", "r") as f:
        mapping = json.load(f)

    df = pd.DataFrame({
        "bid_price": [100, 150],
        "ask_price": [110, 160],
        "bid_volume": [10, 20],
        "ask_volume": [5, 10]
    })

    factory = FeatureFactory(mapping)
    factory.apply_features(df, ["spread", "log_return"])
    print(df)
