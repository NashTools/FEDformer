import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_provider import data_loader


class TestNormalizeFunction(unittest.TestCase):
    df = pd.DataFrame({
        "bid_price": [0, 150, 200, 250, 300, 350, 400, 450, 500, 550],
        "ask_price": [0, 160, 210, 260, 310, 360, 410, 460, 510, 560],
        "bid_volume": [0, 20, 20, 30, 30, 40, 40, 50, 50, 60],
        "ask_volume": [0, 10, 10, 15, 15, 20, 20, 25, 25, 30],
        "id": [0, 9, 30, 30, 35, 60, 99, 102, 222, 245],
        "log_return": [0.0, 0.0, 0.3, 0.4, 0.5, 0.6, -0.1, 0.8, -0.9, 1.0]
    })

    def test_known_input_scaling(self):
        expected_scaled_data = StandardScaler().fit_transform(self.df.values)
        scaled_data, zero_scaled = data_loader.normalize(self.df)
        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)
        np.testing.assert_array_almost_equal(scaled_data[0], zero_scaled)


if __name__ == '__main__':
    unittest.main()
