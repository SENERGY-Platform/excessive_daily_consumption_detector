"""
   Copyright 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from fake_timescale import Timescale, meter
from main import CustomConfig, Operator

EXCESSIVE = {"value": "Am gestrigen Tag wurde übermäßig verbraucht."}


def ordinary(days, seed=1):
    # daily consumptions of a household, around 8 kWh
    return list(np.round(8 + np.random.default_rng(seed).normal(0, 0.5, days), 3))


class TestJudgement(unittest.TestCase):
    def judge(self, consumptions):
        # The output of the first reading after the given days, which Timescale holds. A day before them is in
        # Timescale too, since the day the history begins in is not recorded.
        readings = meter("2026-06-01", [1.0] + consumptions)
        timescale = Timescale()
        timescale.write_all(readings)
        operator = Operator()
        with timescale.patched(), warnings.catch_warnings():
            # kneed warns when it finds no knee
            warnings.simplefilter("ignore")
            operator.init(None, None, None, None, None, None, config=CustomConfig({"device_name": "Zähler"}))
            last_at, last_reading = readings[-1]
            return operator.run({"Energy_Consumption": last_reading}, "", "device", last_at + pd.Timedelta(15, "min"))

    def test_reports_a_day_far_above_the_others(self):
        self.assertEqual(EXCESSIVE, self.judge(ordinary(40) + [20.0]))

    def test_does_not_report_an_ordinary_day(self):
        self.assertIsNone(self.judge(ordinary(40) + [8.0]))

    def test_judges_from_the_24th_day_on(self):
        self.assertIsNone(self.judge(ordinary(22) + [20.0]))
        self.assertEqual(EXCESSIVE, self.judge(ordinary(23) + [20.0]))

    def test_does_not_report_a_device_that_consumes_nothing(self):
        # Every day the same: kneed finds no knee.
        self.assertIsNone(self.judge([0.0] * 41))

    def test_reports_the_first_consumption_of_a_device_that_consumed_nothing_before(self):
        # kneed finds the knee at 0.
        self.assertEqual(EXCESSIVE, self.judge([0.0] * 40 + [5.0]))


if __name__ == '__main__':
    unittest.main()
