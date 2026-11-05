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

import numpy as np
import pandas as pd

from fake_timescale import Timescale, meter
from main import CustomConfig, Operator

EXCESSIVE = {"value": "Am gestrigen Tag wurde übermäßig verbraucht."}


def ordinary(days, seed=1):
    # daily consumptions of a household, around 8 kWh
    return list(np.round(8 + np.random.default_rng(seed).normal(0, 0.5, days), 3))


class TestJudgement(unittest.TestCase):
    def judge(self, consumptions, false_alarms_per_month="1"):
        # The output of the first reading after the given days, which Timescale holds. A day before them is in
        # Timescale too, since the day the history begins in is not recorded.
        readings = meter("2026-06-01", [1.0] + consumptions)
        timescale = Timescale()
        timescale.write_all(readings)
        operator = Operator()
        with timescale.patched():
            operator.init(None, None, None, None, None, None,
                          config=CustomConfig({"device_name": "Zähler", "false_alarms_per_month": false_alarms_per_month}))
        last_at, last_reading = readings[-1]
        return operator.run({"Energy_Consumption": last_reading}, "", "device", last_at + pd.Timedelta(15, "min"))

    def test_reports_a_day_far_above_the_others(self):
        self.assertEqual(EXCESSIVE, self.judge(ordinary(60) + [20.0]))

    def test_does_not_report_an_ordinary_day(self):
        self.assertIsNone(self.judge(ordinary(60) + [8.0]))

    def test_judges_once_ten_of_the_days_before_lie_in_their_tail(self):
        # Above the 75 % quantile of 37 days lie 9 of them, of 38 days 10.
        self.assertIsNone(self.judge(ordinary(37) + [20.0]))
        self.assertEqual(EXCESSIVE, self.judge(ordinary(38) + [20.0]))

    def test_reports_a_day_less_readily_with_fewer_false_alarms_per_month(self):
        # 10 kWh lies above the alarm level of these days at one false alarm a month, below it at one in 100 months.
        self.assertEqual(EXCESSIVE, self.judge(ordinary(60) + [10.0], "1"))
        self.assertIsNone(self.judge(ordinary(60) + [10.0], "0.01"))

    def test_does_not_judge_a_device_that_consumed_nothing_before(self):
        # Without consumption no day before lies in their tail.
        self.assertIsNone(self.judge([0.0] * 60 + [5.0]))


if __name__ == '__main__':
    unittest.main()
