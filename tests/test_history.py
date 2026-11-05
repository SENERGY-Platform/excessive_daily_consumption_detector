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

import datetime
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import HistoryNeedsRayError

from fake_timescale import Timescale, meter
from main import CustomConfig, Operator

EXCESSIVE = {"value": "Am gestrigen Tag wurde übermäßig verbraucht."}


def day(text):
    return datetime.date.fromisoformat(text)


def varied(days):
    # daily consumptions around 8 kWh
    return list(np.round(8 + np.random.default_rng(1).normal(0, 0.5, days), 3))


class TestHistory(unittest.TestCase):
    def setUp(self):
        self.timescale = Timescale()
        # kneed warns when it finds no knee
        catcher = warnings.catch_warnings()
        catcher.__enter__()
        self.addCleanup(catcher.__exit__, None, None, None)
        warnings.simplefilter("ignore")

    def start(self, reader=None):
        # an operator that reads its history from self.timescale unless another reader is given
        operator = Operator()
        with reader or self.timescale.patched() as self.read:
            operator.init(None, None, None, None, None, None, config=CustomConfig({"device_name": "Zähler"}))
        return operator

    def live(self, operator, readings):
        return [result for at, reading in readings
                if (result := operator.run({"Energy_Consumption": reading}, "", "device", at)) is not None]

    def days(self, operator):
        return [(d, round(consumption, 6)) for d, consumption in operator.daily_consumption_list]

    def test_reads_the_last_90_days_from_timescale(self):
        self.start()
        self.read.assert_called_once_with(pd.Timedelta(90, "d"))

    def test_records_the_complete_days_of_the_history(self):
        # The history begins in the morning of June 1, so that day is not complete; June 4 is in progress.
        self.timescale.write_all(meter("2026-06-01", [5.0, 6.0, 7.0, 8.0])[40:-50])
        self.assertEqual([(day("2026-06-02"), 6.0), (day("2026-06-03"), 7.0)], self.days(self.start()))

    def test_derives_the_same_days_from_the_history_as_from_live_readings(self):
        readings = meter("2026-06-01", varied(30))
        running = self.start()
        self.live(running, readings)
        self.timescale.write_all(readings)
        self.assertEqual(self.days(running), self.days(self.start()))

    def test_continues_the_day_a_restart_cuts_through(self):
        readings = meter("2026-06-01", [5.0, 6.0, 7.0])
        running = self.start()
        self.live(running, readings)
        self.timescale.write_all(readings[:150])
        restarted = self.start()
        self.live(restarted, readings[150:])
        self.assertEqual(self.days(running), self.days(restarted))

    def test_leaves_out_the_readings_that_the_history_holds_already(self):
        readings = meter("2026-06-01", [5.0, 6.0, 7.0])
        running = self.start()
        self.live(running, readings)
        self.timescale.write_all(readings[:150])
        restarted = self.start()
        # Kafka delivers again what came before the restart.
        self.live(restarted, readings[100:])
        self.assertEqual(self.days(running), self.days(restarted))

    def test_compares_with_the_last_90_days(self):
        operator = self.start()
        self.live(operator, meter("2026-05-01", varied(120)))
        days = [d for d, _ in operator.daily_consumption_list]
        # August 28 is in progress.
        self.assertEqual((day("2026-05-30"), day("2026-08-27"), 90), (days[0], days[-1], len(days)))

    def test_compares_a_restarted_operator_with_the_same_days_as_a_running_one(self):
        readings = meter("2026-05-01", varied(100))
        restart = pd.Timestamp("2026-08-05 12:00", tz="Europe/Berlin")
        running = self.start()
        self.live(running, readings)
        # Timescale returns what the library reads at the restart: the last 90 days.
        self.timescale.write_all([(at, reading) for at, reading in readings if restart - pd.Timedelta(90, "d") < at <= restart])
        restarted = self.start()
        self.live(restarted, [(at, reading) for at, reading in readings if at > restart])
        self.assertEqual(self.days(running), self.days(restarted))

    def test_judges_the_first_day_after_the_start_with_enough_history(self):
        readings = meter("2026-06-01", varied(31) + [20.0])
        # The start in the evening of the excessive day.
        self.timescale.write_all(readings[:-20])
        results = self.live(self.start(), readings[-20:] + [(readings[-1][0] + pd.Timedelta(15, "min"), readings[-1][1])])
        self.assertEqual([EXCESSIVE], results)

    def test_counts_the_days_in_german_time(self):
        # 22:30 UTC on June 1 is 00:30 on June 2 in German summer time.
        readings = [(pd.Timestamp(utc, tz="UTC"), reading) for utc, reading in [("2026-06-01 10:00", 100.0),
                                                                                ("2026-06-01 21:30", 101.0),
                                                                                ("2026-06-01 22:30", 105.0),
                                                                                ("2026-06-02 21:30", 106.5),
                                                                                ("2026-06-02 22:30", 107.0)]]
        with self.subTest("history"):
            timescale = Timescale()
            timescale.write_all(readings)
            self.assertEqual([(day("2026-06-02"), 1.5)], self.days(self.start(timescale.patched())))
        with self.subTest("live"):
            operator = self.start()
            self.live(operator, readings)
            self.assertEqual([(day("2026-06-02"), 1.5)], self.days(operator))

    def test_keeps_the_whole_consumption_of_the_day_the_clocks_go_back(self):
        # On October 25, 2026, the clocks go back from 03:00 to 02:00, so the day has 25 hours and the hour from 02:00
        # comes twice.
        self.timescale.write_all(meter("2026-10-24", [5.0, 6.0, 7.0]))
        self.assertEqual([(day("2026-10-25"), 6.0)], self.days(self.start()))

    def test_leaves_out_missing_values(self):
        readings = meter("2026-06-01", [5.0, 6.0, 7.0])
        with self.subTest("history"):
            timescale = Timescale()
            timescale.write_all(readings[:120] + [(readings[120][0], None)] + readings[121:])
            self.assertEqual([(day("2026-06-02"), 6.0)], self.days(self.start(timescale.patched())))
        with self.subTest("live"):
            operator = self.start()
            self.live(operator, readings[:120] + [(readings[120][0], None), (readings[120][0], float("nan"))] + readings[121:])
            self.assertEqual([(day("2026-06-02"), 6.0)], self.days(operator))

    def test_starts_without_history_where_timescale_does_not_have_it(self):
        for error in [HistoryNeedsRayError("the history of input topic t (d) is only in Kafka"),
                      MissingConfigValueError("cannot read history for topic t")]:
            with self.subTest(error=type(error).__name__):
                operator = self.start(mock.patch("main.provide_historic_frames_local", side_effect=error))
                self.live(operator, meter("2026-06-01", [5.0, 6.0, 7.0]))
                self.assertEqual([(day("2026-06-02"), 6.0)], self.days(operator))

    def test_starts_without_history_for_more_than_one_device(self):
        frame = pd.DataFrame({"time": [pd.Timestamp("2026-06-01 08:00"), pd.Timestamp("2026-06-02 08:00")],
                              "Energy_Consumption": [100.0, 200.0], "device_id": ["a", "b"]})
        operator = self.start(mock.patch("main.provide_historic_frames_local", return_value=[frame]))
        self.assertEqual([], operator.consumption_same_day)

    def test_does_not_start_when_timescale_cannot_be_read(self):
        with self.assertRaises(ConnectionError):
            self.start(mock.patch("main.provide_historic_frames_local", side_effect=ConnectionError("no connection")))


if __name__ == '__main__':
    unittest.main()
