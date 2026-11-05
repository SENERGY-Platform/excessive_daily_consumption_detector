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

import math
import unittest

from main import DAYS_PER_MONTH, alarm_level

# one false alarm per month
PROBABILITY = 1 / DAYS_PER_MONTH
# the days before a day in the last 90 days
DAYS = 89


def exponential(scale):
    # the consumptions of 89 days that are distributed exponentially, at their quantiles
    return [-scale * math.log(1 - (i + 0.5) / DAYS) for i in range(DAYS)]


class TestAlarmLevel(unittest.TestCase):
    def test_puts_the_level_of_an_exponential_consumption_near_the_quantile_of_the_false_alarms(self):
        # an exponential consumption with a mean of 10 kWh exceeds 10 * ln(30) once a month
        expected = -10 * math.log(PROBABILITY)
        self.assertAlmostEqual(expected, alarm_level(exponential(10), PROBABILITY), delta=0.1 * expected)

    def test_raises_the_level_for_fewer_false_alarms(self):
        self.assertGreater(alarm_level(exponential(10), PROBABILITY / 4), alarm_level(exponential(10), PROBABILITY))

    def test_puts_the_level_of_a_bounded_consumption_above_its_largest_day_for_rare_false_alarms(self):
        # evenly between 0 and 10 kWh; a tail shape below 0 would cap the level at about 10
        uniform = [10 * (i + 0.5) / DAYS for i in range(DAYS)]
        self.assertGreater(alarm_level(uniform, PROBABILITY / 4), 11)

    def test_has_no_level_with_nine_days_in_the_tail(self):
        self.assertIsNone(alarm_level(list(range(37)), PROBABILITY))

    def test_has_a_level_with_ten_days_in_the_tail(self):
        self.assertIsNotNone(alarm_level(list(range(38)), PROBABILITY))

    def test_has_a_level_above_a_tail_whose_days_are_all_alike(self):
        # mostly nothing, on a quarter of the days 6 kWh: the tail is those days
        level = alarm_level([0.0] * 67 + [6.0] * 22, PROBABILITY)
        self.assertTrue(6 < level < math.inf, level)

    def test_has_no_level_for_days_that_differ_by_floating_point_rounding_only(self):
        even = [8.0 + (i % 7) * 1e-11 for i in range(DAYS)]
        self.assertIsNone(alarm_level(even, PROBABILITY))


if __name__ == '__main__':
    unittest.main()
