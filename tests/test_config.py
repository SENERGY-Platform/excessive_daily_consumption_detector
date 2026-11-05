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

from main import CustomConfig


class TestConfig(unittest.TestCase):
    def test_takes_one_false_alarm_per_month_when_it_is_missing(self):
        self.assertEqual(1.0, CustomConfig({}).false_alarms_per_month)

    def test_takes_one_false_alarm_per_month_when_it_is_empty(self):
        self.assertEqual(1.0, CustomConfig({"false_alarms_per_month": ""}).false_alarms_per_month)

    def test_takes_the_false_alarms_per_month_that_are_set(self):
        self.assertEqual(0.25, CustomConfig({"false_alarms_per_month": "0.25"}).false_alarms_per_month)

    def test_rejects_no_false_alarms_per_month(self):
        with self.assertRaises(ValueError):
            CustomConfig({"false_alarms_per_month": "0"})

    def test_rejects_negative_false_alarms_per_month(self):
        with self.assertRaises(ValueError):
            CustomConfig({"false_alarms_per_month": "-1"})


if __name__ == '__main__':
    unittest.main()
