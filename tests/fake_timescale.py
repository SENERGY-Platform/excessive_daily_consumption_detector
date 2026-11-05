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

from unittest import mock

import numpy as np
import pandas as pd

from main import TIMEZONE


class Timescale:
    # The history of the input that the operator reads at its start, as the library returns it:
    # one frame with the UTC time without a timezone and the meter reading.

    def __init__(self):
        self.rows = []

    def write(self, at, consumption):
        self.rows.append((pd.Timestamp(at).tz_convert("UTC").tz_localize(None), consumption))

    def write_all(self, readings):
        for at, consumption in readings:
            self.write(at, consumption)

    def patched(self):
        # the reader of the operator for an init within the with block
        frame = pd.DataFrame(self.rows, columns=["time", "Energy_Consumption"])
        return mock.patch("main.provide_historic_frames_local", return_value=[frame])


def meter(first_day, consumptions, every="15min"):
    # The readings of a meter, every 15 minutes of each German day from first_day on, as (UTC time, reading). On day i
    # it consumes consumptions[i] in kWh, evenly between the first and the last reading of the day.
    readings = []
    reading = 1000.0
    for i, consumption in enumerate(consumptions):
        day = pd.Timestamp(first_day) + pd.Timedelta(i, "d")
        # in absolute time, so that the day the clocks change has its 23 or 25 hours
        times = pd.date_range(day.tz_localize(TIMEZONE), (day + pd.Timedelta(1, "d")).tz_localize(TIMEZONE),
                              freq=every, inclusive="left")
        readings += [(at.tz_convert("UTC"), reading + step)
                     for at, step in zip(times, np.linspace(0, consumption, len(times)))]
        reading += consumption
    return readings
