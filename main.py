"""
   Copyright 2022 InfAI (CC SES)

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

__all__ = ("Operator", )

from operator_lib.util import OperatorBase, logger
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.helpers import provide_historic_frames_local, HistoryNeedsRayError
import pandas as pd
import numpy as np

import typing
import datetime

from operator_lib.util import Config, Inputs, Outputs

# The days are those of German time.
TIMEZONE = 'Europe/Berlin'
# A day is compared with the days of this length up to it. As much history is read at every start, and older days
# are dropped, so that a running operator compares with the same days as one just started.
HISTORY_LENGTH = pd.Timedelta(90, 'd')
# A day is excessive above the alarm level of the days before it, as in leakage-detection. The tail of those days
# begins above this quantile of them; a generalized Pareto distribution is fitted to it, and the level is the
# consumption that it exceeds with the probability that gives the false alarms per month of the config. Lower than
# the 90 % of leakage-detection, whose slots are many: 38 days hold a tail of 10 days, and the 89 days before a day 22.
TAIL_QUANTILE = 0.75
# A day whose days before have fewer in their tail is not judged.
MIN_TAIL = 10
# The shape of the tail is taken no lower than 0, an exponential tail: estimated from a few dozen days it comes out
# below 0 by chance, which caps the level near the largest day seen. Towards 0.5 its estimate no longer holds.
MAX_SHAPE = 0.45
# What the false alarms per month refer to.
DAYS_PER_MONTH = 30
# The days are compared to 1 Wh: days of an even consumption differ by the rounding of floating point numbers only,
# which would make a tail of a few nanowatt hours and an alarm of every larger day.
DECIMALS = 3


def alarm_level(consumptions, probability):
    # The consumption of a day that the days `consumptions` exceed with `probability`, or None if too few of them lie
    # in their tail.
    consumptions = np.round(np.asarray(consumptions, dtype=float), DECIMALS)
    if len(consumptions) == 0:
        return None
    threshold = np.quantile(consumptions, TAIL_QUANTILE)
    excesses = consumptions[consumptions > threshold] - threshold
    if len(excesses) < MIN_TAIL:
        return None
    shape = tail_shape(excesses)
    # the mean of the excesses is scale / (1 - shape)
    scale = excesses.mean() * (1 - shape)
    # the probability within the tail
    ratio = probability * len(consumptions) / len(excesses)
    if shape == 0:
        return float(threshold - scale * np.log(ratio))
    return float(threshold + scale / shape * (ratio ** -shape - 1))


def tail_shape(excesses):
    # The shape of a generalized Pareto distribution from the probability weighted moments of the excesses over its
    # threshold (Hosking and Wallis, 1987), between 0 and MAX_SHAPE.
    excesses = np.sort(excesses)
    n = len(excesses)
    a0 = excesses.mean()
    a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * excesses) / n
    if a0 - 2 * a1 <= 0:
        # excesses that are all alike: as light a tail as can be
        return 0.0
    return float(np.clip(2 - a0 / (a0 - 2 * a1), 0, MAX_SHAPE))


def config_number(value, default):
    # A flow that sets no value sends an empty string.
    return default if value == '' else float(value)


class CustomConfig(Config):
    device_name: str = ''
    false_alarms_per_month: float = 1
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "warning"

    def __init__(self, d, **kwargs):
        super().__init__(d, **kwargs)
        self.false_alarms_per_month = config_number(self.false_alarms_per_month, 1)
        if not self.false_alarms_per_month > 0:
            raise ValueError(f'false_alarms_per_month must be above 0, not {self.false_alarms_per_month}')

# The time of a reading is the time of its Kafka message, so there is no input for it.
class ConsumptionInputs(Inputs):
    Energy_Consumption: float

class ConsumptionOutputs(Outputs):
    value: str

class Operator(OperatorBase):
    configType = CustomConfig
    inputs = ConsumptionInputs
    outputs = ConsumptionOutputs

    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.device_name = self.config.device_name
        # The probability with which a day of usual consumption lies above the alarm level of the days before it.
        self.false_alarm_probability = self.config.false_alarms_per_month / DAYS_PER_MONTH

        # (day, consumption) per completed day, oldest first, and the readings of the day in progress. Like everything
        # derived from the readings not kept: rebuilt from the history at every start.
        self.daily_consumption_list = []
        self.consumption_same_day = []
        # The day of the first reading, which is not recorded: its readings begin during it.
        self.incomplete_day = None
        for at, reading in self.read_history():
            self.add_reading(at, reading)

    def read_history(self):
        # Timescale is the only source of the history: it is read at every start and extended live, so a restart or a
        # new volume loses nothing. Any other failure of the read fails the start, and the platform starts it again.
        try:
            frames = provide_historic_frames_local(HISTORY_LENGTH)
        except (HistoryNeedsRayError, MissingConfigValueError) as e:
            logger.warning(f'Starting without history: {e}')
            return []
        if len(frames) != 1 or 'device_id' in frames[0].columns:
            logger.warning('Starting without history: the consumption comes from more than one input or device')
            return []
        # As live: German time without a timezone, and without missing values.
        frame = frames[0].dropna(subset=['Energy_Consumption']).sort_values('time', kind='stable')
        times = pd.DatetimeIndex(pd.to_datetime(frame['time'], utc=True)).tz_convert(TIMEZONE).tz_localize(None)
        return zip(times, frame['Energy_Consumption'].astype(float))

    def add_reading(self, at, reading):
        # Adds the meter reading at `at`, German time without a timezone, and returns the day it completes, if one is
        # recorded.
        if not self.consumption_same_day:
            self.incomplete_day = at.date()
        elif at <= self.consumption_same_day[-1]['Energy_Time']:
            # A repeated or late reading, as Kafka delivers after a restart what the history holds already.
            return None
        completed = None
        if self.consumption_same_day and at.date() != self.consumption_same_day[-1]['Energy_Time'].date():
            completed = self.update_daily_consumption_list()
            self.consumption_same_day = []
        self.consumption_same_day.append({'Energy_Consumption': reading, 'Energy_Time': at})
        return completed

    def update_daily_consumption_list(self):
        readings = [datapoint['Energy_Consumption'] for datapoint in self.consumption_same_day]
        day = self.consumption_same_day[-1]['Energy_Time'].date()
        if day == self.incomplete_day:
            return None
        self.daily_consumption_list.append((day, max(readings)-min(readings)))
        while day - self.daily_consumption_list[0][0] >= HISTORY_LENGTH:
            self.daily_consumption_list.pop(0)
        return day

    def is_excessive(self):
        # Whether the last recorded day lies above the alarm level of the days before it; False while they have too few
        # in their tail.
        level = alarm_level([consumption for _, consumption in self.daily_consumption_list[:-1]], self.false_alarm_probability)
        excessive = level is not None and self.daily_consumption_list[-1][1] > level
        if excessive:
            logger.warning(f'Gestern wurde durch {self.device_name} ungewöhnlich viel Strom verbraucht.')
        return excessive

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone.
        timestamp = pd.Timestamp(timestamp).tz_convert(TIMEZONE).tz_localize(None)
        logger.debug('energy: '+str(data['Energy_Consumption'])+'  '+'time: '+str(timestamp))
        if pd.isna(data['Energy_Consumption']):
            # Left out, as in the history.
            return
        day = self.add_reading(timestamp, float(data['Energy_Consumption']))
        if day is None:
            return
        if self.is_excessive() and day == timestamp.date()-pd.Timedelta(1,'days'):
            return {'value': f'Am gestrigen Tag wurde übermäßig verbraucht.'} # Excessive daily consumption detected yesterday.
        return  # No excessive daily consumtion yesterday.

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="excessive-daily-consumption-detection-operator", git_info_file='git_commit')
