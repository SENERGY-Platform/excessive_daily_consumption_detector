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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed

import typing
import datetime

from operator_lib.util import Config, Inputs, Outputs

# The days are those of German time.
TIMEZONE = 'Europe/Berlin'
# A day is compared with the days of this length up to it. As much history is read at every start, and older days
# are dropped, so that a running operator compares with the same days as one just started.
HISTORY_LENGTH = pd.Timedelta(90, 'd')
# The days needed before a day is judged.
MIN_DAYS = 24


class CustomConfig(Config):
    device_name: str = ''
    # A field of the library, declared again so that flows can set it.
    logger_level: str = "warning"

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

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array([daily_consumption for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([daily_consumption for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        if not epsilon:
            # No knee, or one at 0 as when nearly every day consumed the same: DBSCAN needs an epsilon above 0.
            return 1
        return epsilon

    def create_clustering(self, epsilon):
        daily_consumption_clustering = DBSCAN(eps=epsilon, min_samples=10).fit(np.array([daily_consumption
                                                                     for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        return daily_consumption_clustering.labels_

    def test_daily_consumption(self, clustering_labels):
        # The days that DBSCAN leaves out of every cluster and that lie above the 95 % quantile of all days.
        anomalous_indices = np.where(clustering_labels==-1)[0]
        quantile = np.quantile([daily_consumption for _, daily_consumption in self.daily_consumption_list],0.95)
        anomalous_indices_high = [i for i in anomalous_indices if self.daily_consumption_list[i][1] > quantile]
        if len(self.daily_consumption_list)-1 in anomalous_indices_high:
            logger.warning(f'Gestern wurde durch {self.device_name} ungewöhnlich viel Strom verbraucht.')
        return [self.daily_consumption_list[i] for i in anomalous_indices_high]

    def run(self, data: typing.Dict[str, typing.Any], selector: str, device_id, timestamp: datetime.datetime):
        # Convert to german time and then forget the timezone.
        timestamp = pd.Timestamp(timestamp).tz_convert(TIMEZONE).tz_localize(None)
        logger.debug('energy: '+str(data['Energy_Consumption'])+'  '+'time: '+str(timestamp))
        if pd.isna(data['Energy_Consumption']):
            # Left out, as in the history.
            return
        day = self.add_reading(timestamp, float(data['Energy_Consumption']))
        if day is None or len(self.daily_consumption_list) < MIN_DAYS:
            return
        epsilon = self.determine_epsilon()
        clustering_labels = self.create_clustering(epsilon)
        days_with_excessive_consumption = self.test_daily_consumption(clustering_labels)
        if timestamp.date()-pd.Timedelta(1,'days') in [day for day, _ in days_with_excessive_consumption]:
            return {'value': f'Am gestrigen Tag wurde übermäßig verbraucht.'} # Excessive daily consumption detected yesterday.
        return  # No excessive daily consumtion yesterday.

from operator_lib.operator_lib import OperatorLib
if __name__ == "__main__":
    OperatorLib(Operator(), name="excessive-daily-consumption-detection-operator", git_info_file='git_commit')
