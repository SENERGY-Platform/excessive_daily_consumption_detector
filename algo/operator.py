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

import util
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import kneed
import os
from itertools import chain

class Operator(util.OperatorBase):
    def __init__(self, device_id, data_path, device_name='das Gerät'):
        if not os.path.exists(data_path):
            os.mkdir(data_path)

        self.device_id = device_id
        self.device_name = device_name

        self.daily_consumption_list = []

        self.consumption_same_day = []

        self.model_file_path = f'{data_path}/{self.device_id}_model.pt'

    def todatetime(self, timestamp):
        if str(timestamp).isdigit():
            if len(str(timestamp))==13:
                return pd.to_datetime(int(timestamp), unit='ms')
            elif len(str(timestamp))==19:
                return pd.to_datetime(int(timestamp), unit='ns')
        else:
            return pd.to_datetime(timestamp)

    def update_daily_consumption_list(self):
        min_index = np.argmin([float(datapoint['Energy_Consumption']) for datapoint in self.consumption_same_day])
        max_index = np.argmax([float(datapoint['Energy_Consumption']) for datapoint in self.consumption_same_day])
        day_consumption_max = float(self.consumption_same_day[max_index]['Energy_Consumption'])
        day_consumption_min = float(self.consumption_same_day[min_index]['Energy_Consumption'])
        #day_consumption_max_time = self.todatetime(self.consumption_same_day[max_index]['energy_time']).tz_localize(None)
        #day_consumption_min_time = self.todatetime(self.consumption_same_day[min_index]['energy_time']).tz_localize(None)
        overall_daily_consumption = day_consumption_max-day_consumption_min
        day = self.todatetime(self.consumption_same_day[-1]['Energy_Time']).tz_localize(None).date()
        self.daily_consumption_list.append((day, overall_daily_consumption))
        return

    def determine_epsilon(self):
        neighbors = NearestNeighbors(n_neighbors=10)
        neighbors_fit = neighbors.fit(np.array([daily_consumption for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        distances, _ = neighbors_fit.kneighbors(np.array([daily_consumption for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        distances = np.sort(distances, axis=0)
        distances_x = distances[:,1]
        kneedle = kneed.KneeLocator(np.linspace(0,1,len(distances_x)), distances_x, S=0.9, curve="convex", direction="increasing")
        epsilon = kneedle.knee_y
        return epsilon

    def create_clustering(self, epsilon):
        daily_consumption_clustering = DBSCAN(eps=epsilon, min_samples=10).fit(np.array([daily_consumption 
                                                                     for _, daily_consumption in self.daily_consumption_list]).reshape(-1,1))
        return daily_consumption_clustering.labels_
    
    def test_daily_consumption(self, clustering_labels):
        anomalous_indices = np.where(clustering_labels==clustering_labels.min())
        median = np.median([daily_consumption for _, daily_consumption in self.daily_consumption_list])
        anomalous_indices_high = [i for i in anomalous_indices if self.daily_consumption_list[i][1] > median]
        if len(self.consumption_same_day)-1 in anomalous_indices:
            print(f'Gestern wurde durch {self.device_name} ungewöhnlich viel Strom verbraucht.')
        return [self.daily_consumption_list[i] for i in anomalous_indices_high]
    
    def run(self, data, selector='energy_func'):
        if os.getenv("DEBUG") is not None and os.getenv("DEBUG").lower() == "true":
            print(selector + ": " + 'energy: '+str(data['Energy_Consumption'])+'  '+'time: '+str(self.todatetime(data['Energy_Time']).tz_localize(None)))
        if self.consumption_same_day == []:
            self.consumption_same_day.append(data)
            return
        elif self.consumption_same_day != []:
            if self.todatetime(data['Energy_Time']).tz_localize(None).date()==self.todatetime(self.consumption_same_day[-1]['Energy_Time']).tz_localize(None).date():
                self.consumption_same_day.append(data)
                return
            else:
                self.update_daily_consumption_list()
                if len(self.daily_consumption_list) >= 50:
                    epsilon = self.determine_epsilon()
                    clustering_labels = self.create_clustering(epsilon)
                    days_with_excessive_consumption = self.test_daily_consumption(clustering_labels)                    
                    self.consumption_same_day = [data]
                    if pd.Timestamp.now().date()-pd.Timedelta(1,'days') in list(chain.from_iterable(days_with_excessive_consumption)):
                        return {'value': 1} # Excessive daily consumption detected yesterday.
                    else:
                        return {'value': 0} # No excessive daily consumtion yesterday.
                else:
                    return
