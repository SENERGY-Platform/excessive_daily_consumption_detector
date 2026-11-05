# Excessive Daily Consumption Detector

Reports a day whose electricity consumption is far higher than on the other days of the last 90 days.

## How it works

- **Days:** The consumption of a day is the difference between the highest and the lowest meter reading of that day in German time. What is consumed between the last reading of a day and the first reading of the next counts for neither day. The day the readings begin in is incomplete and is not recorded.
- **Judgement:** A day is judged when the first reading of the next day arrives, once 24 days are recorded. It is compared with the recorded days of the 90 days up to it, itself included. It is excessive if a DBSCAN clustering of those days (at least 10 days per cluster) leaves it out of every cluster and it lies above their 95 % quantile. The epsilon of the clustering is the knee of the sorted distances of the days to their nearest neighbour; where there is no knee or it lies at 0, as when nearly every day consumed the same, it is 1 (kWh, the unit of the meter).
- **History:** At every start the operator reads the meter readings of the last 90 days from Timescale and derives the days from them, so a restart, an update or a new deployment loses nothing and judges from the first day on if Timescale holds enough. The readings of the day in progress are taken over, so the day a restart cuts through keeps its whole consumption. A reading no later than the one before it is left out, which also drops what Kafka delivers again after a restart. This needs a database connection (`TS_CONN` in the environment or `ts_conn` in the config, which the flow engine gives a deployed operator). Without one, or for an input whose history only Kafka has or that has several devices, the operator starts without history, logs a warning and needs 24 days again after every restart. If Timescale cannot be read at all, the start fails and the platform starts the operator again.

The operator keeps no state of its own; everything is rebuilt from Timescale at every start.

## Input

| key | type | description |
|----------------------|-------|-------------------------|
| `Energy_Consumption` | float | Meter reading in kWh. |

The time of a reading is the time of its Kafka message. A reading without a value is left out.

## Output

One output when the day that a reading completes is excessive and is the day before the reading. Nothing is sent otherwise.

| key | type | description |
|---------|--------|-------------------------------------------------------|
| `value` | string | `Am gestrigen Tag wurde übermäßig verbraucht.` |

## Config options

| key | type | description |
|----------------|--------|---------------------------------------------------------------|
| `device_name` | string | Name of the device, used in the log message. default: "" |
| `logger_level` | string | default: "warning" |

`operator-manifest.json` declares the inputs, outputs and config options; the CI regenerates it from `main.py` with `python -m operator_lib manifest main.py` before each release.

## Tests

```bash
python -m unittest discover -s tests
```
