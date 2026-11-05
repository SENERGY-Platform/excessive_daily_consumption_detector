# Excessive Daily Consumption Detector

Reports a day whose electricity consumption is far higher than on the other days of the last 90 days.

## How it works

- **Days:** The consumption of a day is the difference between the highest and the lowest meter reading of that day in German time. What is consumed between the last reading of a day and the first reading of the next counts for neither day. The day the readings begin in is incomplete and is not recorded.
- **Judgement:** A day is judged when the first reading of the next day arrives, against the recorded days before it in the last 90 days. It is excessive above their alarm level, which a usual day exceeds about `false_alarms_per_month` times a month. The level comes from extreme value theory, as in [leakage-detection](https://github.com/SENERGY-Platform/leakage-detection): the days above the 75 % quantile of the days before are the tail, a generalized Pareto distribution is fitted to it with probability weighted moments, and the level is where that distribution reaches the probability of the false alarms. The shape of the tail is taken no lower than 0, an exponential tail. Days are compared to 1 Wh.
- **Too few days:** While fewer than 10 of the days before lie in their tail, no day is judged. With some consumption every day that takes 38 days; a device that consumes nothing on most days is not judged at all.
- **Sensitivity:** In simulated households, one false alarm a month as configured gave 0.5 to 1.3 false alarms a month, and 1.5 to 2.1 where the consumption rose steadily; the DBSCAN judgement before gave 1.5 to 2.8. A day of twice the usual consumption was reported in 90 to 100 % of the cases, one of 1.5 times only where the days differ little. The method has not been checked on real meters yet. If it reports too often, lower `false_alarms_per_month` first. The figures and the reasons are in [docs/alarm-level-from-extreme-value-theory.md](docs/alarm-level-from-extreme-value-theory.md).
- **History:** At every start the operator reads the meter readings of the last 90 days from Timescale and derives the days from them, so a restart, an update or a new deployment loses nothing and judges from the first day on if Timescale holds enough. The readings of the day in progress are taken over, so the day a restart cuts through keeps its whole consumption. A reading no later than the one before it is left out, which also drops what Kafka delivers again after a restart. This needs a database connection (`TS_CONN` in the environment or `ts_conn` in the config, which the flow engine gives a deployed operator). Without one, or for an input whose history only Kafka has or that has several devices, the operator starts without history, logs a warning and needs 38 days again after every restart. If Timescale cannot be read at all, the start fails and the platform starts the operator again.

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
|--------------------------|--------|---------------------------------------------------------------|
| `device_name` | string | Name of the device, used in the log message. default: "" |
| `false_alarms_per_month` | float | How often a month a usual day may be reported as excessive; above 0. The higher, the lower the alarm level. default: 1 |
| `logger_level` | string | default: "warning" |

`operator-manifest.json` declares the inputs, outputs and config options; the CI regenerates it from `main.py` with `python -m operator_lib manifest main.py` before each release.

## Tests

```bash
python -m unittest discover -s tests
```
