# The alarm level comes from extreme value theory

## Scope

This applies to how a day is judged in [main.py](../main.py) (`alarm_level` and `Operator.is_excessive`): the consumption of a day against up to 89 days before it. The method is the one of [leakage-detection](https://github.com/SENERGY-Platform/leakage-detection) (`docs/alarm-level-from-extreme-value-theory.md` there), which judges slots of five minutes; this note records what differs for days. The figures below come from simulated households over 730 days with two random seeds, not from real meters. They compare the variants with each other, and they do not predict the alarm levels of real meters.

## Decision

A day is excessive above a level that comes from the tail of the days before it: a generalized Pareto distribution fitted to the days above the 75 % quantile, with a shape no lower than 0 and at least 10 days in the tail. The only parameter is `false_alarms_per_month`, by default 1. This replaced DBSCAN with an epsilon from the knee of the nearest-neighbour distances and a 95 % quantile in November 2026.

## Why

- **The former judgement was a rank test.** The epsilon came from the distance to the nearest neighbour, but DBSCAN needs 10 days within it, so the upper days counted as noise and the 95 % quantile decided. The simulated households got 1.5 to 2.8 false alarms a month, and nothing could be set to change that.
- **The 75 % quantile instead of the 90 % of leakage-detection.** The 89 days before a day hold 9 above their 90 % quantile, fewer than the 10 a fit needs, so no day would ever be judged. Above the 75 % quantile lie 22 of them, and 38 days already hold 10. The 80 % quantile needs 47 days and gave slightly more false alarms.
- **Days are far fewer than slots.** One false alarm a month means 1 in 30 for a day, a consumption within the range of the 89 days seen, while leakage-detection's level lies far beyond its slots. So the level here can lie below the largest day of the last 90 days, and the model mainly smooths the tail; it extrapolates only for rarer false alarms.
- **What it costs.** At 1 a month the simulated households got 0.5 to 1.3 false alarms a month, and 1.5 to 2.1 where the consumption rose steadily over the year: against the days before, a rise looks like excess with any method. A day of twice the usual consumption was reported in 90 to 100 % of the cases, one of 1.5 times only where the days differ little, as with DBSCAN. Days are judged from 38 days on instead of 24, and a device that consumes nothing on most days is not judged at all, because its days have no tail.
- **Rejected: a lower default.** At 0.25 a month the false alarms fell to 0 to 0.5 a month, but where the consumption is skewed (gamma or lognormal days) only 10 to 37 % of the doubled days were reported.

## Before revisiting

Replay the history of several real meters and compare their false alarms with `false_alarms_per_month`. If there are too many, lower `false_alarms_per_month` before changing the method. If a seasonal rise causes them, a shorter history would follow it faster, but it leaves fewer days in the tail.
