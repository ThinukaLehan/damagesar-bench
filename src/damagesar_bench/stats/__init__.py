"""Metrics and statistics (FR10), written test-first.

* ``metrics``     AUROC, average precision, DeLong (single and paired)
* ``bootstrap``   within-event tile-cluster bootstrap (describes one event only)
* ``event_level`` the claim layer: t-interval over per-event gains, ordered v11.1 decision
                  table, Bonferroni level, exact sign-flip (supporting), Hartung-Knapp
                  (secondary), MDE and the pilot-SD rule
* ``gates``       Gate 1 (C2 go/no-go on development events)
"""

from .event_level import (  # noqa: F401
    WORTHWHILE, Interval, PilotAction, SignFlip, Verdict, bonferroni_level, decide,
    decide_interval, hartung_knapp, mde, pilot_sd_rule, sign_flip_test, t_interval,
)
from .gates import Gate1, gate1  # noqa: F401
from .metrics import MetricError, auroc, average_precision, delong, delong_paired_diff  # noqa: F401
