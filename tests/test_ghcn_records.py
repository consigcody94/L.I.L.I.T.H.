from datetime import date

import pytest

from data.providers.ghcn_records import decode_daily
from training.prepare_global import temporal_split
from training.verification_metrics import error_metrics


def line(element, value=100, mflag=" ", qflag=" "):
    return "USW00094728" + "202402" + element + f"{value:5d}{mflag}{qflag}0" * 31


def test_units_flags_trace_missing_and_leap_calendar():
    rows = decode_daily("\n".join([line("TMAX"), line("SNOW"), line("SNWD"),
                                  line("TMIN", -9999), line("PRCP", 0, "T")]))
    assert len(rows) == 29 * 5
    values = {r["element"]: r for r in rows}
    assert values["TMAX"]["value"] == 10
    assert values["SNOW"]["value"] == 100
    assert values["SNWD"]["value"] == 100
    assert values["TMIN"]["value"] is None
    assert values["PRCP"]["trace"]
    assert values["PRCP"]["value"] == 0
    assert values["TMAX"]["source_flag"] == "0"
    assert all(row["value"] is None for row in decode_daily(line("TMAX", qflag="X")))


def test_split_purges_input_and_target_at_boundaries():
    assert temporal_split(date(2016, 11, 1), date(2016, 12, 1), date(2016, 12, 14)) == "train"
    assert temporal_split(date(2016, 12, 1), date(2016, 12, 31), date(2017, 1, 13)) is None
    assert temporal_split(date(2016, 12, 20), date(2017, 1, 19), date(2017, 2, 1)) is None
    assert temporal_split(date(2021, 1, 1), date(2021, 1, 31), date(2021, 2, 13)) == "test"


def test_metrics_reject_broadcast_and_mixed_unit_scalar():
    import numpy as np
    predicted = np.array([[[2, 4, 6]], [[0, 0, 0]]], dtype=float)
    scores = error_metrics(predicted, np.zeros_like(predicted))
    assert scores["mae"] == [1, 2, 3]
    with pytest.raises(ValueError):
        error_metrics(predicted, np.zeros((1, 1, 3)))
