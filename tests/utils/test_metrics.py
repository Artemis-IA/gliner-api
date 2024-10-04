# tests/utils/test_metrics.py
from src.utils.metrics import REQUEST_COUNT, REQUEST_LATENCY

def test_metrics_initialization():
    assert REQUEST_COUNT is not None
    assert REQUEST_LATENCY is not None
