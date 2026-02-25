import json
import os

import pytest
from fastapi.testclient import TestClient

from wolf_app import APP

_HAS_DB = bool(os.getenv("DATABASE_URL"))


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(APP)


def test_multi_predictions_endpoint_ok_field_present(client: TestClient) -> None:
    response = client.get("/api/predictions/multi/run")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert "ok" in data


@pytest.mark.skipif(not _HAS_DB, reason="Requires DATABASE_URL (PostgreSQL)")
def test_multi_predictions_endpoint_shape_basic(client: TestClient) -> None:
    response = client.get("/api/predictions/multi/run")
    assert response.status_code == 200

    data = response.json()

    # Envelope keys
    assert "predictions" in data
    assert "counts" in data
    assert "total" in data
    assert "timestamp" in data

    predictions = data["predictions"]
    counts = data["counts"]

    # Predictions should be a mapping, even if empty
    assert isinstance(predictions, dict)
    for key in ("stocks", "crypto", "vip"):
        assert key in predictions
        assert isinstance(predictions[key], list)

    # Counts should be a mapping of ints
    assert isinstance(counts, dict)
    for key in ("stocks", "crypto", "vip"):
        assert key in counts
        assert isinstance(counts[key], int)

    # Total should equal the sum of counts and be non-negative
    total = data["total"]
    assert isinstance(total, int)
    assert total >= 0
    assert total == counts["stocks"] + counts["crypto"] + counts["vip"]


