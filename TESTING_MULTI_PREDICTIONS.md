## Testing Multi-Symbol Predictions Endpoint

This project now includes a focused test for the multi-symbol
predictions endpoint exposed at `/api/predictions/multi/run` in
`wolf_app.APP`.

### Prerequisites

- Python environment with project dependencies installed.
- `pytest` and `fastapi` test utilities available in the environment.

### How to Run the Test

From the repository root:

```bash
cd /workspaces/ghost-protocol
pytest tests/test_multi_predictions.py

```text

The test will import `wolf_app.APP`, spin up a `TestClient`, and verify
that `/api/predictions/multi/run`:

- Returns HTTP 200.
- Returns a JSON object with:
  - `ok` flag present.
  - `predictions` dictionary containing `stocks`, `crypto`, and `vip`


    lists (which may be empty in some environments).

  - `counts` dictionary with integer counts for `stocks`, `crypto`, and


    `vip`.

  - `total` field representing the total number of predictions.
