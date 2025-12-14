web: uvicorn wolf_app:APP --host 0.0.0.0 --port $PORT
evaluator: while true; do python3 core/prediction_evaluator.py; sleep 3600; done
worker: WORKER_MODE=1 uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8000}
