"""
Background Task Queue for Long-Running Simulations

Allows simulations to run asynchronously without blocking HTTP requests.
Users can queue a simulation, receive a task ID, and poll for results.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

LOGGER = logging.getLogger(__name__)

# In-memory task storage (use Redis in production for persistence)
_TASKS: dict[str, dict[str, Any]] = {}


class SimulationTask:
    """Represents a background simulation task"""

    def __init__(self, task_id: str, params: dict[str, Any]):
        self.task_id = task_id
        self.params = params
        self.status = "queued"  # queued, running, completed, failed
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert task to dict for API response"""
        elapsed = None
        if self.started_at and self.completed_at:
            elapsed = self.completed_at - self.started_at

        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time_s": elapsed,
            "params": self.params,
            "result": self.result if self.status == "completed" else None,
            "error": self.error if self.status == "failed" else None
        }


def create_simulation_task(
    symbols: list[str] | None = None,
    num_predictions: int = 50,
    days_back: int = 7
) -> str:
    """
    Queue a simulation task for background execution.

    Args:
        symbols: List of symbols to simulate
        num_predictions: Target number of predictions
        days_back: Days of historical data

    Returns:
        Task ID for polling
    """
    task_id = str(uuid.uuid4())

    task = SimulationTask(
        task_id=task_id,
        params={
            "symbols": symbols,
            "num_predictions": num_predictions,
            "days_back": days_back
        }
    )

    _TASKS[task_id] = task
    LOGGER.info(f"Created simulation task {task_id}")

    # Start background execution
    asyncio.create_task(_execute_simulation(task))

    return task_id


async def _execute_simulation(task: SimulationTask):
    """Execute simulation in background"""
    try:
        task.status = "running"
        task.started_at = time.time()

        LOGGER.info(f"Starting simulation task {task.task_id}")

        # Run the simulation
        from core.historical_simulator import get_historical_simulator

        simulator = get_historical_simulator()
        result = await simulator.run_simulation(
            symbols=task.params["symbols"],
            num_predictions=task.params["num_predictions"],
            days_back=task.params["days_back"]
        )

        task.status = "completed"
        task.result = result
        task.completed_at = time.time()

        LOGGER.info(f"Completed simulation task {task.task_id} in {task.completed_at - task.started_at:.1f}s")

    except Exception as e:
        task.status = "failed"
        task.error = str(e)
        task.completed_at = time.time()

        LOGGER.error(f"Simulation task {task.task_id} failed: {e}", exc_info=True)


def get_task_status(task_id: str) -> dict[str, Any] | None:
    """
    Get status of a simulation task.

    Args:
        task_id: Task ID from create_simulation_task

    Returns:
        Task status dict or None if not found
    """
    task = _TASKS.get(task_id)
    return task.to_dict() if task else None


def cleanup_old_tasks(max_age_hours: int = 24):
    """Remove tasks older than max_age_hours"""
    cutoff = time.time() - (max_age_hours * 3600)
    to_remove = [
        task_id for task_id, task in _TASKS.items()
        if task.created_at < cutoff
    ]

    for task_id in to_remove:
        del _TASKS[task_id]

    if to_remove:
        LOGGER.info(f"Cleaned up {len(to_remove)} old simulation tasks")


def list_tasks(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """
    List simulation tasks.

    Args:
        status: Filter by status (queued, running, completed, failed)
        limit: Maximum number of tasks to return

    Returns:
        List of task dicts
    """
    tasks = list(_TASKS.values())

    if status:
        tasks = [t for t in tasks if t.status == status]

    # Sort by created_at descending (newest first)
    tasks.sort(key=lambda t: t.created_at, reverse=True)

    return [t.to_dict() for t in tasks[:limit]]
