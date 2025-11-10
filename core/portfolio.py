import time
from dataclasses import dataclass


@dataclass
class Position:
    id: str
    kind: str
    symbol: str
    qty: float
    price: float
    opened_at: float


_POSITIONS = [
    Position(
        id="1",
        kind="crypto",
        symbol="bitcoin",
        qty=0.01,
        price=20000.0,
        opened_at=time.time() - 86400,
    )
]


def list_positions():
    return _POSITIONS


def add(kind, symbol, qty, price):
    p = Position(
        id=str(len(_POSITIONS) + 1),
        kind=kind,
        symbol=symbol,
        qty=qty,
        price=price,
        opened_at=time.time(),
    )
    _POSITIONS.append(p)
    return p


def remove(position_id):
    global _POSITIONS
    _POSITIONS = [p for p in _POSITIONS if p.id != position_id]


def update(position_id, **kwargs):
    for p in _POSITIONS:
        if p.id == position_id:
            for k, v in kwargs.items():
                setattr(p, k, v)
            return p
