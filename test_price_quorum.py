import math

from core.price_quorum import PriceDecision, PriceProvider, PriceQuorum


def _provider(value: float, prev: float) -> PriceProvider:
    return PriceProvider(
        name=f"p{value}",
        fetcher=lambda v=value, p=prev: (v, p, f"p{v}"),
    )


def test_price_quorum_reaches_consensus():
    quorum = PriceQuorum(min_quorum_open=3, min_quorum_closed=1, tolerance_open=0.02)
    providers = [
        _provider(10.0, 9.5),
        _provider(10.01, 9.5),
        _provider(9.99, 9.5),
        _provider(10.02, 9.5),
    ]

    decision = quorum.get_price(
        symbol="WOLF",
        providers=providers,
        prev_close=9.5,
        is_market_open=True,
        timeout=1.0,
    )
    quorum.close()

    assert isinstance(decision, PriceDecision)
    assert decision.price is not None
    assert decision.quorum_size >= 3
    assert math.isclose(decision.price, 10.0, rel_tol=0.02)


def test_price_quorum_detects_failure():
    quorum = PriceQuorum(min_quorum_open=3, min_quorum_closed=1, tolerance_open=0.02)
    providers = [
        _provider(10.0, 9.5),
        _provider(12.0, 9.5),
        _provider(14.0, 9.5),
    ]

    decision = quorum.get_price(
        symbol="WOLF",
        providers=providers,
        prev_close=9.5,
        is_market_open=True,
        timeout=1.0,
    )
    quorum.close()

    assert decision.price is None
    assert decision.reason != "consensus"
    assert decision.quorum_size < 3
