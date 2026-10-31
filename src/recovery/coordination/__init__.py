"""
Координаційні механізми multi-agent recovery.

Реалізує: Протокол M1, §4.3
(див. docs/Протокол_M1.docx).
"""

from src.recovery.coordination.base_coordinator import CoordinatorBase
from src.recovery.coordination.auction import AuctionCoordinator
from src.recovery.coordination.broadcast import BroadcastCoordinator
from src.recovery.coordination.independent import IndependentCoordinator
from src.recovery.coordination.token import TokenCoordinator


COORDINATOR_REGISTRY = {
    "auction": AuctionCoordinator,
    "broadcast": BroadcastCoordinator,
    "independent": IndependentCoordinator,
    "token": TokenCoordinator,
}


def create_coordinator(mode: str) -> CoordinatorBase:
    """Створює coordinator за назвою режиму."""
    try:
        return COORDINATOR_REGISTRY[mode]()
    except KeyError as exc:
        allowed = ", ".join(sorted(COORDINATOR_REGISTRY))
        raise ValueError(f"Невідомий coordination mode '{mode}'. Дозволені: {allowed}.") from exc


__all__ = [
    "COORDINATOR_REGISTRY",
    "AuctionCoordinator",
    "BroadcastCoordinator",
    "CoordinatorBase",
    "IndependentCoordinator",
    "TokenCoordinator",
    "create_coordinator",
]
