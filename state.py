"""
State Management Module

Handles saving and loading of trading system state for recovery.
Persists: current regime, open positions, trade history, session stats.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.regime_strategies import Direction


@dataclass
class PositionState:
    """Serialized position for state snapshot."""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    size: float
    quantity: float
    entry_time: str
    regime_id: int
    regime_name: str


@dataclass
class TradeRecord:
    """Serialized trade for history."""
    trade_id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    size: float
    quantity: float
    pnl: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    status: str
    regime_id: int
    regime_name: str


@dataclass
class SessionStats:
    """Session statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    regime_distribution: dict[str, int] = field(default_factory=dict)
    start_time: str = ""
    last_update: str = ""


@dataclass
class TradingState:
    """Complete trading system state for persistence."""
    version: str = "1.0"
    current_regime_id: int = 0
    current_regime_name: str = "UNKNOWN"
    current_regime_probability: float = 0.0
    is_flickering: bool = False
    flicker_count: int = 0
    last_bar_time: Optional[str] = None
    positions: list[PositionState] = field(default_factory=list)
    trade_history: list[TradeRecord] = field(default_factory=list)
    session_stats: SessionStats = field(default_factory=SessionStats)
    hmm_model_path: str = ""
    hmm_last_trained: Optional[str] = None
    saved_at: str = ""


class StateManager:
    """Manages trading system state persistence."""

    def __init__(self, state_path: Optional[Path] = None):
        self._state_path = state_path or Path("state_snapshot.json")
        self._state: Optional[TradingState] = None

    @property
    def state_path(self) -> Path:
        return self._state_path

    def load(self) -> Optional[TradingState]:
        """Load state from file."""
        if not self._state_path.exists():
            return None

        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)

            state = TradingState(
                version=data.get("version", "1.0"),
                current_regime_id=data.get("current_regime_id", 0),
                current_regime_name=data.get("current_regime_name", "UNKNOWN"),
                current_regime_probability=data.get("current_regime_probability", 0.0),
                is_flickering=data.get("is_flickering", False),
                flicker_count=data.get("flicker_count", 0),
                last_bar_time=data.get("last_bar_time"),
                hmm_model_path=data.get("hmm_model_path", ""),
                hmm_last_trained=data.get("hmm_last_trained"),
                saved_at=data.get("saved_at", ""),
            )

            # Deserialize positions
            for p in data.get("positions", []):
                state.positions.append(PositionState(
                    symbol=p["symbol"],
                    direction=p["direction"],
                    entry_price=p["entry_price"],
                    stop_loss=p["stop_loss"],
                    size=p["size"],
                    quantity=p["quantity"],
                    entry_time=p["entry_time"],
                    regime_id=p["regime_id"],
                    regime_name=p["regime_name"],
                ))

            # Deserialize trade history
            for t in data.get("trade_history", []):
                state.trade_history.append(TradeRecord(
                    trade_id=t["trade_id"],
                    symbol=t["symbol"],
                    direction=t["direction"],
                    entry_price=t["entry_price"],
                    exit_price=t.get("exit_price"),
                    size=t["size"],
                    quantity=t["quantity"],
                    pnl=t.get("pnl"),
                    entry_time=t["entry_time"],
                    exit_time=t.get("exit_time"),
                    status=t["status"],
                    regime_id=t["regime_id"],
                    regime_name=t["regime_name"],
                ))

            # Deserialize session stats
            stats = data.get("session_stats", {})
            state.session_stats = SessionStats(
                total_trades=stats.get("total_trades", 0),
                winning_trades=stats.get("winning_trades", 0),
                losing_trades=stats.get("losing_trades", 0),
                total_pnl=stats.get("total_pnl", 0.0),
                regime_distribution=stats.get("regime_distribution", {}),
                start_time=stats.get("start_time", ""),
                last_update=stats.get("last_update", ""),
            )

            self._state = state
            return state

        except Exception as e:
            print(f"Warning: Failed to load state: {e}")
            return None

    def save(
        self,
        regime_id: int = 0,
        regime_name: str = "UNKNOWN",
        regime_probability: float = 0.0,
        is_flickering: bool = False,
        flicker_count: int = 0,
        last_bar_time: Optional[datetime] = None,
        positions: Optional[list] = None,
        trade_history: Optional[list] = None,
        session_stats: Optional[SessionStats] = None,
        hmm_model_path: str = "",
        hmm_last_trained: Optional[datetime] = None,
    ) -> None:
        """Save state to file."""
        state = TradingState(
            version="1.0",
            current_regime_id=regime_id,
            current_regime_name=regime_name,
            current_regime_probability=regime_probability,
            is_flickering=is_flickering,
            flicker_count=flicker_count,
            last_bar_time=last_bar_time.isoformat() if last_bar_time else None,
            hmm_model_path=hmm_model_path,
            hmm_last_trained=hmm_last_trained.isoformat() if hmm_last_trained else None,
            saved_at=datetime.now(timezone.utc).isoformat(),
        )

        # Convert positions
        if positions:
            for p in positions:
                state.positions.append(PositionState(
                    symbol=p.symbol,
                    direction=p.direction.value if hasattr(p.direction, 'value') else str(p.direction),
                    entry_price=p.entry_price,
                    stop_loss=p.stop_loss,
                    size=p.size,
                    quantity=p.quantity,
                    entry_time=p.entry_time.isoformat() if isinstance(p.entry_time, datetime) else str(p.entry_time),
                    regime_id=p.regime_id,
                    regime_name=p.regime_name,
                ))

        # Convert trade history
        if trade_history:
            for t in trade_history:
                state.trade_history.append(TradeRecord(
                    trade_id=t.get("trade_id", ""),
                    symbol=t.get("symbol", ""),
                    direction=t.get("direction", ""),
                    entry_price=t.get("entry_price", 0.0),
                    exit_price=t.get("exit_price"),
                    size=t.get("size", 0.0),
                    quantity=t.get("quantity", 0.0),
                    pnl=t.get("pnl"),
                    entry_time=t.get("entry_time", ""),
                    exit_time=t.get("exit_time"),
                    status=t.get("status", ""),
                    regime_id=t.get("regime_id", 0),
                    regime_name=t.get("regime_name", ""),
                ))

        # Use provided session stats or default
        state.session_stats = session_stats or SessionStats()

        # Ensure start_time is set
        if not state.session_stats.start_time:
            state.session_stats.start_time = datetime.now(timezone.utc).isoformat()
        state.session_stats.last_update = datetime.now(timezone.utc).isoformat()

        # Serialize to JSON
        data = {
            "version": state.version,
            "current_regime_id": state.current_regime_id,
            "current_regime_name": state.current_regime_name,
            "current_regime_probability": state.current_regime_probability,
            "is_flickering": state.is_flickering,
            "flicker_count": state.flicker_count,
            "last_bar_time": state.last_bar_time,
            "hmm_model_path": state.hmm_model_path,
            "hmm_last_trained": state.hmm_last_trained,
            "saved_at": state.saved_at,
            "positions": [asdict(p) for p in state.positions],
            "trade_history": [asdict(t) for t in state.trade_history],
            "session_stats": asdict(state.session_stats),
        }

        # Ensure directory exists
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._state_path, "w") as f:
            json.dump(data, f, indent=2)

        self._state = state

    def get_state(self) -> Optional[TradingState]:
        """Get current state."""
        return self._state


def create_state_manager(path: Optional[str] = None) -> StateManager:
    """Factory function to create StateManager."""
    return StateManager(Path(path) if path else None)