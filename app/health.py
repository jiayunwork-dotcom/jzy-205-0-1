"""Health event log and time-indexed channel state derivation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import EventType, HealthEvent


@dataclass(frozen=True)
class ChannelState:
    healthy: bool
    gain_error_db: float
    phase_error_deg: float
    event_ids: tuple[str, ...]

    @property
    def multiplier(self) -> complex:
        if not self.healthy:
            return 0.0j
        gain = float(np.power(10.0, self.gain_error_db / 20.0))
        return gain * np.exp(1j * np.deg2rad(self.phase_error_deg))


@dataclass(frozen=True)
class ArrayState:
    timestamp: float
    channels: tuple[ChannelState, ...]
    event_ids: tuple[str, ...]

    @property
    def multipliers(self) -> np.ndarray:
        return np.asarray([channel.multiplier for channel in self.channels], dtype=complex)

    @property
    def failed_elements(self) -> list[int]:
        return [index + 1 for index, channel in enumerate(self.channels) if not channel.healthy]

    def actual_weights(self, nominal_weights: np.ndarray) -> np.ndarray:
        return np.asarray(nominal_weights, dtype=complex) * self.multipliers


class EventLog:
    """Idempotent event collection with chronological late-arrival support."""

    def __init__(self, element_count: int):
        if element_count < 2:
            raise ValueError("element_count must be at least 2")
        self.element_count = element_count
        self._events: dict[str, HealthEvent] = {}
        self._received_order: dict[str, int] = {}
        self._next_order = 0

    def add(self, event: HealthEvent) -> bool:
        if event.element < 1 or event.element > self.element_count:
            raise ValueError(f"event references non-existent element {event.element}")
        if event.event_id in self._events:
            return False
        self._events[event.event_id] = event
        self._received_order[event.event_id] = self._next_order
        self._next_order += 1
        return True

    def add_many(self, events: Sequence[HealthEvent]) -> list[bool]:
        return [self.add(event) for event in events]

    @property
    def events(self) -> list[HealthEvent]:
        return sorted(
            self._events.values(),
            key=lambda event: (event.timestamp, self._received_order[event.event_id], event.event_id),
        )

    def get(self, event_id: str) -> HealthEvent | None:
        return self._events.get(event_id)

    def derive_state(self, timestamp: float | None = None) -> ArrayState:
        events = self.events
        if timestamp is not None:
            events = [event for event in events if event.timestamp <= timestamp]
        channels: list[dict] = []
        active_by_channel: list[list[str]] = [[] for _ in range(self.element_count)]
        for index in range(self.element_count):
            channels.append({"healthy": True, "gain": 0.0, "phase": 0.0})
        for event in events:
            index = event.element - 1
            active_by_channel[index].append(event.event_id)
            channel = channels[index]
            if event.kind == EventType.failed:
                channel["healthy"] = False
            elif event.kind == EventType.recovered:
                channel["healthy"] = True
                channel["gain"] = 0.0
                channel["phase"] = 0.0
                active_by_channel[index] = []
            elif event.kind == EventType.gain_error:
                channel["gain"] = event.gain_error_db
            elif event.kind == EventType.phase_error:
                channel["phase"] = event.phase_error_deg
        channel_states = tuple(
            ChannelState(
                healthy=item["healthy"],
                gain_error_db=float(item["gain"]),
                phase_error_deg=float(item["phase"]),
                event_ids=tuple(active_by_channel[i]),
            )
            for i, item in enumerate(channels)
        )
        active_ids = tuple(event.event_id for event in events)
        state_time = events[-1].timestamp if events else float("-inf")
        if timestamp is not None:
            state_time = timestamp
        return ArrayState(state_time, channel_states, active_ids)

    def event_timestamps(self) -> list[float]:
        return sorted({event.timestamp for event in self.events})
