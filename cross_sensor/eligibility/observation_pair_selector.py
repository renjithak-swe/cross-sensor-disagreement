"""Pair each eligible Landsat observation with its nearest-in-time eligible Sentinel-2 observation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..discovery.observation_discovery import Observation, Sensor
from .comparison_eligibility import ACCEPT, EligibilityDecision, ReasonCode

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class ObservationPair:
    pair_id: str
    landsat: Observation
    sentinel2: Observation | None
    time_difference_hours: float | None
    decision: EligibilityDecision

    @property
    def signed_time_difference_hours(self) -> float | None:
        """Sentinel-2 time minus Landsat time."""
        if self.sentinel2 is None:
            return None
        return (self.sentinel2.acquired_at - self.landsat.acquired_at).total_seconds() / SECONDS_PER_HOUR


def deduplicate_acquisitions(observations: Sequence[Observation]) -> tuple[list[Observation], list[tuple[Observation, EligibilityDecision]]]:
    """Keep one item per (sensor, grid, acquisition time). A reprocessed duplicate with the highest processing
    version wins; ties go to the lexically last item ID so the choice is deterministic."""
    groups: dict[tuple[Sensor, str, str], list[Observation]] = {}
    for observation in observations:
        key = (observation.sensor, observation.grid_id, observation.acquired_at.isoformat(timespec="seconds"))
        groups.setdefault(key, []).append(observation)
    kept: list[Observation] = []
    rejected: list[tuple[Observation, EligibilityDecision]] = []
    for members in groups.values():
        ordered = sorted(members, key=lambda item: (item.processing_version, item.item_id))
        kept.append(ordered[-1])
        rejected.extend(
            (duplicate, EligibilityDecision(ReasonCode.DUPLICATE_ACQUISITION, f"superseded by {ordered[-1].item_id}"))
            for duplicate in ordered[:-1]
        )
    return sorted(kept, key=lambda item: (item.acquired_at, item.item_id)), rejected


class ObservationPairSelector:
    def __init__(self, max_time_difference_hours: float) -> None:
        self._max_hours = max_time_difference_hours

    def select(self, landsat: Sequence[Observation], sentinel2: Sequence[Observation]) -> list[ObservationPair]:
        """One candidate pair per Landsat observation, so every Landsat scene appears in the report."""
        ordered_landsat = sorted(landsat, key=lambda item: (item.acquired_at, item.item_id))
        pairs: list[ObservationPair] = []
        for index, landsat_observation in enumerate(ordered_landsat, start=1):
            pair_id = f"P{index:03d}"
            if not sentinel2:
                pairs.append(ObservationPair(pair_id, landsat_observation, None, None, EligibilityDecision(ReasonCode.NO_COUNTERPART)))
                continue
            nearest = min(
                sentinel2,
                key=lambda candidate: (abs((candidate.acquired_at - landsat_observation.acquired_at).total_seconds()), candidate.item_id),
            )
            hours = abs((nearest.acquired_at - landsat_observation.acquired_at).total_seconds()) / SECONDS_PER_HOUR
            decision = (
                ACCEPT
                if hours <= self._max_hours
                else EligibilityDecision(ReasonCode.EXCESSIVE_TIME_DIFFERENCE, f"{hours:.1f} h > {self._max_hours:.1f} h")
            )
            pairs.append(ObservationPair(pair_id, landsat_observation, nearest, hours, decision))
        return pairs
