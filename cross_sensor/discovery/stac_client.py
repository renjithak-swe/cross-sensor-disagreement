"""STAC catalogue search. Returns plain item dictionaries; no asset is signed or read here."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any, Protocol

from ..errors import CatalogError

LOGGER = logging.getLogger(__name__)

StacItem = dict[str, Any]
MAX_ATTEMPTS = 4
RETRY_DELAY_SECONDS = 5.0


class StacCatalog(Protocol):
    endpoint: str

    def search(self, collection: str, bbox: tuple[float, float, float, float], start: date, end: date) -> list[StacItem]: ...

    def fetch_items(self, collection: str, item_ids: Sequence[str]) -> list[StacItem]: ...


def _catalog_errors() -> tuple[type[BaseException], ...]:
    import requests
    from pystac_client.exceptions import APIError

    return (APIError, requests.RequestException, ValueError, KeyError)


class PystacCatalog:
    """A STAC API catalogue accessed through pystac-client."""

    def __init__(self, endpoint: str, open_client: Callable[[str], Any] | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.endpoint = endpoint
        self._errors = _catalog_errors()
        self._sleep = sleep
        if open_client is None:
            from pystac_client import Client

            open_client = Client.open
        try:
            self._client = open_client(endpoint)
        except self._errors as error:
            raise CatalogError(f"cannot open STAC catalogue {endpoint}: {error}") from error

    def search(self, collection: str, bbox: tuple[float, float, float, float], start: date, end: date) -> list[StacItem]:
        interval = f"{start.isoformat()}/{end.isoformat()}"
        return self._run_search(collection, collections=[collection], bbox=list(bbox), datetime=interval)

    def fetch_items(self, collection: str, item_ids: Sequence[str]) -> list[StacItem]:
        if not item_ids:
            return []
        return self._run_search(collection, collections=[collection], ids=list(item_ids))

    def _run_search(self, collection: str, **parameters: Any) -> list[StacItem]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                items = [item.to_dict() for item in self._client.search(**parameters).items()]
                break
            except self._errors as error:
                if attempt == MAX_ATTEMPTS:
                    raise CatalogError(f"search of {collection} at {self.endpoint} failed: {str(error)[:300]}") from error
                LOGGER.warning("catalogue search failed, retrying", extra={"collection": collection, "attempt": attempt})
                self._sleep(RETRY_DELAY_SECONDS * attempt)
        LOGGER.info("catalogue search", extra={"collection": collection, "item_count": len(items)})
        return sorted(items, key=lambda item: str(item.get("id")))
