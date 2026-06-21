"""
LILITH Data Module

Handles GHCN data download, processing, and loading.

Submodules are imported lazily so lightweight, dependency-free pieces - such as
``data.providers.open_meteo`` (standard library only) - can be imported without
pulling in httpx, pandas or torch.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # for type-checkers / IDEs only
    from data.download.ghcn_daily import GHCNDailyDownloader
    from data.download.ghcn_hourly import GHCNHourlyDownloader

__all__ = ["GHCNDailyDownloader", "GHCNHourlyDownloader"]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "GHCNDailyDownloader":
        from data.download.ghcn_daily import GHCNDailyDownloader

        return GHCNDailyDownloader
    if name == "GHCNHourlyDownloader":
        from data.download.ghcn_hourly import GHCNHourlyDownloader

        return GHCNHourlyDownloader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
