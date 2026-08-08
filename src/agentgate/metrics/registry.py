"""Metric registry.

Metrics register themselves by name at import time. The registry is the single place that knows
which metrics exist, which family each belongs to, and what each requires — which is what lets
``docs/metrics.md`` be generated from the code rather than maintained beside it.
"""

from __future__ import annotations

from collections.abc import Iterable

from agentgate.errors import ConfigError
from agentgate.metrics.base import BaseMetric, Metric
from agentgate.schemas.common import MetricFamily

_REGISTRY: dict[str, Metric] = {}


def register[MetricT: type[BaseMetric]](metric_cls: MetricT) -> MetricT:
    """Class decorator: instantiate a metric and add it to the registry.

    Args:
        metric_cls: A concrete :class:`~agentgate.metrics.base.BaseMetric` subclass.

    Returns:
        The class, unchanged, so it stays importable and subclassable.

    Raises:
        ConfigError: On a missing or duplicate metric name.
    """
    instance = metric_cls()
    if not instance.name:
        msg = f"{metric_cls.__name__} does not declare a metric name"
        raise ConfigError(msg)
    if instance.name in _REGISTRY:
        msg = f"duplicate metric name {instance.name!r}"
        raise ConfigError(msg)
    _REGISTRY[instance.name] = instance
    return metric_cls


def register_instance(metric: Metric) -> Metric:
    """Register an already-constructed metric (used by parameterised metrics)."""
    if metric.name in _REGISTRY:
        msg = f"duplicate metric name {metric.name!r}"
        raise ConfigError(msg)
    _REGISTRY[metric.name] = metric
    return metric


def get(name: str) -> Metric:
    """Look up a metric by name.

    Args:
        name: Registered metric name.

    Returns:
        The metric instance.

    Raises:
        ConfigError: When no such metric is registered.
    """
    if name not in _REGISTRY:
        msg = f"unknown metric {name!r}; registered metrics: {', '.join(names())}"
        raise ConfigError(msg)
    return _REGISTRY[name]


def has(name: str) -> bool:
    """True when ``name`` is registered."""
    return name in _REGISTRY


def names() -> list[str]:
    """Every registered metric name, sorted."""
    return sorted(_REGISTRY)


def all_metrics() -> list[Metric]:
    """Every registered metric, in name order."""
    return [_REGISTRY[name] for name in names()]


def by_family(family: MetricFamily) -> list[Metric]:
    """Registered metrics in one family, in name order."""
    return [metric for metric in all_metrics() if metric.family is family]


def select(names_wanted: Iterable[str] | None = None) -> list[Metric]:
    """Resolve a metric selection.

    Args:
        names_wanted: Metric names, or ``None`` for everything registered.

    Returns:
        The selected metrics.

    Raises:
        ConfigError: When a requested name is unknown.
    """
    if names_wanted is None:
        return all_metrics()
    return [get(name) for name in names_wanted]


def clear() -> None:
    """Empty the registry. Test-only; production code registers at import time."""
    _REGISTRY.clear()
