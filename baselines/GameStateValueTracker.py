from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional
from collections import defaultdict

MetricDict = Dict[str, Callable[[Any, Any, Any, int], Any]]

@dataclass
class ValueCfg:
    data_type: type
    metrics: Optional[MetricDict] = field(default_factory=dict)
    default_value: Optional[Any] = None


class GameStateValueTracker:
    """
    A utility class for tracking game-state values computed during an environment's step function.
    Values are registered with a name and a corresponding value. Initialization requires a
    val_config dictionary that maps the name to a ValueCfg object containing the data type and metrics.

    Automatic metrics for maximum, minimum, and average are provided for int and float data types.
    Metrics are computed upon value registration if configured and are accessible for retrieval.

    Attributes:
        _values (dict): Stores the current and previous values registered.
        _metrics (dict): Stores computed metrics for the registered values.
        _counts (dict): Stores the number of times each value has been registered in an episode.
        _val_config (dict): Configuration mapping of value names to their corresponding ValueCfg.
    """

    def __init__(self, val_config: Optional[Dict[str, ValueCfg]] = None):
        if val_config is None:
            val_config = {}  # Ensures the object is initialized properly.
        self._values = defaultdict(lambda: {'current': None, 'previous': None})
        self._metrics = defaultdict(dict)
        self._counts = defaultdict(int)
        self._val_config = val_config

        # Initialize default values
        for name, cfg in self._val_config.items():
            self._values[name]['current'] = cfg.default_value
            self._values[name]['previous'] = cfg.default_value

            # Automatically add common metrics for numerical data types
            if cfg.data_type in [int, float]:
                # Handle None safely by checking if cur and metric are not None before comparison
                # cfg.metrics.setdefault('max', lambda cur, prev, metric, count: max(filter(lambda x: x is not None, (cur, metric))) if cur is not None else metric)
                # cfg.metrics.setdefault('min', lambda cur, prev, metric, count: min(filter(lambda x: x is not None, (cur, metric))) if cur is not None else metric)

                # cfg.metrics.setdefault('avg', lambda cur, prev, metric, count: ((metric * (count - 1)) + cur) / count if count else cur)
                pass


    def register_value(self, name: str, value: Any):
        """
        Registers a new value and updates metrics.

        Args:
            name (str): The name of the value to register.
            value (Any): The value to be registered.

        Raises:
            ValueError: If the name does not correspond to any entry in the configuration or if the
                        value type does not match the configured data type.
        """
        # If the value is configured, check its type
        cfg = self._val_config.get(name)
        if cfg is not None and not isinstance(value, cfg.data_type):
            raise ValueError(f"Value for '{name}' must be of type {cfg.data_type}.")

        self._counts[name] += 1
        self._values[name]['previous'] = self._values[name]['current']
        self._values[name]['current'] = value

        # Compute metrics only if there are metrics configured for this name
        if cfg is not None and cfg.metrics:
            for metric_name, metric_fn in cfg.metrics.items():
                self._metrics[name][metric_name] = metric_fn(
                    value,
                    self._values[name]['previous'],
                    self._metrics[name].get(metric_name),
                    self._counts[name]
                )

    def curr(self, name: str, default: Any = None, none_gets_default: bool = True) -> Any:
        """
        Retrieves the current value for a given name, or a default if not set or if None and none_gets_default is True.

        Args:
            name (str): The name of the value to retrieve.
            default (Any, optional): The default value to return if the current value is not set or is None.
            none_gets_default (bool, optional): If True, None values will be replaced with the default.

        Returns:
            Any: The current value registered or the default.
        """
        current_value = self._values.get(name, {'current': default})['current']
        return default if none_gets_default and current_value is None else current_value

    def prev(self, name: str, default: Any = None, none_gets_default: bool = True) -> Any:
        """
        Retrieves the previous value for a given name, or a default if not set or if None and none_gets_default is True.

        Args:
            name (str): The name of the value to retrieve.
            default (Any, optional): The default value to return if the previous value is not set or is None.
            none_gets_default (bool, optional): If True, None values will be replaced with the default.

        Returns:
            Any: The previous value registered or the default.
        """
        previous_value = self._values.get(name, {'previous': default})['previous']
        return default if none_gets_default and previous_value is None else previous_value

    def metric(self, value_name: str, metric_name: str, default: Any = None, none_gets_default: bool = True) -> Any:
        """
        Retrieves the given metric for a given value name, or a default if not set or if None and none_gets_default is True.

        Args:
            value_name (str): The name of the value whose metric is to be retrieved.
            metric_name (str): The name of the metric to be retrieved.
            default (Any, optional): The default value to return if the metric is not set or is None.
            none_gets_default (bool, optional): If True, None values will be replaced with the default.

        Returns:
            Any: The computed metric or the default.
        """
        metric_value = self._metrics.get(value_name, {}).get(metric_name, default)
        return default if none_gets_default and metric_value is None else metric_value


    def get_metrics(self, name: str) -> MetricDict:
        """
        Retrieves the metrics for a given value name.

        Args:
            name (str): The name of the value whose metrics are to be retrieved.

        Returns:
            MetricDict: A dictionary of computed metrics.
        """
        return self._metrics[name]

    def reset(self):
        """
        Resets the tracker's state for a new episode.
        """
    def reset(self):
        for name in self._val_config.keys():
            default_value = self._val_config[name].default_value
            self._values[name]['current'] = default_value
            self._values[name]['previous'] = default_value
        self._metrics.clear()
        self._counts.clear()
