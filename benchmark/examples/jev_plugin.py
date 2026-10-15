"""An external plugin example: replay choice distributions exported from a future JEV model.

Enable with plugins: [benchmark.examples.jev_plugin]. No provider or gold data is accessed.
"""
from benchmark.plugins.builtin import DecisionScores
from benchmark.registry import register


@register("relation_predictor", "jev_recorded", version="example-1", mode="replay")
class RecordedJev(DecisionScores):
    """Same interface as a local or hosted JEV predictor, using recorded probabilities."""
