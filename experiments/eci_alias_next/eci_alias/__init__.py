"""Offline alias experiment kit; no model provider or training dependencies."""
from .core import Document, View, prepare_view, project_one_generation, aggregate
__all__=["Document","View","prepare_view","project_one_generation","aggregate"]
