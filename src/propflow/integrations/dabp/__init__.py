"""DABP (Deep Attentive Belief Propagation) integration.

Requires the optional ``[dabp]`` extra (torch + torch-geometric). Import is
explicit and never triggered by the core package::

    from propflow.integrations.dabp import DABPEngine
"""

from .engine import DABPEngine

__all__ = ["DABPEngine"]
