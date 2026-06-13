"""Optional integrations that bridge external solvers into PropFlow.

Sub-packages here may depend on heavy optional libraries (e.g. PyTorch) and are
never imported by the core package. Import them explicitly, e.g.::

    from propflow.integrations.dabp import DABPEngine
"""
