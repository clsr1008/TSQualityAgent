"""
Label schema for Perceiver training data.

Defines severity levels and the mapping from severity to tool requirement.
"""

# Two-level severity: light defects are reasoning-only, heavy need tools.
SEVERITIES = ["light", "heavy"]

SEVERITY_TOOL_MAP = {
    "light": True,    # needs tool: subtle difference requires precise measurement
    "heavy": False,   # reasoning-only: obvious difference visible from preview/stats
}

# Dimension count distribution: P(N) for N injected dimensions.
# Skewed toward fewer dimensions (realistic).
N_DIM_WEIGHTS = {
    0: 0.05,   # tie case
    1: 0.35,
    2: 0.30,
    3: 0.18,
    4: 0.08,
    5: 0.03,
    6: 0.01,
}


def needs_tool(severity: str) -> bool:
    """Whether a dimension at this severity level requires tool-based assessment."""
    return SEVERITY_TOOL_MAP[severity]
