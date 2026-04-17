"""
Label schema for Perceiver training data.

Defines severity levels and the mapping from severity to tool requirement.
tool_required is now controlled entirely via only_severity in DEFECT_VARIANTS:
  only_severity="heavy" → always heavy → tool_required=False
  only_severity="light" → always light → tool_required=True
  no only_severity       → severity-dependent (default)
"""

# Two-level severity: light needs tool, heavy is reasoning-only.
SEVERITIES = ["light", "heavy"]

SEVERITY_TOOL_MAP = {
    "light": True,    # needs tool: subtle difference requires precise measurement
    "heavy": False,   # reasoning-only: obvious difference visible from preview/stats
}

# Dimension count distribution: P(N) for N injected dimensions.
# Skewed toward fewer dimensions (realistic).
N_DIM_WEIGHTS = {
    1: 0.35,
    2: 0.35,
    3: 0.20,
    4: 0.08,
    5: 0.02,
}


def needs_tool(severity: str) -> bool:
    """Whether this severity level requires tool-based assessment."""
    return SEVERITY_TOOL_MAP[severity]
