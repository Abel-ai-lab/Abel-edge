"""Abel Proof validation — metric triangle gate for strategy admission.

Three leverage-invariant, orthogonal dimensions:
  Ratio (Lo-adj Sharpe or Sharpe) — mean/std quality
  Rank  (IC)                      — prediction quality
  Shape (Omega)                   — gain/loss asymmetry

No known transformation improves all three except genuine signal improvement.

The audited live validation contract uses applicable-gate denominators
(commonly 6 or 8, rising when Omega and full-year loss accounting are applicable)
rather than the older fixed 20/21-style narrative.
"""

from causal_edge.validation.metrics import compute_all_metrics, validate
from causal_edge.validation.gate import validate_strategy

__all__ = ["compute_all_metrics", "validate", "validate_strategy"]
