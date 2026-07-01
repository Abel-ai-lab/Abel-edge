"""Abel Proof validation — metric triangle gate for strategy admission.

Three leverage-invariant, orthogonal dimensions:
  Ratio (Lo-adj Sharpe or Sharpe) — mean/std quality
  Rank  (IC)                      — prediction quality
  Shape (Omega)                   — gain/loss asymmetry

No known transformation improves all three except genuine signal improvement.

The audited live validation contract uses applicable-gate denominators
(commonly 5 or 7, rising when Omega and full-year loss accounting are applicable)
rather than the older fixed 20/21-style narrative.
"""

from abel_edge.validation.gate_explain import explain_metric_gates
from abel_edge.validation.gate_vocabulary import list_gate_vocabulary
from abel_edge.validation.metrics import compute_all_metrics, validate
from abel_edge.validation.gate import validate_strategy

__all__ = [
    "compute_all_metrics",
    "explain_metric_gates",
    "list_gate_vocabulary",
    "validate",
    "validate_strategy",
]
