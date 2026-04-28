# Why Causal?

## The Core Argument

Correlation is a property of **data**. Causation is a property of the **data generating process** (DGP).

When regimes change (bull→bear, policy shift, crisis):
- Data changes → correlations break → correlation-based signals die
- DGP persists → causal links survive → causal signals live

This is not a preference. It is Judea Pearl's definition of causation: **a causal relationship is one that remains invariant under intervention.** Regime change *is* intervention on the market.

## Three Dimensions

| Dimension | Why causal wins | Root cause |
|-----------|----------------|------------|
| **Math** | Causation = intervention invariance (Pearl do-calculus) | Structural, not distributional |
| **Quant** | Higher persistence, lower crowding, higher stability | DGP invariance → OOS survival → higher live Sharpe |
| **AI** | Only inductive bias guaranteeing OOD generalization | Causal models valid under distribution shift |

## Consequence for Strategy Validation

**Consequence 1:** Causal signals have higher live Sharpe because they survive regime changes that kill correlative signals.

**Consequence 2:** Causal structure constrains the search space (K) to mechanistically justified relationships → Deflated Sharpe Ratio passes honestly → discoveries are real, not noise.

This is the meta-standard: not "use causal because K is smaller" but "use causal because it's the only relationship type that persists in live trading, and smaller K is a *consequence* of only searching for what can persist."

## The Abel Plugin (Optional)

abel-edge integrates with [Abel's Causal API](https://abel.ai) for automated causal discovery:

```bash
# Discover what causally drives an asset
abel-edge discover ETHUSD

# Or inspect the markov blanket
abel-edge discover ETHUSD --mode mb

# Output: YAML snippet to paste into strategies.yaml
```

Without Abel, you can manually specify causal parents in `strategies.yaml` under the `parents:` field. The framework works identically — Abel just automates the discovery step.

## Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Bailey & Lopez de Prado (2014). *The Deflated Sharpe Ratio*
