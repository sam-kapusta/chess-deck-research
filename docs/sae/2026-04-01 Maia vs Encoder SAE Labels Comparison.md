# Maia vs Encoder SAE Labels — What Each Model Sees

## Summary
1000 features interpreted ($4 total). The two models see fundamentally different things.

## Maia (human modeling, 23M params, ~1900 Elo)
Sees WHAT: concrete positions, piece placement, board geography.
- **center** (132x), **rank** (106x), **board** (47x) — spatial awareness
- **black/white** (77x/32x) — color-specific patterns
- **undeveloped/displaced/unmoved** — development state
- **seventh/sixth/fifth** — specific rank features

## Encoder (deep analysis, 270M params, 2895 Elo)
Sees WHY: abstract evaluation, threat assessment, urgency.
- **safety/vulnerability** (101x/46x) — danger assessment
- **attack/pressure** (81x/24x) — threat detection
- **immediate/requires** (52x/48x) — urgency of action
- **severe** (25x) — degree of danger
- **dominance/coordination** (24x/22x) — positional quality

## The coaching implication
Together they tell a complete story:
- Maia: "Rook on seventh rank, king in center"
- Encoder: "King safety requires immediate defense, severe attack pressure"
- Combined coaching: "Your rook on the seventh rank is perfectly placed. The opponent's king is in danger — this position demands immediate action, not quiet maneuvering."

This is the L1+L2 integration working as designed: the encoder provides the evaluation context, Maia provides the positional description, and the LLM weaves them into coaching.
