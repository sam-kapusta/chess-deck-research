# jr512_k8 labeled by the Personas decile-Opus method + the "is 2048 too many" answer (2026-07-12)

Follows `2026-07-11_jumprelu_sweep_l7diff.md`. Answers Sam's question ("is 2048 too many features?")
by training a compact **512** JumpReLU with the CANONICAL scheme and labeling it with the real
SandstonePersonas autointerp method (not the earlier tagger-vote hack).

## The model: `jr512_k8_final`
Canonical JumpReLU (`train_jr_canonical.py`, ported from SandstonePersonas `model.py` JumpReLUSAEAuxK):
**target_l0 quadratic loss + AuxK revival + separate ~33× LR on log_threshold**. That threshold
param-group is the piece the earlier `train_jr_sweep.py` missed — it's why θ "couldn't climb" and
`l0_coeff` looked inert. With it, `target_l0` becomes the real sparsity knob.
- dict=512, l0_alpha=4.0 (0.1 default left L0~24; had to crank it — the "very tricky" part), bw=0.1,
  init_threshold=0.5, 60 epochs. Result: **L0≈9.3, FVU 0.30, 1 dead.**
- **Fire-rate shape:** on a fixed 512 dict, LOW L0 gives the best 0.1–5% distribution; raising L0 just
  grows blobs (>10%) without adding in-band features (measured: tl8→172 in-band/15 blobs, tl26→133/96).
  **Dict size, not L0, is the lever for more in-band features** — 2048 had 1695 in 0.1-5% purely
  because 4× more features populate the band. (Per-feature fire rate ≈ L0/dict.)

## Labeling: the SandstonePersonas decile method (`label_features_decile_opus.py`)
Faithful port of `opus_label_audit_2.py` (see `SandstonePersonas/knowledge/naming-rules.md`):
- **Two orthogonal axes.** STRENGTH verdict {good / diffuse / too_broad / wrong / polysemantic /
  noise} = is there a sharp concept at the top activators. REACH `good_until_decile` (D10=top10% only …
  D1=nearly all) = how deep it holds — INDEPENDENT of strength (a good concept can decay fast or hold
  deep). `broad_label` records the normal decay (precise → generic) so it's not punished.
- **Per-band signal = the existing 62,956 Opus per-position analyses** (tactical_motif + tags +
  blunder_summary), aggregated per activation decile. Deciles computed over the **~60k analyzed
  positions only** (Sam) so every band is fully backed (vs 35% coverage if deciling all 168k).
- Opus 4.8 medium/adaptive, 16 threads, JSONL checkpoint. ~20min for 512.

## Results
**252 good · 186 diffuse · 41 too_broad · 26 polysemantic · 7 noise** (0 errors).
- **Reach among the 252 good:** peak at D8–D9 (hold through top 20-30%); **28 hold to D4 or deeper**
  (the broad-reach gold — e.g. f204/f267 "Hanging Piece Left En Prise" @D2, f258 "King Walks Into
  Mating Net" @D3).
- Labels are genuinely **mechanism-level** — "Trading Into Lost Pawn Endgame", "Premature Passed Pawn
  Push", "Missed Back-Rank Mate", "Premature Exchange Squandering Advantage" — far beyond the
  tagger-vote's terse tags (which topped out at best_uci-only 1-ply detection).

## The answer to "is 2048 too many?"  → **yes, and 512 still has slack.**
155 distinct `good_label`s across the 252 good features, BUT the top ~10 are all "hanging piece"
variants — **~70 of 252 good features (~28%) re-encode "you left a piece hanging"** (Left Undefended /
En Prise / Overlooked / Ignored…). Even at 512 the dictionary spends a quarter of its good capacity on
near-duplicates of the single concept the l7-diff representation captures most strongly. So:
- The l7-diff rep has maybe **~150 genuinely distinct coaching concepts**, and even those have
  near-synonyms. This is the same collapse the k6/v7 + architecture-comparison work found.
- 2048 was inherited from the k6/v7 lineage, not justified for coaching. 512 loses redundancy, not
  concepts (same categories covered). A dedup/merge pass (decoder-cosine or label-embedding) would
  cut the ~70 hanging-piece features to a handful and get closer to the true concept count.

## Artifacts
- `output/jumprelu_l7diff/jr512_k8_final.pt` (+jr256), `final_results.jsonl` — models (gitignored .pt).
- `output/jumprelu_l7diff/labels_decile_jr512.json` — the 512 labels (verdict/reach/labels/desc/reason).
- `output/jumprelu_l7diff/labels_decile_jr512.html` — browsable explorer (verdict/reach/decay filters).
- Scripts (committed): `scripts/sae/train_jr_canonical.py`, `scripts/03_feature_labeling/label_features_decile_opus.py`.
- Also on chess-poc `~/SageMaker/jr_canon_out/`.

## Next (open)
- **Dedup the hanging-piece cluster** to recover the true distinct-concept count.
- Optionally label the 256 model + compare (leaner still).
- The 62,956 Opus analyses cover only 35% of the l7 cache — a top-up batch would deepen low-decile
  reach signal, but Sam chose to use what exists.
