# Taxonomy method — the persona-atlas approach (reference)

The method Sam wants applied to the chess SAE taxonomy. Copied from the Sandstone
Persona taxonomy work (`SandstonePersonas/.../lab/phi_clustering/`). This is the
**correct** order: cluster first (bottom-up), categories emerge. NOT top-down
assignment into pre-baked buckets.

## Core principles

1. **Cluster on label-text semantics, not geometry or co-firing.**
   - Rejected: co-activation/phi (conflates "same person does both" with "same kind of thing"). Blunders barely co-fire anyway.
   - Rejected: decoder cosine (can't link "Bishop hangs on g4" to "Knight hangs on e4" — geometrically distant, same coaching lesson).
   - **Chosen: `BAAI/bge-m3` (1024-dim) embeddings of the feature's name + description.** Captures "are these the same kind of behavior/mistake." (bge-m3 is cached locally; MiniLM underperformed it.)

2. **Two-level structure: categories first as a constraint, then sub-groups WITHIN each.**
   - Assign each feature a top-level category, THEN cluster within that category. Constraining within-category prevents cross-domain false merges (a keyboard and a piano merging on shopping-language similarity).
   - For chess: ~12-20 coaching categories, sub-clusters (~7-15 features) inside each.

3. **One agent per category reads ALL its features holistically and designs groups.**
   - NEVER independent per-feature classification — that produces the magnet effect (one category swallows everything, others starve). This is exactly the bug in the abandoned `taxonomy_v2.json` (Slow Play 408 vs Undefended 4).
   - Rule: **name the TYPE OF MISTAKE, not the move.** "Hanging a piece," not "Bishop-to-g4 errors." Broad coherent groups are GOOD; fragmentation is the failure mode.
   - Aim for ~N/12–N/7 groups per category. Every feature placed; a few singletons for genuine niches OK.

4. **Population metrics per group:** reach % (fraction of positions firing ANY member) + sum-of-fire-rates % (gap indicates intra-group overlap).

## Quality control (three independent passes — what makes it defensible)

- **5a. Cross-category reconciliation** — collect misfiled features, re-route each to correct category + best group there. Result: every feature placed once, no orphans/dupes.
- **5b. Member-level verification** — one independent agent per category re-reads every group + member, flags wrong-group / wrong-category / generic names. Catches errors a coherence metric can't.
- **5c. Coherence-bar audit** — define an objective bar from hand-built reference categories (persona used mean pairwise bge-m3 cosine ≥ 0.593). Score every group; investigate any under the bar.

## Worked-example structure (per-category regroup prompt)

> You are regrouping SAE features within ONE category into clean groups. Each group = a distinct TYPE OF MISTAKE.
> STEP 1: list ALL features (id, fire_rate%, name, description) — read them all.
> STEP 2: design groups. Name the type of mistake, not the move. Title Case, 2-4 words. Broad coherent groups good. Aim ~N/12–N/7 groups. Every feature lands somewhere. If a feature belongs in a DIFFERENT category, exclude + report as misfit.
> STEP 3: score + save (compute reach, write json). Iterate until every non-misfit feature is placed.

## Chess-specific adaptation

- "Type of person" → "type of mistake" (Hanging a piece, King walks into danger, Premature pawn push…).
- Reference categories: hand-build 3-4 first to set the quality bar, then parallel agents for the rest using them as worked examples.
- Source signal = the per-feature `description` (accurate, board-verified) + chip. Fire rate from the flat k=32 model over the v2 cache.

## Result shape (persona had)

19 categories, 280 personas, 4096 features, complete coverage; browsable atlas
(`persona_atlas.html` style — warm paper palette, Fraunces + IBM Plex, sidebar +
expandable card grid). Chess target: ~15-20 categories, ~200-280 sub-clusters,
~2000 features, with fire rates at every level.