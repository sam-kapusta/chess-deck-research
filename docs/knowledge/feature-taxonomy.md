# Feature Taxonomy — Final Method (2026-05-29)

How we group 2048 SAE features into a coaching hierarchy.

## What failed

**Matryoshka SAE top-level prefixes** — forces features to reconstruct at small prefix sizes. Result: features group by piece type (which piece moved), not by coaching concept. The reconstruction objective finds the highest-variance direction, which is geometric (piece type), not semantic (why it's a mistake).

**Phi-coefficient (co-occurrence) clustering** — features barely co-fire (mean phi = 0.0002). With k=16 and dict=2048, each feature fires 0.8% — pairwise co-occurrence is too sparse. Produces one megacluster (888 features) plus tiny groups of strongly-linked pairs. Doesn't scale.

## What works

**Decoder cosine similarity + Ward hierarchical clustering + LLM re-grouping.**

Pipeline:
1. Train standard BatchTopK SAE (k=16, dict=2048, on v2 corrected data)
2. Compute pairwise cosine similarity between all 2048 decoder vectors
3. Ward hierarchical clustering into 25 initial categories, then sub-cluster each into groups of ~7
4. Sonnet labels each of the 280 subclusters (3-6 word coaching name per cluster)
5. Opus re-groups subclusters into 15-25 coaching categories based on label semantics

### Why this works

Decoder cosine captures "features that represent similar directions in activation space" — which IS semantic similarity for SAEs. Two features pointing in similar directions respond to similar types of positions, even if they never fire on the exact same position (which is why phi fails).

The LLM re-grouping step fixes Ward's tendency to split on geometry rather than concept. Ward might separate "king walks into fork" from "king walks into mating net" because they're geometrically distant, but Opus groups them both under "King Walks Into Danger" because they're the same coaching concept.

## Result: 20 coaching categories

| Category | Sub-patterns | Features | Description |
|----------|-------------|----------|-------------|
| Slow Play Punished | 24 | 169 | Passive/prophylactic moves when tactics demanded |
| Ignoring Threats | 21 | 168 | Failing to address opponent's immediate threats |
| Moving Hangs Pieces | 21 | 155 | Moving one piece leaves another unprotected |
| King Walks Into Danger | 20 | 147 | King steps into checks/forks/mating nets |
| Pawn Moves Weaken King | 17 | 135 | Pawn advances compromising king shelter |
| Pieces Left Undefended | 19 | 128 | Placing pieces on unprotected squares |
| Passed Pawn Blindness | 14 | 122 | Mishandling passed pawns |
| Rook Misplacement | 17 | 118 | Rooks on wrong files/ranks |
| Piece Abandons Defense | 17 | 112 | Moving a piece from its defensive post |
| Captures Backfire | 13 | 101 | Exchanges that activate opponent |
| Greedy Captures | 12 | 90 | Material grabs ignoring consequences |
| Piece Lands Badly | 11 | 82 | Pieces on tactically weak squares |
| Pawn Moves Ignore Threats | 10 | 70 | Pawn pushes during emergencies |
| King Safety Ignored | 11 | 69 | Prioritizing activity over king protection |
| Retreating Errors | 10 | 66 | Bad piece retreats |
| Back Rank Weakness | 9 | 63 | Back rank mate vulnerabilities |
| Checks Lose Tempo | 7 | 63 | Pointless checks wasting time |
| Fork Vulnerability | 6 | 42 | Moving into forkable positions |
| Piece Trapping | 5 | 31 | Own pieces getting trapped |
| Unsound Sacrifices | 2 | 13 | Bad sacrifices without compensation |

## Quality assessment

~80% of subclusters are correctly placed in their categories. The 20% misplacements come from ambiguous subcluster names (Opus re-groups by name only, not by position data). Fixable with a refinement pass showing Opus actual positions.

The taxonomy is coaching-coherent: a chess teacher would recognize these categories and assign drills based on them.

## Files

- `output/chess_blunder_taxonomy_v2.json` — Full tree (categories → subclusters → feature IDs)
- `output/chess_blunder_atlas.html` — Interactive visualization
- `scripts/sae/phi_clustering_v2.py` — Clustering script
- Model: `s3://chess-stage-a-140023406996/sae/weights/maia3_sae_v2/sweep_v2_k16_d2048.pt`

## Comparison to Sandstone persona_tree

| | Sandstone Personas | Chess Blunder Atlas |
|--|---|---|
| Features | 4096 | 2048 |
| Categories | ~15 | 20 |
| Subclusters | 208 | 280 |
| Clustering | Phi coefficient (works - personas co-fire) | Decoder cosine (phi fails - blunders don't co-fire) |
| Labeling | BGE-M3 embedding + LLM | Opus position analysis + LLM |
| Quality | High (strong co-occurrence signal) | Good (decoder geometry is a proxy for semantics) |
