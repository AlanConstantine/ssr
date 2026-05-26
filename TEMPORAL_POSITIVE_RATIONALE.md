# Temporal Positives for SSR

## Critical Assessment

Using nearby frames from a molecular dynamics trajectory as positive pairs is reasonable only under controlled assumptions.

The chemical intuition is that adjacent or short-lag frames usually preserve the same local solvation basin: Li coordination, ion-pairing state, first-shell composition, and local solvent arrangement often evolve continuously rather than changing independently at every saved frame. Time-lagged molecular representation methods use this continuity to learn slow collective variables and kinetic structure.

However, adjacent MD frames are also strongly autocorrelated. Treating every adjacent frame as an independent positive can overstate sample size and can teach the model trivial continuity rather than chemically meaningful state similarity. Negative sampling is also risky: if negatives are drawn mostly from different formulations, the model may learn composition/signature differences rather than structural dynamics.

Therefore SSR uses a conservative temporal definition:

- require explicit trajectory id, center Li id, and frame index;
- use same-trajectory bounded time windows for positives;
- use far same-signature frames as preferred negatives;
- use different-trajectory negatives only as fallback;
- expose lag/window parameters rather than hard-coding a universal timescale.

## Implemented Definition

For an anchor frame `i`, frame `j` is a positive if:

```text
trajectory_i == trajectory_j
center_li_id_i == center_li_id_j
temporal_min_lag <= abs(frame_i - frame_j) <= temporal_positive_window
```

Frame `j` is a negative if:

```text
trajectory_i == trajectory_j
abs(frame_i - frame_j) >= temporal_negative_min_gap
```

or if it comes from another trajectory. Same-signature negatives are preferred when available.

## Metadata Requirements

Recommended `.xyz` comment line:

```text
signature: Li_2DMC_2EC_2EMC trajectory: TrajA center_id: 1030 frame: 100
```

Supported filename style:

```text
TrajA_Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
```

In this project naming convention, `id1030` is the center Li ion id. Temporal mode groups frames by `trajectory_id + center_id`, not by trajectory alone. This matters because different Li ions in the same trajectory can have different solvation environments and should not be treated as the same temporal object.

Temporal mode intentionally fails if trajectory id, center Li id, or frame index cannot be parsed. This avoids silently treating unordered `.xyz` collections as trajectories.

## Choosing Windows

The right lag depends on saved-frame stride and solvation dynamics.

Use smaller positive windows when:

- frames are saved sparsely;
- solvent exchange is fast;
- Li coordination state changes quickly;
- the target is short-time local geometry.

Use larger positive windows when:

- frames are saved very frequently;
- the goal is to learn slow solvation-state variables;
- autocorrelation analysis indicates the local state remains stable.

`temporal_negative_min_gap` should be larger than `temporal_positive_window` and ideally set beyond the local-state autocorrelation time. If this is unknown, start conservatively and verify with coordination/RDF/shell-state probes.

## References

- Time-lagged autoencoders show how time-lagged molecular configurations can be used to learn slow collective variables: https://arxiv.org/abs/1710.11239
- Reviews of temporal data-driven collective variables discuss using time-lagged structure in MD to identify slow modes: https://www.cambridge.org/core/journals/qrb-discovery/article/chasing-collective-variables-using-temporal-datadriven-strategies/83B8318410B46F49F79AB1F3D1D54282
- MD autocorrelation analysis highlights that trajectory frames are correlated and that sampling time must be considered carefully: https://pmc.ncbi.nlm.nih.gov/articles/PMC6325644/
- Temporal contrastive learning literature warns that negatives from different sequences can introduce shortcut features, motivating same-signature hard negatives when possible: https://www.emergentmind.com/topics/temporal-contrastive-representations
