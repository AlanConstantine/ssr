# Using SSR Embeddings for Electrolyte Formulation Property Prediction

## Question

SSR currently learns embeddings for local solvation structures. A natural extension is to ask whether these embeddings can help predict properties of full electrolyte formulations, not only properties of individual solvation structures.

The answer is yes, with an important qualification: an SSR embedding should not be treated as a direct replacement for formulation descriptors. Instead, SSR should provide a microscopic solvation-structure channel that is fused with composition, molecular descriptors, concentration, temperature, and other experimental or simulation conditions.

## Feasibility Assessment

This direction is feasible and scientifically well motivated.

Existing electrolyte formulation models already map component molecular structures and formulation ratios to electrolyte or cell-level performance. For example, formulation graph models represent each component with molecular graph features and combine them using molar-fraction weighting to predict battery electrolyte performance. More recent liquid-electrolyte formulation models also explicitly handle formulation invariance, temperature, concentration, conductivity, and solvation-related targets.

The physical rationale is also strong. Electrolyte properties such as ionic conductivity, viscosity, diffusion coefficient, and Li+ transference number depend on local Li+ coordination, ion pairing, aggregate formation, solvent exchange, and the distribution of solvation shell states. SSR embeddings are designed to summarize exactly this local structural information from `.xyz` atomic configurations.

Therefore, the right abstraction is:

```text
formulation
  -> many sampled local solvation structures
  -> many SSR embeddings
  -> formulation-level distribution descriptor
  -> property predictor
```

## Key Idea

For each formulation, collect a set of local solvation structures from MD, AIMD, MLMD, or curated structural samples:

```text
F_i = {x_1, x_2, ..., x_N}
```

Use the SSR encoder to map each structure to an embedding:

```text
z_j = SSR(x_j)
```

Then aggregate the set of embeddings into a formulation-level descriptor:

```text
Z_F = aggregate({z_1, z_2, ..., z_N})
```

This `Z_F` describes the solvation-structure distribution induced by the formulation. It can then be combined with formulation and molecular descriptors to predict formulation-level properties.

## Recommended Model Architecture

Use a multi-channel predictor:

```text
composition channel
  salt, solvents, additives, molar ratios, concentration, temperature

molecular descriptor channel
  SMILES graph embeddings, dielectric constant, donor/acceptor number,
  dipole moment, HOMO/LUMO, viscosity prior, molecular weight

solvation structure channel
  pooled SSR embeddings, coordination statistics, RDF bins,
  solvation shell composition, shell-state fractions

fusion model
  MLP / Ridge / XGBoost / GNN / Transformer

targets
  ionic conductivity, viscosity, diffusion coefficient,
  Li+ transference number, CE, capacity retention, cycle life
```

For early work, prefer Ridge, Random Forest, XGBoost, or a small MLP before using a complex neural architecture. The important question is whether SSR adds incremental predictive value over composition-only and handcrafted physical descriptors.

## Formulation-Level Aggregation Options

### 1. Statistical Pooling

The simplest approach is to summarize the SSR embeddings with fixed statistics:

```text
mean(z)
std(z)
min/max(z)
quantiles(z)
```

This is easy to implement, robust on small datasets, and works with classical regressors.

### 2. Shell-State Fractions

Cluster SSR embeddings into interpretable local states:

```text
SSIP, CIP, AGG, solvent-rich shell, anion-rich shell, low/high coordination states
```

Then represent each formulation by the fraction of structures in each state:

```text
Z_F = [fraction_state_1, fraction_state_2, ..., fraction_state_K]
```

This is highly interpretable and can directly connect predictions to solvation chemistry.

### 3. Attention Pooling / Multiple Instance Learning

Treat each formulation as a bag of solvation structures:

```text
formulation = {z_1, z_2, ..., z_N}
```

Use an attention module to learn which local structures matter most for a target property:

```text
Z_F = AttentionPool({z_j})
```

This is more flexible than mean pooling and is useful when rare states, such as aggregates or unusual coordination environments, strongly affect properties.

### 4. Distribution Distance

Represent each formulation as a distribution in SSR embedding space. Compare formulations using distances such as:

```text
Wasserstein distance
MMD
cluster histogram distance
```

This is useful for similarity search, formulation retrieval, and active learning.

## Minimum Viable Experiment

### Data Layout

Use one directory per formulation:

```text
data/formulations/
  F001/
    metadata.json
    structures/
      frame_000_li_00.xyz
      frame_001_li_03.xyz
      ...
  F002/
    metadata.json
    structures/
      ...
```

Example `metadata.json`:

```json
{
  "formulation_id": "F001",
  "salt": "LiPF6",
  "solvents": {"EC": 0.3, "EMC": 0.7},
  "additives": {"FEC": 0.02},
  "salt_concentration_m": 1.0,
  "temperature_K": 298.15,
  "targets": {
    "ionic_conductivity_mS_cm": 10.2,
    "viscosity_cP": 3.1,
    "li_transference_number": 0.36
  }
}
```

### Feature Construction

For each formulation:

1. Export SSR embeddings for all sampled local structures.
2. Compute pooled embedding statistics.
3. Compute physical descriptors:
   - Li coordination number mean/std
   - Li-O / Li-F / Li-N distance statistics
   - RDF bins
   - solvation shell composition
   - shell-state cluster fractions
4. Concatenate these with formulation descriptors:
   - molar ratios
   - salt concentration
   - temperature
   - molecular descriptors

### Baselines

A credible experiment should compare:

```text
composition-only
molecular-descriptor-only
RDF/CN-only
SSR-only
composition + molecular descriptors
composition + molecular descriptors + RDF/CN
composition + molecular descriptors + SSR
composition + molecular descriptors + SSR + RDF/CN
```

The key success criterion is not absolute performance alone. SSR is useful if it consistently improves over composition-only and handcrafted physical baselines, especially under held-out formulation splits or few-shot settings.

## Recommended Targets

High-confidence targets:

- ionic conductivity
- viscosity
- Li+ diffusion coefficient
- anion diffusion coefficient
- Li+ transference number
- degree of ion pairing or aggregate fraction

Medium-confidence targets:

- oxidative/reductive stability trend
- desolvation energy
- solvation free energy
- SEI precursor tendency

Cell-level targets require more caution:

- Coulombic efficiency
- capacity retention
- cycle life
- rate capability

These are influenced by electrolyte composition, solvation, electrode chemistry, interphase formation, cycling protocol, cell format, and impurities. SSR can still help, but it must be fused with broader experimental metadata.

## Implementation Plan for SSR

### Stage A: Formulation Aggregator

Add a script such as:

```text
scripts/export_formulation_embeddings.py
```

Responsibilities:

- read formulation directories
- load a trained SSR checkpoint
- export per-structure embeddings
- aggregate embeddings per formulation
- save `formulation_embeddings.csv`
- save `formulation_metadata.csv`

Suggested output columns:

```text
formulation_id
num_structures
z_mean_0 ... z_mean_D
z_std_0 ... z_std_D
coordination_mean
coordination_std
rdf_0 ... rdf_K
shell_state_fraction_0 ... shell_state_fraction_M
```

### Stage B: Property Predictor

Add a script such as:

```text
scripts/train_formulation_property.py
```

Inputs:

```text
--features formulation_embeddings.csv
--targets formulation_targets.csv
--target ionic_conductivity_mS_cm
--model ridge|random_forest|xgboost|mlp
--split random|heldout_formula|heldout_salt|heldout_solvent
```

Outputs:

```text
metrics.json
predictions.csv
feature_ablation.csv
```

### Stage C: Ablation and Validation

Run ablations:

```text
composition only
composition + RDF/CN
composition + SSR
composition + SSR + RDF/CN
```

Use robust splits:

```text
random split
held-out formulation
held-out salt
held-out solvent family
held-out concentration
```

For a small dataset, use repeated K-fold cross-validation and report uncertainty.

### Stage D: Active Learning

Use the trained formulation predictor to suggest new formulations:

1. Generate candidate compositions.
2. Predict target properties and uncertainty.
3. Select candidates with high expected improvement or high uncertainty.
4. Run MD/MLMD or experiments.
5. Add new structures and labels back into the training set.

## Risks and Limitations

### Data Availability

SSR requires sampled structures. If only formulation ratios are available, SSR cannot directly contribute unless a surrogate model is trained:

```text
composition -> predicted SSR distribution
```

### Sampling Bias

The formulation-level embedding depends on how structures are sampled. MD length, force field quality, temperature, equilibration, and Li-centered extraction rules can strongly affect the descriptor.

### Target Leakage

If structures are generated using conditions too close to the target measurement, care is needed to avoid leakage. Splits should be formulation-aware and ideally chemistry-aware.

### Cell-Level Properties

Cycle life and capacity retention are not electrolyte-only properties. A model trained on these targets must include electrode, protocol, cell format, and interphase metadata.

## Practical Recommendation

Start with transport properties rather than cell-level outcomes.

The first practical milestone should be:

```text
composition + condition + SSR pooled embedding
  vs.
composition + condition
  vs.
composition + condition + RDF/CN
```

on ionic conductivity, viscosity, or diffusion coefficient.

If SSR improves held-out formulation prediction and the learned shell-state fractions are chemically interpretable, then the approach is validated enough to extend toward CE, retention, and active formulation optimization.

## References

- Formulation graphs for mapping battery electrolyte structure/composition to device performance: https://research.ibm.com/publications/formulation-graphs-for-mapping-structure-composition-of-battery-electrolytes-to-device-performance
- arXiv version of formulation graph electrolyte modeling: https://arxiv.org/abs/2307.03811
- Liquid electrolyte formulation model with conductivity and solvation-structure targets: https://arxiv.org/abs/2504.18728
- Machine-learning molecular dynamics for LiTFSI/G3 electrolyte transport and Li-O solvation effects: https://arxiv.org/abs/2503.20243
