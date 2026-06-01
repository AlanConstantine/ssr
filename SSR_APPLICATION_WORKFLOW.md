# Applying Cross-Formulation SSR Embeddings

## Positioning

SSR embeddings should be treated as local solvation-environment representations. They are most useful when they encode geometry, atom identity, local coordination, ion-pairing state, and shell composition in a way that can be compared across electrolyte formulations.

This is consistent with electrolyte literature: Li-ion electrolyte behavior is often interpreted through solvation-shell composition, coordination, ion-pairing state, aggregate formation, viscosity, diffusion, and conductivity. Quantum-chemical studies also show that HOMO/LUMO levels are not purely isolated-molecule properties; they can be shifted or renormalized by Li coordination and the surrounding solvation shell.

The right application pattern is two-level:

```text
local xyz solvation structure -> SSR embedding -> local property prediction

many local embeddings from one formulation -> pooled formulation descriptor -> formulation property prediction
```

This avoids a common mistake: using a single local structure as if it represented a whole electrolyte formulation.

## Local Solvation-Structure Property Prediction

### Targets

Local structure embeddings can be used to predict properties computed for that same local cluster or shell, such as:

- HOMO
- LUMO
- HOMO-LUMO gap
- local oxidation / reduction tendency
- solvation energy
- desolvation energy proxy
- coordination number
- ion-pairing state
- shell composition class

For HOMO/LUMO specifically, labels should come from a consistent quantum-chemistry workflow on the extracted local structure or a chemically well-defined cluster derived from it.

This point matters chemically. HOMO/LUMO trends are often used to reason about oxidative and reductive stability, but solvent orbital levels can change when molecules are coordinated to Li+ or embedded in a solvation shell. Therefore a local-cluster label is better aligned with SSR than a free-molecule HOMO/LUMO label.

Example label table:

```csv
path,formulation_id,homo_ev,lumo_ev,gap_ev,method,basis
formula_001/Frame100_Li_2DMC_2EC_2EMC_id1030.xyz,formula_001,-7.21,-0.84,6.37,wB97X-D,def2-SVP
formula_002/Frame100_Li_2DMC_2EC_2EMC_id1030.xyz,formula_002,-7.05,-0.78,6.27,wB97X-D,def2-SVP
```

The `path` field is preferred because different formulations may contain the same filename.

### Minimal Model

1. Train or load SSR encoder.
2. Export per-structure embeddings:

```bash
python scripts/export_embeddings.py \
  --data_dir ./data/formulations \
  --ckpt ./runs/ssr/checkpoints/best.pt \
  --out_dir ./runs/ssr/embeddings \
  --feature_mode atom_phys_shell \
  --center_on_li
```

3. Join `embeddings.csv` with a local property label table by `path`.
4. Train a supervised model:

```text
X = [SSR embedding, optional RDF/CN/shell descriptors]
y = HOMO or LUMO
model = Ridge / RandomForest / XGBoost / MLP
```

### Recommended Splits

Use multiple splits because random split can overestimate transfer:

- random structure split
- held-out formulation split
- held-out salt split
- held-out solvent family split
- held-out center Li trajectory split

The most meaningful result is held-out formulation performance. If SSR only works in random split, it may be memorizing formulation-specific local patterns.

### Baselines

Compare:

```text
composition/signature only
RDF/CN/shell descriptors only
atom-count descriptors only
SSR embedding only
SSR + RDF/CN/shell descriptors
```

SSR is useful if it improves held-out prediction over handcrafted local descriptors.

## Formulation Property Prediction

Formulation properties are not properties of one local structure. A formulation induces a distribution of local solvation environments.

For each formulation:

```text
formula_i -> {z_1, z_2, ..., z_N}
```

where each `z_j` is an SSR embedding for one Li-centered local structure.

Aggregate the distribution into a formulation-level descriptor:

```text
mean(z)
std(z)
quantiles(z)
cluster fractions
coordination mean/std
RDF mean/std
shell-state fractions
```

Then fuse with formulation-level variables:

```text
salt identity
solvent ratios
additive ratios
salt concentration
temperature
molecular descriptors
```

### Targets

High-confidence formulation targets:

- ionic conductivity
- viscosity
- Li diffusion coefficient
- anion diffusion coefficient
- Li transference number
- ion-pair / aggregate fraction

These targets are close to the microscopic information represented by SSR. Concentrated electrolyte literature distinguishes SSIP, CIP, and aggregate states, and links solvation/association structure to transport mechanisms, effective ion size, viscosity, and diffusion.

More difficult cell-level targets:

- Coulombic efficiency
- capacity retention
- cycle life
- rate capability

Cell-level targets require electrode, protocol, loading, separator, voltage window, and formation-cycle metadata. SSR embeddings alone are not enough.

### Minimal Formulation Predictor

Recommended workflow:

```text
1. Export all local SSR embeddings with path and formulation_id.
2. Group by formulation_id.
3. Compute pooled embedding statistics.
4. Join with formulation labels and conditions.
5. Train a property model.
```

Feature matrix:

```text
X_formula = [
  composition features,
  condition features,
  molecular descriptors,
  mean/std/quantile SSR embedding,
  RDF/CN/shell-state statistics
]
```

Prediction:

```text
y_formula = conductivity / viscosity / diffusion / transference number
```

### Cross-Formulation Interpretation

A useful SSR embedding should support questions like:

```text
Which formulations produce similar Li solvation environments?
Which formulation has more anion-rich Li shells?
Which local structures correlate with low LUMO or easier reduction?
Which shell-state fractions correlate with high conductivity or low viscosity?
```

This can be analyzed by:

- clustering SSR embeddings across all formulations
- computing cluster fractions per formulation
- correlating cluster fractions with formulation properties
- inspecting representative xyz structures from each cluster

## Physical Positive Training

To make embeddings more transferable across formulations, the next training mode should not define positives by formulation id. It should define positives by local physical similarity:

```text
similar coordination number
similar Li-X distance distribution
similar RDF
similar shell composition
same ion-pairing state
```

Use soft positives:

```text
w_ij = exp(-d(physical_descriptor_i, physical_descriptor_j) / tau)
```

This lets structures from different formulations be close if their local solvation physics is close, while allowing same-signature structures to remain far apart if geometry or ion-pairing state differs.

Hard negatives should include:

```text
same signature but different RDF / coordination / shell state
different formulation but deceptively similar composition label
same formulation but different Li solvation basin
```

## Data Loading Requirements

The code now supports formulation-id directories:

```text
data/
  formula_001/
    Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
  formula_002/
    Frame100_Li_2DMC_2EC_2EMC_id1030.xyz
```

Identical filenames across formulations are allowed. Metadata exports include:

```text
path
formulation_id
signature
```

For temporal mode, the identity is:

```text
formulation_id + trajectory_id + center Li id
```

This prevents two different formulations with identical filenames from being treated as one trajectory.

## Practical Recommendation

Start with two supervised probes:

1. Local probe:

```text
SSR embedding -> HOMO / LUMO / gap
```

using quantum labels for local clusters.

2. Formulation probe:

```text
pooled SSR embedding distribution + composition + condition -> conductivity / viscosity / diffusion
```

If SSR improves held-out formulation prediction over composition-only and RDF/CN-only baselines, then the embedding is useful beyond a single electrolyte formula.

## Literature Support

The following references motivate the two-level SSR application workflow:

- Ion-solvent chemistry reviews discuss how Li+ solvation shells and multi-solvent complexes affect electrolyte redox stability, including HOMO/LUMO shifts of coordinated solvents: https://www.sciencedirect.com/science/article/pii/S2667325821001011
- Reviews of electrode-electrolyte interphase chemistry connect electrolyte HOMO/LUMO levels, Li+ solvation complexes, and reduction/oxidation behavior: https://pmc.ncbi.nlm.nih.gov/articles/PMC7500179/
- Work on electrolyte-renormalized oxidative stability argues that solvent HOMO levels are modified by the solvation environment, so isolated-molecule descriptors can be insufficient: https://www.osti.gov/pages/biblio/1799381
- Concentration-dependent electrolyte reviews summarize how SSIP, CIP, and aggregate formation change local solvation structure, effective ion size, viscosity, diffusion, and transport mechanisms: https://pmc.ncbi.nlm.nih.gov/articles/PMC9448741/
- Functional electrolyte additive reviews note that practical electrolyte design involves solvation structure, ionic conductivity, transference number, viscosity, and HOMO/LUMO-related reactivity: https://pmc.ncbi.nlm.nih.gov/articles/PMC12393036/
- Recent formulation-level machine-learning models predict liquid electrolyte properties from ionic conductivity to solvation structure while respecting mixture permutation invariance and condition dependence: https://arxiv.org/abs/2504.18728
- Multiscale electrolyte-design reviews connect DFT, MD, AIMD, and machine-learning simulations to solvation energy, coordination, redox stability, ion transport kinetics, viscosity, diffusion, and conductivity: https://www.sciencedirect.com/science/article/pii/S2405829726000607
