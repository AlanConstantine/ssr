# Atom Feature Design for Electrolyte SSR

## Motivation

The original SSR prototype used a small element one-hot vocabulary:

```text
H C N O F Li P S Cl Br
```

This was too narrow for realistic electrolyte and battery-interface data. Training files can contain `B` and `Si`, and common electrolyte salts, solvents, additives, decomposition products, and interphase fragments can also involve elements such as `Na`, `Mg`, `Al`, `K`, `Ca`, and `I`.

If an element is missing from the one-hot vocabulary, it becomes an all-zero vector and is indistinguishable from any other unknown atom. That is chemically unsafe for SSR because B-containing salts/additives and Si-containing interface or silane species have distinct coordination, Lewis acidity, and interphase chemistry.

## Implemented Vocabulary

SSR now uses the following electrolyte-oriented element identity set:

```text
H Li B C N O F Na Mg Al Si P S Cl K Ca Br I
```

This covers common Li-ion electrolyte components and several adjacent battery chemistries:

- carbonate, ether, nitrile, sulfone, and fluorinated solvents: `H C N O F S`
- lithium salts such as LiPF6, LiFSI, LiTFSI, LiBOB, LiBF4, LiClO4: `Li P F S N O C B Cl`
- boron and phosphorus additives/salts: `B P`
- silicon-containing additives or Si-anode/interphase fragments: `Si`
- alternative metal salts or impurities in broader electrolyte design: `Na Mg K Ca Al`
- halogenated salts/additives: `F Cl Br I`

## Physical Atom Features

In addition to one-hot identity, SSR now supports `atom_phys` and `atom_phys_shell` modes. These concatenate normalized elemental properties:

```text
atomic number
atomic mass
Pauling electronegativity
covalent radius
van der Waals radius
periodic-table group
period
nominal valence-electron count
```

These features are common in molecular and materials graph neural networks. They provide useful chemical inductive bias: electronegativity helps distinguish electron-rich and electron-poor atoms, radii affect steric/geometric packing, group/period encode periodic trends, and valence-electron count is relevant to bonding and coordination.

## Feature Modes

Available modes:

```text
element
  one-hot element identity

element_shell
  one-hot element identity + Li-centered shell features

atom_phys
  one-hot element identity + normalized atom properties

atom_phys_shell
  one-hot element identity + normalized atom properties + Li-centered shell features
```

The default configuration now uses:

```yaml
feat_dim: 26
feature_mode: atom_phys_shell
```

`feat_dim: 26` is the base atom feature dimension:

```text
18 one-hot elements + 8 atomic properties
```

`atom_phys_shell` appends three Li-shell features at load time:

```text
is Li
nearest Li distance scaled by cutoff
in first Li shell flag
```

so the actual model input dimension is:

```text
26 + 3 = 29
```

## Why Not Only One-Hot?

One-hot identity is expressive when every element has enough training examples. It is weak for generalization because the model has no prior that chemically similar elements may behave similarly.

Physical atom properties help in low-data and extrapolation settings. For example, if a model has limited examples of `B` or `Si`, their electronegativity, radius, group, period, and valence features still provide useful structure to the representation.

## Limitations

These atom properties are element-level constants. They do not replace:

- partial charge
- oxidation state
- force-field atom type
- molecular identity
- bond topology
- periodic boundary conditions
- local electronic polarization

Those require additional topology, force-field, quantum-chemical, or trajectory metadata. The current implementation is a conservative improvement over one-hot encoding, not a full chemically complete featurization.

## References

- Extended study on atomic featurization in graph neural networks for molecular property prediction: https://pmc.ncbi.nlm.nih.gov/articles/PMC10507875/
- Molecular graph models commonly use atomic number, atomic mass, covalent/vdW radius, valence, electronegativity, and related atom descriptors: https://pmc.ncbi.nlm.nih.gov/articles/PMC13112750/
- Electrolyte design spans common elements including C, N, O, S, F, P, Si, and B: https://www.sciencedirect.com/science/article/pii/S2451929426000902
- Silicon-anode electrolyte reviews discuss salts/additives containing Li, P, F, S, B, H, Cl, and related interphase chemistry: https://www.mdpi.com/2313-0105/11/11/399
