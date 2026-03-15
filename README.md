# PeptideAbsorb

Predicting oral peptide absorption and designing analogues with improved bioavailability using machine learning.

## Motivation

Over 95% of therapeutic peptides fail as oral drugs because they are degraded in the GI tract or cannot cross the intestinal epithelium. During my quality control traineeship at Somaí Pharmaceuticals, I observed firsthand how formulation microstructure (dissolution kinetics, particle-size distribution, excipient interactions) determines whether an active compound reaches its biological target. This project applies machine learning to the same problem at the molecular level: given a peptide's sequence and physicochemical properties, can we predict its intestinal permeability and suggest modifications that improve oral absorption?

This is directly relevant to nanomedicine, where lipid nanoparticle and polymer-based carriers are designed to protect and deliver peptides. Understanding which molecular features govern absorption helps optimise both the payload and the carrier.

## What This Project Does

1. **Generates a peptide dataset** with clinically relevant physicochemical descriptors (molecular weight, hydrophobicity, charge, hydrogen bond donors/acceptors, polar surface area, rotatable bonds, cyclic structure, N-methylation) and simulated Caco-2 permeability values based on published structure-permeability relationships

2. **Engineers features** grounded in pharmaceutical science:
   - Lipinski's Rule of Five compliance (oral druglikeness)
   - Veber's rules (flexibility and polar surface area)
   - Amino acid composition profiles
   - Hydrophobic moment and amphipathicity indices
   - Peptide bond stability estimators

3. **Trains three models** to predict intestinal permeability:
   - **Gradient Boosted Trees** (XGBoost-style via sklearn) for interpretable feature importance
   - **Random Forest** with permutation importance
   - **Neural Network** (from scratch in NumPy) demonstrating understanding of regression with continuous targets

4. **Analyses structure-permeability relationships** through SHAP-style feature attribution, partial dependence plots, and permeability landscape visualisations

5. **Generates peptide analogues** by systematic single-residue substitutions, scoring each analogue's predicted permeability improvement, and ranking modifications by therapeutic potential

## Technical Stack

- **Python 3.10+**
- **NumPy** — numerical computation, neural network from scratch
- **pandas** — data manipulation
- **scikit-learn** — gradient boosting, random forest, evaluation
- **matplotlib / seaborn** — visualisations
- **SciPy** — statistical analysis

## Project Structure

```
peptide-absorption-ml/
├── README.md
├── requirements.txt
├── data/
│   └── generate_peptide_data.py     # Synthetic peptide dataset
├── src/
│   ├── __init__.py
│   ├── descriptors.py               # Physicochemical descriptor calculation
│   ├── feature_engineering.py       # Pharmaceutical feature construction
│   ├── models.py                    # GBT, RF, and NN regression models
│   ├── analogue_generator.py        # Peptide modification and scoring
│   ├── evaluation.py                # Regression metrics and visualisations
│   └── permeability_rules.py        # Lipinski, Veber, and custom rules
├── figures/                          # Generated plots
├── tests/
│   └── test_pipeline.py
└── main.py                           # End-to-end pipeline
```

## Key Design Decisions

**Why predict Caco-2 permeability?** The Caco-2 cell monolayer assay is the industry standard for measuring intestinal epithelial permeability. Papp (apparent permeability coefficient, cm/s) correlates with human oral absorption fraction. log(Papp) > -5 generally indicates good absorption potential.

**Why physicochemical descriptors, not molecular fingerprints?** For peptides specifically, physicochemical properties (MW, LogP, PSA, H-bond count) have stronger mechanistic links to absorption than structural fingerprints. Each feature maps to a known biological barrier: MW > 500 Da impedes passive transcellular diffusion; high PSA reduces membrane partitioning; excess H-bond donors increase desolvation penalty.

**Why generate analogues?** The clinical value of absorption prediction is in guiding peptide optimisation. Single-residue substitutions (e.g., replacing Ser with N-methylated Ala to reduce H-bond donors) are the most common medicinal chemistry strategy for improving oral peptide bioavailability. Ranking these computationally saves months of wet-lab screening.

## Usage

```bash
python data/generate_peptide_data.py   # Generate dataset
python main.py                          # Run full pipeline
python -m pytest tests/                 # Run tests
```

## Results

Expected performance on held-out test set:
- Gradient Boosted Trees: R² ~0.85, MAE ~0.35 log units
- Random Forest: R² ~0.82, MAE ~0.38 log units
- Neural Network (NumPy): R² ~0.78, MAE ~0.42 log units

## Domain Context

This project sits at the intersection of three fields:
- **Computational pharmacology**: predicting ADME properties from molecular structure
- **Peptide therapeutics**: the fastest-growing drug class (GLP-1 agonists, insulin analogues)
- **Nanomedicine**: nanoparticle carriers designed to overcome the barriers this model quantifies

Understanding which molecular features limit oral absorption directly informs the design of LNP, PLGA, and chitosan nanoparticle delivery systems for peptide therapeutics.

## Author

**Veronica Pilagov** — BSc Medical Innovation & Enterprise, UCL  
Background in pharmaceutical quality control, tissue engineering, and AI for healthcare.

## License

MIT
