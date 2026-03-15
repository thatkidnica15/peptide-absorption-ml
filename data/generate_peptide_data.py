"""
Peptide Dataset Generator
==========================
Generates synthetic peptide sequences with physicochemical descriptors
and simulated Caco-2 permeability values.

Permeability is modelled as a function of known structure-absorption
relationships from pharmaceutical literature:
- Molecular weight (negative correlation above ~500 Da)
- Lipophilicity (logP: inverted U-shape, optimal ~2-3)
- Polar surface area (negative correlation above ~140 A²)
- Hydrogen bond donors (negative, desolvation penalty)
- Cyclic structure (positive, conformational rigidity)
- N-methylation (positive, reduces H-bond donors)

Author: Veronica Pilagov
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Standard amino acid properties
AMINO_ACIDS = {
    'A': {'name': 'Ala', 'mw': 89.1,  'logp': 0.31,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'V': {'name': 'Val', 'mw': 117.1, 'logp': 1.22,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'L': {'name': 'Leu', 'mw': 131.2, 'logp': 1.70,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'I': {'name': 'Ile', 'mw': 131.2, 'logp': 1.80,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'P': {'name': 'Pro', 'mw': 115.1, 'logp': 0.72,  'hbd': 0, 'hba': 1, 'psa': 49.3,  'charge': 0, 'hydrophobic': False},
    'F': {'name': 'Phe', 'mw': 165.2, 'logp': 1.79,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'W': {'name': 'Trp', 'mw': 204.2, 'logp': 2.25,  'hbd': 2, 'hba': 1, 'psa': 79.1,  'charge': 0, 'hydrophobic': True},
    'M': {'name': 'Met', 'mw': 149.2, 'logp': 1.23,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': True},
    'G': {'name': 'Gly', 'mw': 75.0,  'logp': -0.67, 'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': False},
    'S': {'name': 'Ser', 'mw': 105.1, 'logp': -0.55, 'hbd': 2, 'hba': 2, 'psa': 83.6,  'charge': 0, 'hydrophobic': False},
    'T': {'name': 'Thr', 'mw': 119.1, 'logp': -0.26, 'hbd': 2, 'hba': 2, 'psa': 83.6,  'charge': 0, 'hydrophobic': False},
    'C': {'name': 'Cys', 'mw': 121.2, 'logp': 0.49,  'hbd': 1, 'hba': 1, 'psa': 63.3,  'charge': 0, 'hydrophobic': False},
    'Y': {'name': 'Tyr', 'mw': 181.2, 'logp': 0.96,  'hbd': 2, 'hba': 2, 'psa': 83.6,  'charge': 0, 'hydrophobic': True},
    'N': {'name': 'Asn', 'mw': 132.1, 'logp': -1.03, 'hbd': 2, 'hba': 2, 'psa': 106.4, 'charge': 0, 'hydrophobic': False},
    'Q': {'name': 'Gln', 'mw': 146.1, 'logp': -0.85, 'hbd': 2, 'hba': 2, 'psa': 106.4, 'charge': 0, 'hydrophobic': False},
    'D': {'name': 'Asp', 'mw': 133.1, 'logp': -0.77, 'hbd': 1, 'hba': 3, 'psa': 100.6, 'charge': -1, 'hydrophobic': False},
    'E': {'name': 'Glu', 'mw': 147.1, 'logp': -0.64, 'hbd': 1, 'hba': 3, 'psa': 100.6, 'charge': -1, 'hydrophobic': False},
    'K': {'name': 'Lys', 'mw': 146.2, 'logp': -0.99, 'hbd': 2, 'hba': 2, 'psa': 89.3,  'charge': 1, 'hydrophobic': False},
    'R': {'name': 'Arg', 'mw': 174.2, 'logp': -1.01, 'hbd': 4, 'hba': 3, 'psa': 128.8, 'charge': 1, 'hydrophobic': False},
    'H': {'name': 'His', 'mw': 155.2, 'logp': 0.13,  'hbd': 1, 'hba': 2, 'psa': 92.1,  'charge': 0.5, 'hydrophobic': False},
}


def generate_random_peptide(rng: np.random.Generator, length: int) -> str:
    """Generate a random peptide sequence of given length."""
    residues = list(AMINO_ACIDS.keys())
    return ''.join(rng.choice(residues, size=length))


def compute_descriptors(sequence: str) -> dict:
    """
    Compute physicochemical descriptors for a peptide sequence.
    
    Returns a dictionary of features used for permeability prediction.
    """
    residues = [AMINO_ACIDS[aa] for aa in sequence if aa in AMINO_ACIDS]
    n = len(residues)
    if n == 0:
        return {}
    
    # Molecular weight (sum of residue MWs minus water loss per bond)
    mw = sum(r['mw'] for r in residues) - 18.015 * (n - 1)
    
    # Lipophilicity (additive model, simplified)
    logp = sum(r['logp'] for r in residues) / n
    
    # Hydrogen bond donors and acceptors
    hbd = sum(r['hbd'] for r in residues)
    hba = sum(r['hba'] for r in residues)
    
    # Polar surface area (additive)
    psa = sum(r['psa'] for r in residues)
    
    # Net charge at pH 7.4
    net_charge = sum(r['charge'] for r in residues)
    
    # Hydrophobic fraction
    hydrophobic_frac = sum(1 for r in residues if r['hydrophobic']) / n
    
    # Rotatable bonds (approximation: 2 per residue for backbone)
    rotatable_bonds = 2 * n
    
    # Amino acid composition features
    aa_counts = {}
    for aa in sequence:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1
    
    aromatic_frac = sum(aa_counts.get(aa, 0) for aa in 'FWY') / n
    charged_frac = sum(aa_counts.get(aa, 0) for aa in 'DEKR') / n
    proline_frac = aa_counts.get('P', 0) / n
    
    return {
        'sequence': sequence,
        'length': n,
        'molecular_weight': round(mw, 1),
        'logp_mean': round(logp, 3),
        'hbd_total': hbd,
        'hba_total': hba,
        'psa_total': round(psa, 1),
        'net_charge': round(net_charge, 1),
        'hydrophobic_frac': round(hydrophobic_frac, 3),
        'aromatic_frac': round(aromatic_frac, 3),
        'charged_frac': round(charged_frac, 3),
        'proline_frac': round(proline_frac, 3),
        'rotatable_bonds': rotatable_bonds,
    }


def simulate_permeability(descriptors: dict, rng: np.random.Generator, is_cyclic: bool = False, n_methylations: int = 0) -> float:
    """
    Simulate Caco-2 apparent permeability (log Papp, cm/s).
    
    Based on published structure-permeability relationships:
    
    1. MW penalty: permeability decreases sharply above 500 Da
       (Lipinski's rule, supported by Caco-2 data)
    
    2. LogP optimum: inverted U-shape with peak at ~2-3
       (too polar = can't partition; too lipophilic = trapped in membrane)
    
    3. PSA penalty: above 140 A², passive transcellular transport drops
       (Veber et al., J Med Chem 2002)
    
    4. HBD penalty: each donor increases desolvation energy
       (Rezai et al., JACS 2006)
    
    5. Cyclisation bonus: reduces conformational entropy,
       shields backbone NH from solvent
       (Cyclosporin A effect, Driggers et al., Nat Rev Drug Discov 2008)
    
    6. N-methylation bonus: removes backbone NH donors
       (Chatterjee et al., Acc Chem Res 2008)
    
    Returns log10(Papp) where Papp is in cm/s.
    Typical range: -7 (impermeable) to -4 (highly permeable).
    """
    mw = descriptors['molecular_weight']
    logp = descriptors['logp_mean']
    psa = descriptors['psa_total']
    hbd = descriptors['hbd_total']
    
    # Baseline: moderate permeability
    log_papp = -5.5
    
    # MW effect (penalty above 500 Da, sigmoid)
    mw_penalty = -1.5 / (1 + np.exp(-0.005 * (mw - 700)))
    log_papp += mw_penalty
    
    # LogP effect (inverted U, optimal around 2)
    logp_effect = -0.3 * (logp - 2.0) ** 2 + 0.5
    log_papp += np.clip(logp_effect, -1.0, 0.5)
    
    # PSA penalty (above 140 A²)
    psa_penalty = -0.8 / (1 + np.exp(-0.01 * (psa - 800)))
    log_papp += psa_penalty
    
    # HBD penalty (-0.1 per donor above 4)
    hbd_penalty = -0.1 * max(0, hbd - 4)
    log_papp += hbd_penalty
    
    # Cyclisation bonus
    if is_cyclic:
        log_papp += 0.5
    
    # N-methylation bonus (per methylation)
    log_papp += 0.15 * n_methylations
    
    # Hydrophobic fraction effect
    hf = descriptors['hydrophobic_frac']
    log_papp += 0.3 * (hf - 0.3)  # slightly positive for more hydrophobic
    
    # Proline effect (conformational rigidity)
    log_papp += 0.1 * descriptors['proline_frac']
    
    # Noise (biological variability in Caco-2 assays)
    noise = rng.normal(0, 0.2)
    log_papp += noise
    
    # Clip to realistic range
    return round(np.clip(log_papp, -7.5, -3.5), 3)


def generate_dataset(n_samples: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a dataset of peptides with descriptors and permeability.
    
    Peptide lengths follow a realistic distribution (4-15 residues),
    matching the therapeutic peptide design space.
    """
    rng = np.random.default_rng(seed)
    
    records = []
    for i in range(n_samples):
        # Length distribution: peaked at 6-10 residues
        length = rng.choice(range(4, 16), p=_length_distribution())
        
        sequence = generate_random_peptide(rng, length)
        descriptors = compute_descriptors(sequence)
        
        if not descriptors:
            continue
        
        # Randomly assign cyclisation (~20%) and N-methylation (~15%)
        is_cyclic = rng.random() < 0.20
        n_methyl = rng.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.10, 0.05])
        
        log_papp = simulate_permeability(descriptors, rng, is_cyclic, n_methyl)
        
        descriptors['peptide_id'] = f'PEP_{i:04d}'
        descriptors['is_cyclic'] = int(is_cyclic)
        descriptors['n_methylations'] = n_methyl
        descriptors['log_papp'] = log_papp
        
        # Classification label: permeable if log_papp > -5.5
        descriptors['permeable'] = int(log_papp > -5.5)
        
        records.append(descriptors)
    
    df = pd.DataFrame(records)
    
    # Reorder columns
    col_order = ['peptide_id', 'sequence', 'length', 'molecular_weight', 'logp_mean',
                 'hbd_total', 'hba_total', 'psa_total', 'net_charge', 'hydrophobic_frac',
                 'aromatic_frac', 'charged_frac', 'proline_frac', 'rotatable_bonds',
                 'is_cyclic', 'n_methylations', 'log_papp', 'permeable']
    df = df[col_order]
    
    return df


def _length_distribution():
    """Probability distribution for peptide lengths 4-15."""
    # Peaked at 7-9 residues (therapeutic sweet spot)
    probs = np.array([0.05, 0.08, 0.12, 0.18, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02, 0.01, 0.01])
    return probs / probs.sum()


if __name__ == '__main__':
    output_dir = Path(__file__).parent
    df = generate_dataset(n_samples=1500)
    
    output_path = output_dir / 'peptide_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated: {len(df)} peptides")
    print(f"Permeable fraction: {df['permeable'].mean():.1%}")
    print(f"Cyclic fraction: {df['is_cyclic'].mean():.1%}")
    print(f"Length range: {df['length'].min()}-{df['length'].max()} residues")
    print(f"MW range: {df['molecular_weight'].min():.0f}-{df['molecular_weight'].max():.0f} Da")
    print(f"log Papp range: {df['log_papp'].min():.2f} to {df['log_papp'].max():.2f}")
    print(f"\nSaved to: {output_path}")
