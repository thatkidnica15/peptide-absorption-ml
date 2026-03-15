"""
Peptide Analogue Generator
============================
Generates modified peptide sequences and predicts their improved
oral absorption using the trained permeability model.

Modification strategies from medicinal chemistry:
1. Single-residue substitutions (replace polar with hydrophobic)
2. N-methylation (reduce H-bond donors)
3. Cyclisation (reduce conformational flexibility)
4. D-amino acid substitution (increase protease resistance)

Author: Veronica Pilagov
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from data.generate_peptide_data import compute_descriptors, AMINO_ACIDS


# Residue substitution rules (medicinal chemistry heuristics)
FAVOURABLE_SUBSTITUTIONS = {
    # Reduce polarity: replace polar with hydrophobic
    'S': ['A', 'V'],     # Ser -> Ala/Val (remove OH)
    'T': ['V', 'I'],     # Thr -> Val/Ile (remove OH)
    'N': ['L', 'A'],     # Asn -> Leu/Ala (remove amide)
    'Q': ['L', 'M'],     # Gln -> Leu/Met (remove amide)
    # Reduce charge
    'K': ['A', 'L'],     # Lys -> Ala/Leu (remove amine)
    'R': ['L', 'F'],     # Arg -> Leu/Phe (remove guanidinium)
    'D': ['A', 'N'],     # Asp -> Ala/Asn (remove carboxyl)
    'E': ['Q', 'A'],     # Glu -> Gln/Ala (remove carboxyl)
    # Increase rigidity
    'G': ['P', 'A'],     # Gly -> Pro/Ala (reduce flexibility)
}


def generate_single_substitutions(
    sequence: str,
    model_predict: Callable,
    feature_engineer: Callable,
    feature_cols: List[str],
    parent_papp: float,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Generate all single-residue substitutions and rank by
    predicted permeability improvement.
    
    Parameters
    ----------
    sequence : str
        Parent peptide sequence.
    model_predict : callable
        Trained model's predict function.
    feature_engineer : callable
        Feature engineering pipeline function.
    feature_cols : list
        Feature column names for the model.
    parent_papp : float
        Parent peptide's predicted log(Papp).
    top_k : int
        Number of top analogues to return.
    
    Returns
    -------
    pd.DataFrame
        Ranked analogues with predicted improvement.
    """
    analogues = []
    
    for pos in range(len(sequence)):
        original_aa = sequence[pos]
        
        # Try favourable substitutions first, then all others
        candidates = FAVOURABLE_SUBSTITUTIONS.get(original_aa, [])
        # Add all other amino acids not already in candidates
        all_aa = [aa for aa in AMINO_ACIDS.keys() if aa != original_aa and aa not in candidates]
        candidates = candidates + all_aa
        
        for new_aa in candidates:
            # Create mutant sequence
            mutant = sequence[:pos] + new_aa + sequence[pos+1:]
            
            # Compute descriptors
            desc = compute_descriptors(mutant)
            if not desc:
                continue
            
            # Add modification features (inherit parent's cyclisation/methylation)
            desc['is_cyclic'] = 0
            desc['n_methylations'] = 0
            
            analogues.append({
                'position': pos + 1,
                'original_aa': original_aa,
                'new_aa': new_aa,
                'mutant_sequence': mutant,
                'substitution': f'{original_aa}{pos+1}{new_aa}',
                **{k: desc[k] for k in desc if k not in ['sequence', 'peptide_id']},
            })
    
    if not analogues:
        return pd.DataFrame()
    
    analogue_df = pd.DataFrame(analogues)
    
    # Engineer features
    analogue_df = feature_engineer(analogue_df)
    
    # Predict permeability
    X = analogue_df[feature_cols].values
    analogue_df['predicted_log_papp'] = model_predict(X)
    analogue_df['delta_log_papp'] = analogue_df['predicted_log_papp'] - parent_papp
    analogue_df['papp_fold_change'] = 10 ** analogue_df['delta_log_papp']
    
    # Classify improvement
    analogue_df['improvement'] = pd.cut(
        analogue_df['delta_log_papp'],
        bins=[-np.inf, -0.1, 0.1, 0.3, np.inf],
        labels=['worse', 'neutral', 'improved', 'strongly_improved'],
    )
    
    # Sort by improvement
    analogue_df = analogue_df.sort_values('delta_log_papp', ascending=False)
    
    return analogue_df.head(top_k)


def suggest_modifications(sequence: str, descriptors: dict) -> List[Dict]:
    """
    Suggest high-level modification strategies based on
    which physicochemical barriers limit the peptide's absorption.
    
    Returns prioritised list of strategies with rationale.
    """
    suggestions = []
    
    mw = descriptors.get('molecular_weight', 0)
    hbd = descriptors.get('hbd_total', 0)
    psa = descriptors.get('psa_total', 0)
    logp = descriptors.get('logp_mean', 0)
    charged = descriptors.get('charged_frac', 0)
    
    if mw > 700:
        suggestions.append({
            'strategy': 'Truncation',
            'rationale': f'MW ({mw:.0f} Da) exceeds optimal range. Consider removing non-essential residues.',
            'priority': 'high',
        })
    
    if hbd > 6:
        suggestions.append({
            'strategy': 'N-methylation',
            'rationale': f'{hbd} H-bond donors exceed threshold. N-methylation of backbone NH groups reduces desolvation penalty.',
            'priority': 'high',
        })
    
    if psa > 800:
        suggestions.append({
            'strategy': 'Polar-to-hydrophobic substitution',
            'rationale': f'PSA ({psa:.0f} A²) is high. Replace Ser/Thr/Asn with Ala/Val/Leu at non-essential positions.',
            'priority': 'medium',
        })
    
    if not descriptors.get('is_cyclic', False) and len(sequence) >= 5:
        suggestions.append({
            'strategy': 'Backbone cyclisation',
            'rationale': 'Cyclisation reduces conformational entropy and shields backbone amides from solvent.',
            'priority': 'medium',
        })
    
    if charged > 0.3:
        suggestions.append({
            'strategy': 'Charge reduction',
            'rationale': f'Charged residue fraction ({charged:.0%}) is high. Replace Lys/Arg/Asp/Glu with neutral alternatives.',
            'priority': 'medium',
        })
    
    if logp < 0:
        suggestions.append({
            'strategy': 'Increase lipophilicity',
            'rationale': f'LogP ({logp:.2f}) is too low for membrane partitioning. Introduce Phe/Trp/Leu.',
            'priority': 'low',
        })
    
    return suggestions
