"""
Pharmaceutical Feature Engineering
====================================
Constructs features grounded in medicinal chemistry rules for oral absorption.

Author: Veronica Pilagov
"""

import numpy as np
import pandas as pd


def add_lipinski_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lipinski's Rule of Five (Lipinski et al., Adv Drug Deliv Rev 1997).
    
    Compounds are likely orally bioavailable if they satisfy:
    - MW <= 500 Da
    - LogP <= 5
    - HBD <= 5
    - HBA <= 10
    
    Each violation reduces the probability of oral absorption.
    For peptides, most violate MW (peptides are >500 Da),
    making the violation count a meaningful continuous feature.
    """
    df = df.copy()
    df['lipinski_mw_ok'] = (df['molecular_weight'] <= 500).astype(int)
    df['lipinski_logp_ok'] = (df['logp_mean'] <= 5).astype(int)
    df['lipinski_hbd_ok'] = (df['hbd_total'] <= 5).astype(int)
    df['lipinski_hba_ok'] = (df['hba_total'] <= 10).astype(int)
    df['lipinski_violations'] = 4 - (
        df['lipinski_mw_ok'] + df['lipinski_logp_ok'] +
        df['lipinski_hbd_ok'] + df['lipinski_hba_ok']
    )
    return df


def add_veber_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veber's rules for oral bioavailability (Veber et al., J Med Chem 2002).
    
    Good oral bioavailability correlates with:
    - PSA <= 140 A² (or <= 12 total H-bonds)
    - Rotatable bonds <= 10
    
    These capture molecular flexibility and polarity,
    which determine membrane permeation rate.
    """
    df = df.copy()
    df['veber_psa_ok'] = (df['psa_total'] <= 140).astype(int)
    df['veber_rotbonds_ok'] = (df['rotatable_bonds'] <= 10).astype(int)
    df['veber_compliant'] = (df['veber_psa_ok'] & df['veber_rotbonds_ok']).astype(int)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features capturing known structure-absorption relationships.
    """
    df = df.copy()
    
    # MW per residue (normalised size)
    df['mw_per_residue'] = (df['molecular_weight'] / df['length']).round(1)
    
    # HBD density (donors per unit molecular weight)
    df['hbd_density'] = (df['hbd_total'] / df['molecular_weight'] * 100).round(3)
    
    # PSA per residue (normalised polarity)
    df['psa_per_residue'] = (df['psa_total'] / df['length']).round(1)
    
    # Charge-to-size ratio
    df['charge_density'] = (np.abs(df['net_charge']) / df['length']).round(3)
    
    # LogP deviation from optimum (2.0)
    df['logp_deviation'] = ((df['logp_mean'] - 2.0) ** 2).round(3)
    
    # Amphipathicity index (balance of hydrophobic and polar)
    df['amphipathicity'] = (df['hydrophobic_frac'] * (1 - df['hydrophobic_frac']) * 4).round(3)
    
    # Combined modification score (cyclisation + N-methylation)
    df['modification_score'] = df['is_cyclic'] * 0.5 + df['n_methylations'] * 0.15
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = add_lipinski_features(df)
    df = add_veber_features(df)
    df = add_derived_features(df)
    return df


# Feature columns for model training (excludes identifiers and target)
FEATURE_COLS = [
    'length', 'molecular_weight', 'logp_mean', 'hbd_total', 'hba_total',
    'psa_total', 'net_charge', 'hydrophobic_frac', 'aromatic_frac',
    'charged_frac', 'proline_frac', 'rotatable_bonds', 'is_cyclic',
    'n_methylations', 'lipinski_violations', 'veber_compliant',
    'mw_per_residue', 'hbd_density', 'psa_per_residue', 'charge_density',
    'logp_deviation', 'amphipathicity', 'modification_score',
]
