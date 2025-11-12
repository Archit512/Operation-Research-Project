"""
Simulation Package
==================
This package contains simulation models, policies, and utilities for
omni-channel fulfillment optimization.

Modules:
    models - Data models (Store class)
    policies - Fulfillment policies (MFP, TMFP, OSIP)
    utils - Helper functions for calculations
    
Simulation Scripts:
    sim_theoretical - Base test case simulation
    sim_real_data - Real product data simulation  
    sim_corrected - Corrected demand distribution simulation
    policy_comparison - Compare policy performance with profit metrics
"""

from .models import Store
from .policies import mfp_fulfillment_policy, tmfp_fulfillment_policy, osip_fulfillment_policy
from .utils import (
    calculate_expected_contribution_next_day,
    calculate_threshold,
    get_days_until_replenishment,
    dummy_value_function_vj
)

__all__ = [
    'Store',
    'mfp_fulfillment_policy',
    'tmfp_fulfillment_policy',
    'osip_fulfillment_policy',
    'calculate_expected_contribution_next_day',
    'calculate_threshold',
    'get_days_until_replenishment',
    'dummy_value_function_vj',
]
