"""
Models Module
=============
Contains data models and classes used across simulations.
"""


class Store:
    """
    Represents a single retail store in the omni-channel network.
    
    Attributes:
        id (int): Unique identifier for the store
        type (str): Store type ('small', 'medium', or 'large')
        mu_demand (float): Average daily in-store demand (Poisson parameter)
        inventory (int): Current inventory level
        outstanding_order (int): Quantity of outstanding replenishment order
    """
    
    def __init__(self, id, type, mu_demand, initial_inventory):
        self.id = id
        self.type = type
        self.mu_demand = mu_demand
        self.inventory = initial_inventory
        self.outstanding_order = 0
    
    def __repr__(self):
        return f"Store {self.id} ({self.type}): Inv={self.inventory}, Q={self.outstanding_order}"
    
    def copy(self):
        """Create a copy of the store for policy comparison."""
        store_copy = Store(self.id, self.type, self.mu_demand, self.inventory)
        store_copy.outstanding_order = self.outstanding_order
        return store_copy
