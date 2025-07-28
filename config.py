# config.py
"""
Configuration management for layout optimization.

Consolidates all configuration loading, validation, and preprocessing logic.
"""

import yaml
import os
from typing import Dict, Set
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Structured configuration for optimization parameters."""
    # Items and positions
    items_to_assign: str
    positions_to_assign: str
    items_to_constrain: str = ""
    positions_to_constrain: str = ""
    items_assigned: str = ""
    positions_assigned: str = ""
    
    # Validation properties
    @property
    def items_to_assign_set(self) -> Set[str]:
        return set(self.items_to_assign.lower())
    
    @property
    def positions_to_assign_set(self) -> Set[str]:
        return set(self.positions_to_assign.upper())
    
    @property
    def items_to_constrain_set(self) -> Set[str]:
        return set(self.items_to_constrain.lower()) if self.items_to_constrain else set()
    
    @property
    def positions_to_constrain_set(self) -> Set[str]:
        return set(self.positions_to_constrain.upper()) if self.positions_to_constrain else set()
    
    @property
    def items_assigned_set(self) -> Set[str]:
        return set(self.items_assigned.lower()) if self.items_assigned else set()
    
    @property
    def positions_assigned_set(self) -> Set[str]:
        return set(self.positions_assigned.upper()) if self.positions_assigned else set()

@dataclass  
class PathConfig:
    """File paths for input and output."""
    # Input paths
    raw_item_scores_file: str
    raw_item_pair_scores_file: str
    raw_position_scores_file: str
    raw_position_pair_scores_file: str
    
    # Output paths
    layout_results_folder: str

    # Auto-generated normalized paths (populated in __post_init__)
    item_scores_file: str = None
    item_pair_scores_file: str = None
    position_scores_file: str = None
    position_pair_scores_file: str = None
    
    def __post_init__(self):
        """Auto-generate normalized file paths."""
        normalized_dir = "output/normalized_input"
        
        # Auto-generate normalized paths with standard names
        self.item_scores_file = f"{normalized_dir}/normalized_item_scores.csv" 
        self.item_pair_scores_file = f"{normalized_dir}/normalized_item_pair_scores.csv"
        self.position_scores_file = f"{normalized_dir}/normalized_position_scores.csv"
        self.position_pair_scores_file = f"{normalized_dir}/normalized_position_pair_scores.csv"

@dataclass
class VisualizationConfig:
    """Visualization and display settings."""
    print_keyboard: bool = False

@dataclass
class Config:
    """Complete configuration container."""
    paths: PathConfig
    optimization: OptimizationConfig
    visualization: VisualizationConfig
    
    # Internal
    _config_path: str = "config.yaml"

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated Config object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    # Create output directories (including normalized input dir)
    normalized_dir = "output/normalized_input"
    output_dirs = [raw_config['paths']['output']['layout_results_folder'], normalized_dir]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Parse paths - only pass the raw input paths and output paths
    paths = PathConfig(
        **raw_config['paths']['input'], 
        **raw_config['paths']['output']
    )
    
    # Normalize optimization strings to proper case
    opt_raw = raw_config['optimization']
    optimization = OptimizationConfig(
        items_to_assign=opt_raw.get('items_to_assign', '').lower(),
        positions_to_assign=opt_raw.get('positions_to_assign', '').upper(),
        items_to_constrain=opt_raw.get('items_to_constrain', '').lower(),
        positions_to_constrain=opt_raw.get('positions_to_constrain', '').upper(),
        items_assigned=opt_raw.get('items_assigned', '').lower(),
        positions_assigned=opt_raw.get('positions_assigned', '').upper()
    )
    
    visualization = VisualizationConfig(**raw_config.get('visualization', {}))
    
    config = Config(paths, optimization, visualization, config_path)
    
    # Validate configuration
    validate_config(config)
    
    return config

def validate_config(config: Config) -> None:
    """
    Comprehensive configuration validation.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    opt = config.optimization
    
    # Check for duplicates within each string
    def check_duplicates(items: str, name: str):
        if len(set(items)) != len(items):
            raise ValueError(f"Duplicate characters in {name}: '{items}'")
    
    check_duplicates(opt.items_to_assign, "items_to_assign")
    check_duplicates(opt.positions_to_assign, "positions_to_assign") 
    check_duplicates(opt.items_assigned, "items_assigned")
    check_duplicates(opt.positions_assigned, "positions_assigned")
    
    # Check matching lengths for assigned items/positions
    if len(opt.items_assigned) != len(opt.positions_assigned):
        raise ValueError(
            f"Mismatched assigned items ({len(opt.items_assigned)}) "
            f"and positions ({len(opt.positions_assigned)})")
    
    # Check no overlap between assigned and to_assign  
    items_overlap = opt.items_assigned_set.intersection(opt.items_to_assign_set)
    if items_overlap:
        raise ValueError(f"items_to_assign overlaps with items_assigned: {items_overlap}")
    
    positions_overlap = opt.positions_assigned_set.intersection(opt.positions_to_assign_set)
    if positions_overlap:
        raise ValueError(f"positions_to_assign overlaps with positions_assigned: {positions_overlap}")
    
    # Check sufficient positions
    if len(opt.items_to_assign) > len(opt.positions_to_assign):
        raise ValueError(
            f"More items to assign ({len(opt.items_to_assign)}) "
            f"than available positions ({len(opt.positions_to_assign)})")
    
    # Validate constraints are subsets
    if not opt.items_to_constrain_set.issubset(opt.items_to_assign_set):
        invalid = opt.items_to_constrain_set - opt.items_to_assign_set
        raise ValueError(f"items_to_constrain contains invalid items: {invalid}")
    
    if not opt.positions_to_constrain_set.issubset(opt.positions_to_assign_set):
        invalid = opt.positions_to_constrain_set - opt.positions_to_assign_set  
        raise ValueError(f"positions_to_constrain contains invalid positions: {invalid}")
    
    # Check sufficient constraint positions
    if len(opt.items_to_constrain) > len(opt.positions_to_constrain):
        raise ValueError(
            f"Not enough constraint positions ({len(opt.positions_to_constrain)}) "
            f"for constraint items ({len(opt.items_to_constrain)})")

def print_config_summary(config: Config) -> None:
    """Print a human-readable configuration summary."""
    opt = config.optimization
    
    print("\nConfiguration Summary:")
    print(f"  Items to assign ({len(opt.items_to_assign)}): {opt.items_to_assign}")
    print(f"  Available positions ({len(opt.positions_to_assign)}): {opt.positions_to_assign}")
    
    if opt.items_to_constrain:
        print(f"  Items to constrain ({len(opt.items_to_constrain)}): {opt.items_to_constrain}")
        print(f"  Constraint positions ({len(opt.positions_to_constrain)}): {opt.positions_to_constrain}")
    
    if opt.items_assigned:
        print(f"  Pre-assigned items ({len(opt.items_assigned)}): {opt.items_assigned}")
        print(f"  Pre-assigned positions ({len(opt.positions_assigned)}): {opt.positions_assigned}")
    
    print(f"  Print keyboard: {config.visualization.print_keyboard}")