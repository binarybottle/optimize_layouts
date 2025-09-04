#!/usr/bin/env python3
"""
Configuration Management for Multi-Objective Layout Optimization

This module provides structured configuration loading, validation, 
and management specifically designed for MOO layout optimization. 
It handles all configuration aspects including items/positions 
to optimize, constraints, file paths, and optimization parameters.

Features:
- YAML-based configuration with comprehensive validation
- Support for partial assignments and constraints
- Automatic path resolution for input/output files
- Clear error messages for configuration issues

"""

import yaml
import os
from typing import Set, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OptimizationConfig:
    """Configuration for optimization parameters and constraints."""
    
    # Core optimization settings
    items_to_assign: str
    positions_to_assign: str
    
    # Pre-assignments (items already placed)
    items_assigned: str = ""
    positions_assigned: str = ""
    
    # Constraints (subset restrictions)
    items_to_constrain: str = ""
    positions_to_constrain: str = ""
    
    def __post_init__(self):
        """Normalize strings to proper case."""
        self.items_to_assign = self.items_to_assign.lower()
        self.positions_to_assign = self.positions_to_assign.upper()
        self.items_assigned = self.items_assigned.lower()
        self.positions_assigned = self.positions_assigned.upper()
        self.items_to_constrain = self.items_to_constrain.lower()
        self.positions_to_constrain = self.positions_to_constrain.upper()
    
    # Set-based properties for validation and lookup
    @property
    def items_to_assign_set(self) -> Set[str]:
        return set(self.items_to_assign)
    
    @property
    def positions_to_assign_set(self) -> Set[str]:
        return set(self.positions_to_assign)
    
    @property
    def items_assigned_set(self) -> Set[str]:
        return set(self.items_assigned) if self.items_assigned else set()
    
    @property
    def positions_assigned_set(self) -> Set[str]:
        return set(self.positions_assigned) if self.positions_assigned else set()
    
    @property
    def items_to_constrain_set(self) -> Set[str]:
        return set(self.items_to_constrain) if self.items_to_constrain else set()
    
    @property
    def positions_to_constrain_set(self) -> Set[str]:
        return set(self.positions_to_constrain) if self.positions_to_constrain else set()

    
@dataclass
class PathConfig:
    """File paths for input and output."""
    position_pair_score_table: str
    item_pair_score_table: str
    layout_results_folder: str

    
@dataclass
class MOOConfig:
    """Multi-objective optimization specific configuration."""
    
    # Default optimization settings
    default_objectives: List[str]
    default_weights: Optional[List[float]] = None
    default_maximize: Optional[List[bool]] = None
    
    # Search limits (can be numbers or "Inf")
    default_max_solutions: Union[int, str] = 100
    default_time_limit: Union[float, str] = 3600.0
    
    # Default file paths for validation
    default_position_pair_score_table: str = "input/keypair_engram7_scores.csv"
    default_item_pair_score_table: str = "input/frequency/normalized-english-letter-pair-counts-google-ngrams.csv"
    
    # Progress and output settings
    show_progress_bar: bool = True
    save_detailed_results: bool = True


@dataclass
class VisualizationConfig:
    """Visualization and display settings."""
    print_keyboard: bool = False
    verbose_output: bool = False


@dataclass
class Config:
    """Complete configuration container."""
    paths: PathConfig
    optimization: OptimizationConfig
    moo: MOOConfig
    visualization: VisualizationConfig
    
    # Internal tracking
    _config_path: str = "config.yaml"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Validated Config object with all settings
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    
    if not raw_config:
        raise ValueError("Configuration file is empty")
    
    # Validate required sections exist
    required_sections = ['paths', 'optimization']
    missing_sections = [section for section in required_sections if section not in raw_config]
    if missing_sections:
        raise ValueError(f"Missing required configuration sections: {missing_sections}")
    
    # Create output directories
    normalized_dir = "output/normalized_input"
    layout_results_dir = raw_config['paths']['layout_results_folder']
    
    for directory in [layout_results_dir, normalized_dir]:
        os.makedirs(directory, exist_ok=True)

    # Parse configuration sections
    try:
        paths = PathConfig(
            position_pair_score_table=raw_config['paths']['position_pair_score_table'],
            item_pair_score_table=raw_config['paths']['item_pair_score_table'],
            layout_results_folder=raw_config['paths']['layout_results_folder']
        )
    except (KeyError, TypeError) as e:
        raise ValueError(f"Error parsing paths configuration: {e}")
    
    try:
        optimization = OptimizationConfig(**raw_config['optimization'])
    except (KeyError, TypeError) as e:
        raise ValueError(f"Error parsing optimization configuration: {e}")
    
    # Optional sections with defaults
    moo_config = raw_config.get('moo', {})
    try:
        moo = MOOConfig(**moo_config)
    except TypeError as e:
        raise ValueError(f"Error parsing MOO configuration: {e}")
    
    visualization_config = raw_config.get('visualization', {})
    try:
        visualization = VisualizationConfig(**visualization_config)
    except TypeError as e:
        raise ValueError(f"Error parsing visualization configuration: {e}")
    
    # Create complete configuration object
    config = Config(paths, optimization, moo, visualization, config_path)
    
    # Validate the complete configuration
    validate_config(config)
    
    return config


def validate_config(config: Config) -> None:
    """
    Perform comprehensive validation of configuration.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If any validation check fails
    """
    opt = config.optimization
    
    # Check for empty required fields
    if not opt.items_to_assign:
        raise ValueError("items_to_assign cannot be empty")
    if not opt.positions_to_assign:
        raise ValueError("positions_to_assign cannot be empty")
    
    # Check for duplicate characters within strings
    def check_duplicates(items: str, name: str):
        if items and len(set(items)) != len(items):
            duplicates = [char for char in set(items) if items.count(char) > 1]
            raise ValueError(f"Duplicate characters in {name}: '{items}' (duplicates: {duplicates})")
    
    check_duplicates(opt.items_to_assign, "items_to_assign")
    check_duplicates(opt.positions_to_assign, "positions_to_assign") 
    check_duplicates(opt.items_assigned, "items_assigned")
    check_duplicates(opt.positions_assigned, "positions_assigned")
    check_duplicates(opt.items_to_constrain, "items_to_constrain")
    check_duplicates(opt.positions_to_constrain, "positions_to_constrain")
    
    # Check matching lengths for pre-assigned items/positions
    if len(opt.items_assigned) != len(opt.positions_assigned):
        raise ValueError(
            f"Mismatched pre-assigned items ({len(opt.items_assigned)}) "
            f"and positions ({len(opt.positions_assigned)}): "
            f"'{opt.items_assigned}' vs '{opt.positions_assigned}'"
        )
    
    # Check no overlap between pre-assigned and items to assign
    items_overlap = opt.items_assigned_set.intersection(opt.items_to_assign_set)
    if items_overlap:
        raise ValueError(f"items_to_assign overlaps with items_assigned: {items_overlap}")
    
    positions_overlap = opt.positions_assigned_set.intersection(opt.positions_to_assign_set)
    if positions_overlap:
        raise ValueError(f"positions_to_assign overlaps with positions_assigned: {positions_overlap}")
    
    # Check sufficient positions for items
    total_items = len(opt.items_to_assign) + len(opt.items_assigned)
    total_positions = len(opt.positions_to_assign) + len(opt.positions_assigned)
    
    if total_items > total_positions:
        raise ValueError(
            f"Insufficient positions: need {total_items} total positions "
            f"but only have {total_positions} available"
        )
    
    if len(opt.items_to_assign) > len(opt.positions_to_assign):
        raise ValueError(
            f"More items to assign ({len(opt.items_to_assign)}) "
            f"than available positions ({len(opt.positions_to_assign)})"
        )
    
    # Validate constraints are subsets of items/positions to assign
    if not opt.items_to_constrain_set.issubset(opt.items_to_assign_set):
        invalid = opt.items_to_constrain_set - opt.items_to_assign_set
        raise ValueError(f"items_to_constrain contains invalid items not in items_to_assign: {invalid}")
    
    if not opt.positions_to_constrain_set.issubset(opt.positions_to_assign_set):
        invalid = opt.positions_to_constrain_set - opt.positions_to_assign_set  
        raise ValueError(f"positions_to_constrain contains invalid positions not in positions_to_assign: {invalid}")
    
    # Check sufficient constraint positions for constraint items
    if len(opt.items_to_constrain) > len(opt.positions_to_constrain):
        raise ValueError(
            f"Not enough constraint positions ({len(opt.positions_to_constrain)}) "
            f"for constraint items ({len(opt.items_to_constrain)}): "
            f"items='{opt.items_to_constrain}' positions='{opt.positions_to_constrain}'"
        )
    
    # Check that we have at least 2 items for meaningful optimization
    if len(opt.items_to_assign) < 2:
        raise ValueError("Need at least 2 items for meaningful optimization")
    
    # Validate MOO configuration
    if config.moo.default_max_solutions <= 0:
        raise ValueError("default_max_solutions must be positive")
    
    if config.moo.default_time_limit <= 0:
        raise ValueError("default_time_limit must be positive")
    

def print_config_summary(config: Config) -> None:
    """Print human-readable configuration summary."""
    opt = config.optimization
    
    print(f"\nConfiguration Summary:")
    print(f"  Config file: {config._config_path}")
    print(f"  Items to assign ({len(opt.items_to_assign)}): {opt.items_to_assign}")
    print(f"  Available positions ({len(opt.positions_to_assign)}): {opt.positions_to_assign}")
    
    if opt.items_assigned:
        print(f"  Pre-assigned items ({len(opt.items_assigned)}): {opt.items_assigned}")
        print(f"  Pre-assigned positions ({len(opt.positions_assigned)}): {opt.positions_assigned}")
    
    if opt.items_to_constrain:
        print(f"  Constrained items ({len(opt.items_to_constrain)}): {opt.items_to_constrain}")
        print(f"  Constraint positions ({len(opt.positions_to_constrain)}): {opt.positions_to_constrain}")
    
    print(f"  MOO defaults: max_solutions={config.moo.default_max_solutions}, time_limit={config.moo.default_time_limit}s")
    print(f"  Visualization: keyboard={config.visualization.print_keyboard}, verbose={config.visualization.verbose_output}")


def create_default_config(output_path: str = "config.yaml") -> None:
    """
    Create a default configuration file with common settings.
    
    Args:
        output_path: Path where to save the default config
    """
    default_config = {
        'paths': {
            'item_pair_score_table': 'input/frequency/normalized-english-letter-pair-counts-google-ngrams.csv',
            'position_pair_score_table': 'input/keypair_engram7_scores.csv',
            'layout_results_folder': 'output/layouts'
        },
        'optimization': {
            'items_to_assign': 'etaoinsrhl',
            'positions_to_assign': 'FDESVRJKIL',
            'items_assigned': '',   
            'positions_assigned': '',  
            'items_to_constrain': '',   
            'positions_to_constrain': ''  
        },
        'moo': {
            'default_objectives': [
                'engram7_load',
                'engram7_strength',
                'engram7_position',
                'engram7_vspan',
                'engram7_hspan',
                'engram7_sequence'
            ],
            'default_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            'default_maximize': [True, True, True, True, True, True],
            'default_max_solutions': 100,
            'default_time_limit': 3600,
            'default_position_pair_score_table': 'input/keypair_engram7_scores.csv',
            'default_item_pair_score_table': 'input/frequency/normalized-english-letter-pair-counts-google-ngrams.csv',
            'show_progress_bar': True,
            'save_detailed_results': True
        },
        'visualization': {
            'print_keyboard': False,
            'verbose_output': False
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")


def validate_files_exist(config: Config, position_pair_score_table: Optional[str] = None, 
                        item_pair_score_table: Optional[str] = None) -> None:
    """
    Validate that external files required for MOO exist.
    
    Args:
        config: Configuration object
        position_pair_score_table: Path to keypair scores table
        item_pair_score_table: Path to frequency file
    """
    # Check keypair table
    if position_pair_score_table:
        if not os.path.exists(position_pair_score_table):
            raise FileNotFoundError(f"Keypair table not found: {position_pair_score_table}")
    elif os.path.exists(config.moo.default_position_pair_score_table):
        print(f"Using default keypair table: {config.moo.default_position_pair_score_table}")
    else:
        raise FileNotFoundError(f"No keypair table found at default location: {config.moo.default_position_pair_score_table}")
    
    # Check frequency file (optional)
    if item_pair_score_table:
        if not os.path.exists(item_pair_score_table):
            print(f"Warning: Frequency file not found: {item_pair_score_table}")
    elif os.path.exists(config.moo.default_item_pair_score_table):
        print(f"Using default frequency file: {config.moo.default_item_pair_score_table}")
    else:
        print(f"Warning: No frequency file found at default location: {config.moo.default_item_pair_score_table}")
        print("Will use unweighted scoring")


if __name__ == "__main__":
    # Example usage and testing
    print("Configuration Management for MOO Layout Optimization")
    
    # Test configuration creation
    try:
        if not os.path.exists("config.yaml"):
            print("Creating default configuration...")
            create_default_config()
        
        print("Loading configuration...")
        config = load_config()
        
        print_config_summary(config)
        
        print(f"\nValidating external files...")
        validate_files_exist(config)
        
        print(f"\nConfiguration validation successful!")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        print(f"\nTo create a default configuration, run:")
        print(f"python config.py")