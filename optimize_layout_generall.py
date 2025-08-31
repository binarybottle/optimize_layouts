# optimize_layout_general.py
"""
Complete General Multi-Objective Layout Optimizer with Pre-computed Matrices

Optimized for extreme-memory cluster nodes. Pre-computes all objective matrices
at startup for maximum speed during search.

Usage:
    # Current system compatibility (2 objectives: item + item-pair)
    python optimize_layout_general.py --config config.yaml --mode current --keypair-table data/keypair_scores.csv
    
    # General MOO (arbitrary objectives from key-pair table)
    python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized,engram8_score_normalized
    
    # With custom weights and directions
    python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --weights 1.0,2.0 --maximize true,false --max-solutions 50 --time-limit 3600
"""

import argparse
import time
import numpy as np
import pandas as pd
import datetime
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from numba import jit

# Import existing modules
from config import Config, load_config
from search import multi_objective_search, pareto_dominates, build_pareto_front
from display import print_optimization_header, print_search_space_info
from validation import run_validation_suite

#-----------------------------------------------------------------------------
# Configuration for General MOO
#-----------------------------------------------------------------------------
@dataclass
class GeneralMOOConfig:
    """Configuration for general multi-objective optimization."""
    keypair_table_path: str
    objective_columns: List[str]
    objective_weights: List[float]
    maximize_objectives: List[bool]
    
    # Layout specification (extracted from config.yaml)
    items_to_assign: str
    positions_to_assign: str
    items_assigned: str = ""
    positions_assigned: str = ""
    items_to_constrain: str = ""
    positions_to_constrain: str = ""
    
    def __post_init__(self):
        """Validate configuration consistency."""
        n_obj = len(self.objective_columns)
        if len(self.objective_weights) != n_obj:
            raise ValueError(f"Weights length ({len(self.objective_weights)}) != objectives length ({n_obj})")
        if len(self.maximize_objectives) != n_obj:
            raise ValueError(f"Maximize flags length ({len(self.maximize_objectives)}) != objectives length ({n_obj})")

#-----------------------------------------------------------------------------
# Pre-computed Matrix System (Optimized for Extreme Memory)
#-----------------------------------------------------------------------------
class ExtremeMemoryObjectiveArrays:
    """
    Pre-computed objective matrices optimized for extreme-memory cluster nodes.
    
    This system pre-computes ALL objective matrices at startup and keeps them
    in memory for ultra-fast scoring during search. Designed to take advantage
    of nodes with hundreds of GB of RAM.
    """
    
    def __init__(self, keypair_df: pd.DataFrame, config: GeneralMOOConfig):
        """
        Pre-compute all objective matrices using maximum memory for maximum speed.
        
        Args:
            keypair_df: DataFrame with key_pair column and all objective columns
            config: General MOO configuration
        """
        self.config = config
        self.objective_columns = config.objective_columns
        self.n_objectives = len(config.objective_columns)
        
        # Setup position mapping
        positions_list = list(config.positions_to_assign.upper())
        self.n_positions = len(positions_list)
        self.position_to_idx = {pos: i for i, pos in enumerate(positions_list)}
        self.positions_list = positions_list
        
        print(f"Pre-computing {self.n_objectives} objective matrices for extreme-memory optimization")
        print(f"Position space: {self.n_positions} positions")
        print(f"Memory strategy: Pre-compute everything, optimize for speed")
        
        # Pre-compute all matrices
        self.objective_matrices = {}
        self.missing_pair_stats = {}
        total_memory_mb = 0
        
        start_time = time.time()
        
        for i, obj_col in enumerate(self.objective_columns):
            print(f"  [{i+1}/{self.n_objectives}] Computing {obj_col}...")
            
            matrix, missing_pairs = self._create_objective_matrix(keypair_df, obj_col)
            
            # Apply weight and direction
            if config.objective_weights[i] != 1.0:
                matrix *= config.objective_weights[i]
            
            if not config.maximize_objectives[i]:
                matrix *= -1.0  # Negate for minimization
            
            self.objective_matrices[obj_col] = matrix
            self.missing_pair_stats[obj_col] = missing_pairs
            
            # Memory usage
            matrix_memory_mb = matrix.nbytes / (1024 * 1024)
            total_memory_mb += matrix_memory_mb
            
            print(f"      Matrix: {matrix.shape}, {matrix_memory_mb:.1f}MB")
            print(f"      Range: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")
            if missing_pairs > 0:
                total_pairs = self.n_positions * (self.n_positions - 1)
                print(f"      Missing: {missing_pairs}/{total_pairs} pairs ({missing_pairs/total_pairs*100:.1f}%)")
        
        elapsed = time.time() - start_time
        print(f"Pre-computation complete: {total_memory_mb:.1f}MB total, {elapsed:.2f}s")
        
        # Pre-compile JIT scoring function for maximum speed
        print("Pre-compiling JIT scoring functions...")
        self._warmup_jit_functions()
        print("JIT compilation complete")
        
    def _create_objective_matrix(self, keypair_df: pd.DataFrame, obj_col: str) -> Tuple[np.ndarray, int]:
        """Create a single objective matrix from the key-pair table."""
        
        # Create position-pair lookup
        pair_lookup = {}
        for _, row in keypair_df.iterrows():
            key_pair = str(row['key_pair'])
            if len(key_pair) == 2 and not pd.isna(row[obj_col]):
                pos1, pos2 = key_pair[0].upper(), key_pair[1].upper()
                if pos1 in self.position_to_idx and pos2 in self.position_to_idx:
                    pair_lookup[(pos1, pos2)] = float(row[obj_col])
        
        # Create matrix
        matrix = np.zeros((self.n_positions, self.n_positions), dtype=np.float32)
        missing_pairs = 0
        
        for i, pos1 in enumerate(self.positions_list):
            for j, pos2 in enumerate(self.positions_list):
                if i != j:  # Skip diagonal (no self-pairs)
                    key = (pos1, pos2)
                    if key in pair_lookup:
                        matrix[i, j] = pair_lookup[key]
                    else:
                        matrix[i, j] = 0.0  # Default for missing pairs
                        missing_pairs += 1
        
        return matrix, missing_pairs
    
    def _warmup_jit_functions(self):
        """Pre-compile JIT functions with dummy data."""
        dummy_mapping = np.array([0, 1, 2], dtype=np.int32)
        dummy_matrix = np.random.random((self.n_positions, self.n_positions)).astype(np.float32)
        
        # Call JIT function to trigger compilation
        _calculate_objective_score_jit(dummy_mapping, dummy_matrix)
    
    def calculate_all_objectives(self, mapping_array: np.ndarray) -> List[float]:
        """
        Calculate all objective scores using pre-computed matrices.
        
        Args:
            mapping_array: Array where mapping_array[i] = position_index for item i
            
        Returns:
            List of objective scores (one per objective)
        """
        objectives = []
        
        for obj_col in self.objective_columns:
            matrix = self.objective_matrices[obj_col]
            score = _calculate_objective_score_jit(mapping_array, matrix)
            objectives.append(score)
        
        return objectives

@jit(nopython=True, fastmath=True)
def _calculate_objective_score_jit(mapping_array: np.ndarray, objective_matrix: np.ndarray) -> float:
    """
    JIT-compiled objective score calculation for maximum speed.
    
    Args:
        mapping_array: Array mapping items to positions
        objective_matrix: Pre-computed position-pair matrix for this objective
        
    Returns:
        Normalized objective score
    """
    total_score = 0.0
    pair_count = 0
    n_items = len(mapping_array)
    
    for i in range(n_items):
        pos_i = mapping_array[i]
        if pos_i >= 0 and pos_i < objective_matrix.shape[0]:  # Valid position
            for j in range(n_items):
                pos_j = mapping_array[j]
                if i != j and pos_j >= 0 and pos_j < objective_matrix.shape[1]:  # Valid pair
                    total_score += objective_matrix[pos_i, pos_j]
                    pair_count += 1
    
    # Normalize by number of pairs
    return total_score / max(1, pair_count)

#-----------------------------------------------------------------------------
# General MOO Layout Scorer (Compatible with Existing Search)
#-----------------------------------------------------------------------------
class GeneralMOOLayoutScorer:
    """
    Layout scorer using pre-computed objective matrices.
    
    Fully compatible with existing multi_objective_search() function
    and all other infrastructure.
    """
    
    def __init__(self, precomputed_arrays: ExtremeMemoryObjectiveArrays):
        self.arrays = precomputed_arrays
        self.mode = 'multi_objective'
        self.mode_name = f"General MOO ({precomputed_arrays.n_objectives} objectives)"
        
        # Create dummy attributes for compatibility with existing search code
        self.n_items = len(precomputed_arrays.config.items_to_assign)
        self.n_positions = precomputed_arrays.n_positions
    
    def score_layout(self, mapping_array: np.ndarray, return_components: bool = False):
        """
        Score layout using pre-computed matrices.
        
        Compatible with existing search algorithms.
        """
        objectives = self.arrays.calculate_all_objectives(mapping_array)
        
        if return_components:
            # For compatibility with existing code that expects breakdown
            combined_score = np.mean(objectives)  # Simple average for display
            return objectives + [combined_score]
        else:
            return objectives
    
    def get_components(self, mapping_array: np.ndarray):
        """Get components for compatibility with existing code."""
        objectives = self.arrays.calculate_all_objectives(mapping_array)
        
        # Create compatibility object
        class GeneralComponents:
            def __init__(self, objectives):
                self.objectives = objectives
                # For legacy compatibility
                self.item_score = objectives[0] if len(objectives) > 0 else 0.0
                self.item_pair_score = objectives[1] if len(objectives) > 1 else objectives[0] if len(objectives) > 0 else 0.0
            
            def as_list(self):
                return self.objectives
            
            def total(self):
                # Return simple average as "total" for display purposes
                return np.mean(self.objectives)
        
        return GeneralComponents(objectives)
    
    def clear_cache(self):
        """No caches to clear in this implementation."""
        pass

#-----------------------------------------------------------------------------
# Current System Compatibility via General Infrastructure
#-----------------------------------------------------------------------------
class CurrentSystemViaGeneralScorer:
    """
    Run current system (item + item-pair scores) through general infrastructure.
    
    This allows validation that the general system produces identical results
    to the current system when configured with equivalent objectives.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mode = 'multi_objective'
        self.mode_name = "Current System via General (validation mode)"
        
        # Load existing normalized scores
        from scoring import load_normalized_scores
        self.normalized_scores = load_normalized_scores(config)
        
        # Create traditional arrays for comparison
        from scoring import prepare_scoring_arrays
        self.traditional_arrays = prepare_scoring_arrays(
            items_to_assign=list(config.optimization.items_to_assign),
            positions_to_assign=list(config.optimization.positions_to_assign),
            norm_item_scores=self.normalized_scores[0],
            norm_item_pair_scores=self.normalized_scores[1],
            norm_position_scores=self.normalized_scores[2],
            norm_position_pair_scores=self.normalized_scores[3]
        )
        
        # Create traditional scorer for delegation
        from scoring import LayoutScorer
        self.traditional_scorer = LayoutScorer(self.traditional_arrays, mode='multi_objective')
        
        print("Current system compatibility scorer initialized")
        print(f"Using traditional scoring: {self.traditional_scorer.mode_name}")
    
    def score_layout(self, mapping_array: np.ndarray, return_components: bool = False):
        """Delegate to traditional scorer for exact compatibility."""
        return self.traditional_scorer.score_layout(mapping_array, return_components)
    
    def get_components(self, mapping_array: np.ndarray):
        """Delegate to traditional scorer."""
        return self.traditional_scorer.get_components(mapping_array)
    
    def clear_cache(self):
        """Delegate to traditional scorer."""
        self.traditional_scorer.clear_cache()

#-----------------------------------------------------------------------------
# Configuration and Data Loading
#-----------------------------------------------------------------------------
def load_and_validate_keypair_table(filepath: str) -> pd.DataFrame:
    """Load and validate the comprehensive key-pair scoring table."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Key-pair table not found: {filepath}")
    
    print(f"Loading comprehensive key-pair table: {filepath}")
    
    # Load with robust parsing
    try:
        df = pd.read_csv(filepath, dtype={'key_pair': str})
    except Exception as e:
        raise ValueError(f"Failed to load key-pair table: {e}")
    
    print(f"  Table size: {len(df)} rows × {len(df.columns)} columns")
    
    # Validate required structure
    if 'key_pair' not in df.columns:
        raise ValueError("Key-pair table must have 'key_pair' column")
    
    # Count valid key-pairs
    valid_pairs = 0
    unique_positions = set()
    
    for key_pair in df['key_pair']:
        key_str = str(key_pair)
        if len(key_str) == 2:
            valid_pairs += 1
            unique_positions.add(key_str[0].upper())
            unique_positions.add(key_str[1].upper())
    
    print(f"  Valid key-pairs: {valid_pairs}")
    print(f"  Unique positions: {len(unique_positions)} ({sorted(unique_positions)})")
    
    # Show available objective columns
    objective_columns = [col for col in df.columns if col != 'key_pair']
    print(f"  Available objectives: {len(objective_columns)}")
    if len(objective_columns) <= 10:
        print(f"    {objective_columns}")
    else:
        print(f"    {objective_columns[:5]} ... {objective_columns[-3:]}")
    
    return df

def extract_general_config_from_yaml(config: Config, keypair_table_path: str,
                                   objectives: List[str], weights: List[float],
                                   maximize: List[bool]) -> GeneralMOOConfig:
    """Extract general MOO configuration from existing config.yaml structure."""
    
    return GeneralMOOConfig(
        keypair_table_path=keypair_table_path,
        objective_columns=objectives,
        objective_weights=weights,
        maximize_objectives=maximize,
        items_to_assign=config.optimization.items_to_assign,
        positions_to_assign=config.optimization.positions_to_assign,
        items_assigned=config.optimization.items_assigned or "",
        positions_assigned=config.optimization.positions_assigned or "",
        items_to_constrain=config.optimization.items_to_constrain or "",
        positions_to_constrain=config.optimization.positions_to_constrain or ""
    )

def parse_objectives_config(objectives_str: str, weights_str: Optional[str] = None,
                           maximize_str: Optional[str] = None) -> Tuple[List[str], List[float], List[bool]]:
    """Parse command-line objective configuration."""
    objectives = [obj.strip() for obj in objectives_str.split(',') if obj.strip()]
    
    # Parse weights
    if weights_str:
        try:
            weights = [float(w.strip()) for w in weights_str.split(',') if w.strip()]
            if len(weights) != len(objectives):
                raise ValueError(f"Number of weights ({len(weights)}) must match objectives ({len(objectives)})")
        except ValueError as e:
            raise ValueError(f"Invalid weights format: {e}")
    else:
        weights = [1.0] * len(objectives)
    
    # Parse maximize/minimize directions
    if maximize_str:
        maximize_flags = []
        for flag in maximize_str.split(','):
            flag = flag.strip().lower()
            if flag in ['true', '1', 'yes', 'max', 'maximize']:
                maximize_flags.append(True)
            elif flag in ['false', '0', 'no', 'min', 'minimize']:
                maximize_flags.append(False)
            else:
                raise ValueError(f"Invalid maximize flag: '{flag}'. Use true/false, 1/0, or max/min")
        
        if len(maximize_flags) != len(objectives):
            raise ValueError(f"Number of maximize flags ({len(maximize_flags)}) must match objectives ({len(objectives)})")
    else:
        maximize_flags = [True] * len(objectives)  # Default: maximize all
    
    return objectives, weights, maximize_flags

def validate_objectives_in_table(objectives: List[str], keypair_df: pd.DataFrame):
    """Validate that all requested objectives exist in the key-pair table."""
    available_columns = set(keypair_df.columns)
    missing_objectives = [obj for obj in objectives if obj not in available_columns]
    
    if missing_objectives:
        print(f"Error: The following objectives were not found in the key-pair table:")
        for obj in missing_objectives:
            print(f"  - {obj}")
        print(f"\nAvailable objective columns:")
        available_objectives = [col for col in available_columns if col != 'key_pair']
        for i, obj in enumerate(available_objectives):
            print(f"  {i+1:2d}. {obj}")
        raise ValueError(f"Missing objectives: {missing_objectives}")

def validate_positions_coverage(config: GeneralMOOConfig, keypair_df: pd.DataFrame):
    """Validate that all required positions are covered in the key-pair table."""
    required_positions = set(config.positions_to_assign.upper())
    
    # Extract positions from key-pairs in table
    available_positions = set()
    for key_pair in keypair_df['key_pair']:
        key_str = str(key_pair)
        if len(key_str) == 2:
            available_positions.add(key_str[0].upper())
            available_positions.add(key_str[1].upper())
    
    missing_positions = required_positions - available_positions
    
    if missing_positions:
        print(f"Warning: Positions in layout not found in key-pair table: {sorted(missing_positions)}")
        print(f"Available positions: {sorted(available_positions)}")
        print("Missing position-pairs will use default score of 0.0")

#-----------------------------------------------------------------------------
# Results Handling
#-----------------------------------------------------------------------------
def save_general_moo_results(pareto_front: List[Dict], config: Config, 
                            general_config: GeneralMOOConfig) -> str:
    """Save general MOO results with comprehensive information."""
    
    # Prepare results data
    results_data = []
    
    for i, solution in enumerate(pareto_front):
        mapping = solution['mapping']
        objectives = solution['objectives']
        
        # Create layout strings
        items_str = ''.join(mapping.keys())
        positions_str = ''.join(mapping.values())
        
        # Base row data
        row = {
            'rank': i + 1,
            'items': items_str,
            'positions': positions_str,
            'layout': f"{items_str} → {positions_str}",
            'n_objectives': len(objectives)
        }
        
        # Add individual objective scores with descriptive names
        for j, obj_col in enumerate(general_config.objective_columns):
            row[f'obj_{j+1}_{obj_col}'] = objectives[j] if j < len(objectives) else 0.0
        
        # Add summary statistics
        row['obj_mean'] = np.mean(objectives)
        row['obj_std'] = np.std(objectives)
        row['obj_min'] = np.min(objectives)
        row['obj_max'] = np.max(objectives)
        
        results_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Generate filename with more detailed naming
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config._config_path).stem if hasattr(config, '_config_path') else 'unknown'
    n_obj = len(general_config.objective_columns)
    filename = f"general_moo_{n_obj}obj_results_{config_name}_{timestamp}.csv"
    
    # Ensure output directory exists
    output_dir = Path(config.paths.output.layout_results_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    # Save with metadata header
    with open(filepath, 'w') as f:
        f.write(f"# General MOO Results\n")
        f.write(f"# Generated: {datetime.datetime.now().isoformat()}\n")
        f.write(f"# Config: {config._config_path if hasattr(config, '_config_path') else 'unknown'}\n")
        f.write(f"# Objectives: {len(general_config.objective_columns)}\n")
        f.write(f"# Pareto solutions: {len(pareto_front)}\n")
        for i, obj in enumerate(general_config.objective_columns):
            weight = general_config.objective_weights[i]
            direction = "max" if general_config.maximize_objectives[i] else "min"
            f.write(f"# Objective {i+1}: {obj} (weight={weight}, {direction})\n")
        f.write(f"#\n")
    
    # Append DataFrame (skip header since we wrote custom header)
    df.to_csv(filepath, mode='a', index=False)
    
    return str(filepath)

def print_general_moo_results(pareto_front: List[Dict], general_config: GeneralMOOConfig):
    """Print formatted general MOO results."""
    
    print(f"\n{'='*60}")
    print(f"GENERAL MOO RESULTS")
    print(f"{'='*60}")
    
    print(f"Pareto Front Size: {len(pareto_front)}")
    print(f"Objectives ({len(general_config.objective_columns)}):")
    
    for i, obj_col in enumerate(general_config.objective_columns):
        weight = general_config.objective_weights[i]
        direction = "maximize" if general_config.maximize_objectives[i] else "minimize"
        print(f"  {i+1}. {obj_col} (weight: {weight}, {direction})")
    
    if pareto_front:
        # Calculate objective statistics
        objectives_matrix = np.array([sol['objectives'] for sol in pareto_front])
        
        print(f"\nObjective Statistics:")
        for i, obj_col in enumerate(general_config.objective_columns):
            values = objectives_matrix[:, i]
            print(f"  {obj_col}:")
            print(f"    Range: [{np.min(values):.6f}, {np.max(values):.6f}]")
            print(f"    Mean: {np.mean(values):.6f}, Std: {np.std(values):.6f}")
        
        # Show correlations between objectives (if more than 1)
        if len(general_config.objective_columns) > 1:
            print(f"\nObjective Correlations:")
            corr_matrix = np.corrcoef(objectives_matrix.T)
            n_obj = len(general_config.objective_columns)
            
            for i in range(n_obj):
                for j in range(i+1, n_obj):
                    corr = corr_matrix[i, j]
                    obj1 = general_config.objective_columns[i]
                    obj2 = general_config.objective_columns[j]
                    print(f"    {obj1} ↔ {obj2}: {corr:.3f}")
        
        # Show top solutions
        print(f"\nTop Pareto Solutions:")
        for i, sol in enumerate(pareto_front[:5]):
            mapping = sol['mapping']
            objectives = sol['objectives']
            
            layout_str = ''.join(mapping.keys()) + " → " + ''.join(mapping.values())
            obj_str = ", ".join([f"{obj:.6f}" for obj in objectives])
            
            print(f"  {i+1}. {layout_str}")
            print(f"     Objectives: [{obj_str}]")
            
            # Show objective breakdown
            for j, obj_col in enumerate(general_config.objective_columns):
                if j < len(objectives):
                    print(f"     {obj_col}: {objectives[j]:.6f}")

#-----------------------------------------------------------------------------
# Main Optimization Functions
#-----------------------------------------------------------------------------
def run_current_system_mode(config: Config, keypair_table_path: str, **kwargs) -> None:
    """
    Run current system for validation/comparison.
    
    Uses the traditional 2-objective approach through general infrastructure
    to validate consistency.
    """
    
    print_optimization_header("Current System Compatibility Mode", config)
    print(f"Key-pair table: {keypair_table_path}")
    print(f"Mode: Current system (item + item-pair scores)")
    
    # Create current system scorer
    scorer = CurrentSystemViaGeneralScorer(config)
    
    # Show search space
    print_search_space_info(config)
    
    # Run multi-objective search using existing infrastructure
    print("Running MOO search (current system mode)...")
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer,
        max_solutions=kwargs.get('max_solutions'),
        time_limit=kwargs.get('time_limit'),
        processes=kwargs.get('processes')
    )
    
    # Display and save results
    if pareto_front:
        # Use existing display functions
        from display import print_moo_results, save_moo_results_to_csv
        objective_names = ['Item Score', 'Item-Pair Score']
        
        print_moo_results(pareto_front, config, scorer.normalized_scores, objective_names)
        
        # Save using existing function
        csv_path = save_moo_results_to_csv(pareto_front, config, scorer.normalized_scores, objective_names)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("No Pareto solutions found!")
    
    # Summary
    print(f"\nCurrent System Summary:")
    print(f"  Pareto solutions: {len(pareto_front)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Objectives: {objective_names}")

def run_general_moo_mode(config: Config, general_config: GeneralMOOConfig, **kwargs) -> None:
    """
    Run general multi-objective optimization with arbitrary objectives.
    """
    
    print_optimization_header("General MOO", config)
    print(f"Key-pair table: {general_config.keypair_table_path}")
    print(f"Objectives: {len(general_config.objective_columns)}")
    
    # Load and validate key-pair table
    keypair_df = load_and_validate_keypair_table(general_config.keypair_table_path)
    
    # Validate objectives exist in table
    validate_objectives_in_table(general_config.objective_columns, keypair_df)
    
    # Validate position coverage
    validate_positions_coverage(general_config, keypair_df)
    
    # Show objective configuration
    print(f"\nObjective Configuration:")
    for i, obj_col in enumerate(general_config.objective_columns):
        weight = general_config.objective_weights[i]
        direction = "maximize" if general_config.maximize_objectives[i] else "minimize"
        print(f"  {i+1}. {obj_col}")
        print(f"      Weight: {weight}")
        print(f"      Direction: {direction}")
    
    # Show search space (reuse existing function)
    print_search_space_info(config)
    
    # Create pre-computed arrays (this is where the magic happens)
    print(f"\nCreating extreme-memory pre-computed arrays...")
    start_precompute = time.time()
    
    precomputed_arrays = ExtremeMemoryObjectiveArrays(keypair_df, general_config)
    
    precompute_time = time.time() - start_precompute
    print(f"Pre-computation completed in {precompute_time:.2f}s")
    
    # Create general MOO scorer
    scorer = GeneralMOOLayoutScorer(precomputed_arrays)
    
    # Run multi-objective search
    print(f"\nRunning general MOO search...")
    print(f"  Objectives: {len(general_config.objective_columns)}")
    if kwargs.get('max_solutions'):
        print(f"  Max solutions: {kwargs['max_solutions']}")
    if kwargs.get('time_limit'):
        print(f"  Time limit: {kwargs['time_limit']}s")
    if kwargs.get('processes'):
        print(f"  Processes: {kwargs['processes']}")
    
    search_start = time.time()
    
    pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
        config, scorer,
        max_solutions=kwargs.get('max_solutions'),
        time_limit=kwargs.get('time_limit'),
        processes=kwargs.get('processes')
    )
    
    search_time = time.time() - search_start
    
    # Display and save results
    if pareto_front:
        print_general_moo_results(pareto_front, general_config)
        
        # Save results
        csv_path = save_general_moo_results(pareto_front, config, general_config)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("No Pareto solutions found!")
    
    # Summary
    print(f"\nGeneral MOO Summary:")
    print(f"  Pareto solutions: {len(pareto_front)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Pre-computation time: {precompute_time:.2f}s")
    print(f"  Search time: {search_time:.2f}s")
    print(f"  Total time: {precompute_time + search_time:.2f}s")
    if nodes_processed > 0:
        print(f"  Search rate: {nodes_processed/search_time:.0f} nodes/sec")

#-----------------------------------------------------------------------------
# Analysis and Diagnostics
#-----------------------------------------------------------------------------
def analyze_objective_space(keypair_df: pd.DataFrame, objectives: List[str], 
                           config: GeneralMOOConfig) -> Dict:
    """Analyze the objective space for the given configuration."""
    
    print(f"\nAnalyzing objective space for {len(objectives)} objectives...")
    
    analysis = {
        'n_objectives': len(objectives),
        'objective_ranges': {},
        'correlation_matrix': None,
        'missing_data_summary': {}
    }
    
    # Analyze each objective
    for obj_col in objectives:
        if obj_col in keypair_df.columns:
            values = keypair_df[obj_col].dropna()
            analysis['objective_ranges'][obj_col] = {
                'min': float(values.min()),
                'max': float(values.max()),
                'mean': float(values.mean()),
                'std': float(values.std()),
                'non_null_count': len(values),
                'null_count': len(keypair_df) - len(values)
            }
    
    # Calculate correlations
    if len(objectives) > 1:
        objective_data = keypair_df[objectives].dropna()
        if len(objective_data) > 0:
            analysis['correlation_matrix'] = objective_data.corr().values
    
    return analysis

def print_analysis_results(analysis: Dict, objectives: List[str]):
    """Print objective space analysis results."""
    
    print(f"Objective Space Analysis:")
    print(f"  Objectives: {analysis['n_objectives']}")
    
    print(f"\nObjective Statistics:")
    for obj in objectives:
        if obj in analysis['objective_ranges']:
            stats = analysis['objective_ranges'][obj]
            print(f"  {obj}:")
            print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
            print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            if stats['null_count'] > 0:
                print(f"    Missing values: {stats['null_count']}")
    
    # Show correlations
    if analysis['correlation_matrix'] is not None and len(objectives) > 1:
        print(f"\nObjective Correlations:")
        corr_matrix = analysis['correlation_matrix']
        for i in range(len(objectives)):
            for j in range(i+1, len(objectives)):
                corr = corr_matrix[i, j]
                print(f"    {objectives[i]} ↔ {objectives[j]}: {corr:.3f}")

#-----------------------------------------------------------------------------
# Command Line Interface
#-----------------------------------------------------------------------------
def main():
    """Main entry point with comprehensive argument handling."""
    
    parser = argparse.ArgumentParser(
        description="General Multi-Objective Layout Optimizer with Pre-computed Matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Current system compatibility mode
  python optimize_layout_general.py --config config.yaml --mode current --keypair-table data/keypair_scores.csv
  
  # General MOO with 3 objectives (equal weights, all maximized)
  python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized,engram8_score_normalized
  
  # Custom weights and directions
  python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --weights 1.0,2.0 --maximize true,false
  
  # High-performance cluster usage
  python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized,engram8_score_normalized --max-solutions 100 --time-limit 3600 --processes 32
  
  # Analysis only (no optimization)
  python optimize_layout_general.py --config config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --analyze-only
        """
    )
    
    # Required arguments
    parser.add_argument('--config', required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--keypair-table', required=True,
                       help='Path to comprehensive key-pair scoring table CSV')
    parser.add_argument('--mode', choices=['current', 'general'], required=True,
                       help='Optimization mode')
    
    # General MOO configuration
    parser.add_argument('--objectives',
                       help='Comma-separated objective column names (required for general mode)')
    parser.add_argument('--weights',
                       help='Comma-separated weights for objectives (default: all 1.0)')
    parser.add_argument('--maximize',
                       help='Comma-separated maximize flags true/false (default: all true)')
    
    # Optimization parameters
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum Pareto solutions to find')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Time limit in seconds')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes')
    
    # Analysis and validation
    parser.add_argument('--analyze-only', action='store_true',
                       help='Analyze objective space without running optimization')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation before optimization')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Mode-specific validation
        if args.mode == 'general' and not args.objectives:
            print("Error: --objectives required for general mode")
            print("Use --help to see examples")
            return 1
        
        # Parse objectives for general mode
        if args.mode == 'general':
            objectives, weights, maximize = parse_objectives_config(
                args.objectives, args.weights, args.maximize
            )
            
            print(f"Parsed configuration:")
            print(f"  Objectives: {len(objectives)}")
            for i, obj in enumerate(objectives):
                direction = "maximize" if maximize[i] else "minimize"
                print(f"    {i+1}. {obj} (weight: {weights[i]}, {direction})")
            
            # Create general config
            general_config = extract_general_config_from_yaml(
                config, args.keypair_table, objectives, weights, maximize
            )
        else:
            general_config = None
        
        # Load and validate key-pair table
        keypair_df = load_and_validate_keypair_table(args.keypair_table)
        
        if args.dry_run:
            print("\nDRY RUN - Configuration validated successfully")
            print(f"Mode: {args.mode}")
            if args.mode == 'general':
                print(f"Objectives: {objectives}")
                print(f"Would pre-compute {len(objectives)} matrices")
                
                # Estimate memory usage
                n_positions = len(config.optimization.positions_to_assign)
                matrix_size_mb = (n_positions * n_positions * 4) / (1024 * 1024)  # 4 bytes per float32
                total_memory_mb = matrix_size_mb * len(objectives)
                print(f"Estimated memory usage: {total_memory_mb:.1f}MB")
            
            return 0
        
        # Run analysis if requested
        if args.mode == 'general' and (args.analyze_only or args.verbose):
            analysis = analyze_objective_space(keypair_df, objectives, general_config)
            print_analysis_results(analysis, objectives)
            
            if args.analyze_only:
                return 0
        
        # Run validation if requested
        if args.validate:
            print("Running validation suite...")
            validation_passed = run_validation_suite(config, quick=False, mode="moo")
            if not validation_passed:
                print("Validation failed. Please fix issues before optimization.")
                return 1
            print("Validation passed!")
        
        # Run optimization based on mode
        optimization_kwargs = {
            'max_solutions': args.max_solutions,
            'time_limit': args.time_limit,
            'processes': args.processes
        }
        
        if args.mode == 'current':
            run_current_system_mode(config, args.keypair_table, **optimization_kwargs)
        elif args.mode == 'general':
            run_general_moo_mode(config, general_config, **optimization_kwargs)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())