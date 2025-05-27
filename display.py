# display.py
"""
Display, visualization, and output formatting for layout optimization.

Consolidates all presentation logic including:
- Keyboard layout visualization
- Results formatting and printing  
- CSV output generation
- Progress reporting
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import Config
from scoring import LayoutScorer, ScoreComponents

#-----------------------------------------------------------------------------
# Keyboard visualization (optional)
#-----------------------------------------------------------------------------
def visualize_keyboard_layout(mapping: Optional[Dict[str, str]] = None, 
                            title: str = "Layout", 
                            config: Config = None,
                            items_to_display: Optional[str] = None,
                            positions_to_display: Optional[str] = None) -> None:
    """
    Print ASCII visual representation of keyboard layout.
    
    Args:
        mapping: Dictionary mapping items to positions (item -> position)
        title: Title for the layout display
        config: Configuration object
        items_to_display: Items to show in unfilled layout
        positions_to_display: Positions to show in unfilled layout
    """
    if config is None:
        raise ValueError("Configuration must be provided")
    
    # Keyboard template with clear visual structure
    KEYBOARD_TEMPLATE = """╭───────────────────────────────────────────────╮
│ Layout: {title:<34}    │
├─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┤
│ {q:^3} │ {w:^3} │ {e:^3} │ {r:^3} ║ {u:^3} │ {i:^3} │ {o:^3} │ {p:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {a:^3} │ {s:^3} │ {d:^3} │ {f:^3} ║ {j:^3} │ {k:^3} │ {l:^3} │ {sc:^3} │
├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
│ {z:^3} │ {x:^3} │ {c:^3} │ {v:^3} ║ {m:^3} │ {cm:^3} │ {dt:^3} │ {sl:^3} │
╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯"""
    
    # Position mapping for special characters
    position_mapping = {
        ';': 'sc',  # semicolon
        ',': 'cm',  # comma  
        '.': 'dt',  # dot/period
        '/': 'sl'   # forward slash
    }
    
    # Initialize layout with empty spaces
    layout_chars = {
        'title': title,
        'q': ' ', 'w': ' ', 'e': ' ', 'r': ' ',
        'u': ' ', 'i': ' ', 'o': ' ', 'p': ' ',
        'a': ' ', 's': ' ', 'd': ' ', 'f': ' ',
        'j': ' ', 'k': ' ', 'l': ' ', 'sc': ' ',
        'z': ' ', 'x': ' ', 'c': ' ', 'v': ' ',
        'm': ' ', 'cm': ' ', 'dt': ' ', 'sl': ' '
    }
    
    if mapping:
        # Fill in complete mapping
        for item, position in mapping.items():
            pos_key = position_mapping.get(position.lower(), position.lower())
            layout_chars[pos_key] = item.upper()
    else:
        # Show positions to be filled
        positions_to_mark = config.optimization.positions_to_assign.lower()
        for position in positions_to_mark:
            pos_key = position_mapping.get(position, position)
            layout_chars[pos_key] = '-'
        
        # Fill in any existing assignments  
        if items_to_display and positions_to_display:
            for item, position in zip(items_to_display, positions_to_display):
                pos_key = position_mapping.get(position.lower(), position.lower())
                layout_chars[pos_key] = item.upper()
    
    print(KEYBOARD_TEMPLATE.format(**layout_chars))

#-----------------------------------------------------------------------------
# Results display
#-----------------------------------------------------------------------------
def print_optimization_header(mode: str, config: Config) -> None:
    """Print header for optimization run."""
    print(f"\n" + "="*60)
    if mode.upper() == "SOO":
        print("SINGLE-OBJECTIVE OPTIMIZATION")
    elif mode.upper() == "MOO":
        print("MULTI-OBJECTIVE OPTIMIZATION")
    else:
        print(f"{mode.upper()} OPTIMIZATION")
    print("="*60)

def print_search_space_info(config: Config) -> None:
    """Print information about the search space size."""
    from math import factorial as fact
    
    opt = config.optimization
    n_items = len(opt.items_to_assign)
    n_positions = len(opt.positions_to_assign)
    n_constrained_items = len(opt.items_to_constrain)
    n_constrained_positions = len(opt.positions_to_constrain)
    
    print("\nSearch Space Analysis:")
    
    if n_constrained_items > 0:
        # Two-phase search
        phase1_perms = fact(n_constrained_positions) // fact(n_constrained_positions - n_constrained_items)
        remaining_items = n_items - n_constrained_items
        remaining_positions = n_positions - n_constrained_items
        phase2_perms_per_phase1 = fact(remaining_positions) // fact(remaining_positions - remaining_items)
        total_perms = phase1_perms * phase2_perms_per_phase1
        
        print(f"  Phase 1 ({n_constrained_items} constrained items): {phase1_perms:,} arrangements")
        print(f"  Phase 2 ({remaining_items} remaining items): {phase2_perms_per_phase1:,} per Phase 1")
        print(f"  Total permutations: {total_perms:,}")
    else:
        # Single-phase search
        total_perms = fact(n_positions) // fact(n_positions - n_items)
        print(f"  Single phase ({n_items} items in {n_positions} positions): {total_perms:,} permutations")

def print_soo_results(results: List[Tuple[float, Dict[str, str], Dict]], 
                     config: Config, 
                     scorer: LayoutScorer,
                     n_display: int = 5,
                     verbose: bool = False) -> None:
    """
    Print single-objective optimization results.
    
    Args:
        results: List of (score, mapping, metadata) tuples
        config: Configuration object
        scorer: Scorer used for optimization
        n_display: Number of solutions to display
        verbose: Whether to show detailed breakdown
    """
    opt = config.optimization
    n_display = min(len(results), n_display)
    
    if len(results) > 1:
        print(f"\nTop {n_display} Solutions:")
    else:
        print("\nOptimal Solution:")
    
    for i, (score, mapping, _) in enumerate(results[:n_display], 1):
        print(f"\n#{i}: Score = {score:.9f}")
        
        # Build complete item->position mapping
        complete_mapping = {}
        if opt.items_assigned:
            complete_mapping.update(dict(zip(opt.items_assigned, opt.positions_assigned.upper())))
        complete_mapping.update({k: v.upper() for k, v in mapping.items()})
        
        # Display layout strings
        all_items = ''.join(complete_mapping.keys())
        all_positions = ''.join(complete_mapping.values())
        
        if opt.items_assigned:
            opt_items = ''.join(mapping.keys())
            opt_positions = ''.join(v.upper() for v in mapping.values())
            print(f"  Layout: {opt.items_assigned} + {opt_items} = {all_items}")
            print(f"  Positions: {opt.positions_assigned.upper()} + {opt_positions} = {all_positions}")
        else:
            print(f"  Layout: {all_items} → {all_positions}")
        
        # Detailed breakdown if requested
        if verbose:
            _print_detailed_breakdown(complete_mapping, config, scorer)
        
        # Keyboard visualization
        if config.visualization.print_keyboard:
            visualize_keyboard_layout(complete_mapping, f"Solution #{i}", config)

def print_moo_results(pareto_front: List[Tuple[np.ndarray, List[float]]], 
                     config: Config,
                     objective_names: List[str] = None,
                     max_display: int = 5) -> None:
    """
    Print multi-objective optimization results.
    
    Args:
        pareto_front: List of (mapping_array, objectives) tuples
        config: Configuration object  
        objective_names: Names for the objectives
        max_display: Maximum number of solutions to display
    """
    if objective_names is None:
        objective_names = ['Item Score', 'Pair Score', 'Cross Score']
    
    opt = config.optimization
    items_list = list(opt.items_to_assign)
    positions_list = list(opt.positions_to_assign)
    
    print(f"\nPareto Front: {len(pareto_front)} non-dominated solutions")
    
    # Sort by combined score for display
    pareto_with_combined = []
    for mapping_array, objectives in pareto_front:
        combined_score = sum(objectives)
        pareto_with_combined.append((combined_score, mapping_array, objectives))
    
    pareto_with_combined.sort(key=lambda x: x[0], reverse=True)
    
    n_display = min(len(pareto_with_combined), max_display)
    print(f"Showing top {n_display} by combined score:")
    
    for i, (combined_score, mapping_array, objectives) in enumerate(pareto_with_combined[:n_display], 1):
        print(f"\n#{i}: Combined Score = {combined_score:.6f}")
        
        # Build mapping dictionary
        item_mapping = {item: positions_list[pos] for item, pos in 
                       zip(items_list, mapping_array) if pos >= 0}
        
        # Complete mapping including pre-assigned
        complete_mapping = {}
        if opt.items_assigned:
            complete_mapping.update(dict(zip(opt.items_assigned, opt.positions_assigned.upper())))
        complete_mapping.update({k: v.upper() for k, v in item_mapping.items()})
        
        # Display objectives
        print("  Objectives:")
        for obj_name, obj_value in zip(objective_names, objectives):
            print(f"    {obj_name}: {obj_value:.6f}")
        
        # Display layout
        all_items = ''.join(complete_mapping.keys())  
        all_positions = ''.join(complete_mapping.values())
        print(f"  Layout: {all_items} → {all_positions}")
        
        # Keyboard visualization
        if config.visualization.print_keyboard:
            visualize_keyboard_layout(complete_mapping, f"Pareto Solution #{i}", config)

def _print_detailed_breakdown(complete_mapping: Dict[str, str], config: Config, scorer: LayoutScorer) -> None:
    """
    Print detailed score breakdown for verbose mode.
    """
    try:
        # Create position lookup dictionary
        opt = config.optimization
        all_positions = list(opt.positions_to_assign) + list(opt.positions_assigned)
        position_to_index = {pos.upper(): idx for idx, pos in enumerate(all_positions)}
        
        # Convert complete_mapping to index-based mapping
        all_items = list(complete_mapping.keys())
        mapping_indices = []
        
        for item in all_items:
            pos_str = complete_mapping[item].upper()
            if pos_str in position_to_index:
                mapping_indices.append(position_to_index[pos_str])
            else:
                print(f"  Warning: Position '{pos_str}' not found in position lookup")
                mapping_indices.append(-1)  # Unassigned
        
        mapping_array = np.array(mapping_indices, dtype=np.int32)
        
        # Now we can safely call get_components
        components = scorer.get_components(mapping_array)
        
        print(f"  Detailed Breakdown:")
        print(f"    Item component: {components.item_score:.6f}")
        print(f"    Pair component: {components.pair_score:.6f}")
        print(f"    Cross component: {components.cross_score:.6f}")
        print(f"    Scoring mode: {scorer.mode_name}")
        
    except Exception as e:
        print(f"  Detailed breakdown unavailable: {e}")
        print(f"  Note: This is a known issue with verbose mode display")

#-----------------------------------------------------------------------------
# CSV output
#-----------------------------------------------------------------------------
def save_soo_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict]], 
                           config: Config) -> str:
    """
    Save single-objective results to CSV file.
    
    Args:
        results: Optimization results
        config: Configuration object
        
    Returns:
        Path to saved CSV file
    """
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.basename(config._config_path).replace('.yaml', '')
    filename = f"soo_results_{config_name}_{timestamp}.csv"
    output_path = os.path.join(config.paths.layout_results_folder, filename)
    
    opt = config.optimization
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Header with configuration
        writer.writerow(['Single-Objective Optimization Results'])
        writer.writerow(['Items to assign', opt.items_to_assign])
        writer.writerow(['Available positions', opt.positions_to_assign])
        writer.writerow(['Items assigned', opt.items_assigned])
        writer.writerow(['Positions assigned', opt.positions_assigned])
        writer.writerow([])
        
        # Results header
        writer.writerow([
            'Rank', 'Complete Items', 'Complete Positions', 
            'Optimized Items', 'Optimized Positions', 'Score'
        ])
        
        # Results data
        for rank, (score, mapping, _) in enumerate(results, 1):
            # Build complete mappings
            complete_mapping = {}
            if opt.items_assigned:
                complete_mapping.update(dict(zip(opt.items_assigned, opt.positions_assigned.upper())))
            complete_mapping.update({k: v.upper() for k, v in mapping.items()})
            
            complete_items = ''.join(complete_mapping.keys())
            complete_positions = ''.join(complete_mapping.values())
            opt_items = ''.join(mapping.keys())
            opt_positions = ''.join(v.upper() for v in mapping.values())
            
            writer.writerow([
                rank, complete_items, complete_positions,
                opt_items, opt_positions, f"{score:.9f}"
            ])
    
    return output_path

def save_moo_results_to_csv(pareto_front: List[Tuple[np.ndarray, List[float]]], 
                           config: Config,
                           objective_names: List[str] = None) -> str:
    """
    Save multi-objective results to CSV file.
    
    Args:
        pareto_front: Pareto front solutions
        config: Configuration object
        objective_names: Names for objectives
        
    Returns:
        Path to saved CSV file
    """
    if objective_names is None:
        objective_names = ['Item Score', 'Pair Score', 'Cross Score']
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.basename(config._config_path).replace('.yaml', '')
    filename = f"moo_results_{config_name}_{timestamp}.csv"
    output_path = os.path.join(config.paths.layout_results_folder, filename)
    
    opt = config.optimization
    items_list = list(opt.items_to_assign)
    positions_list = list(opt.positions_to_assign)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Header
        writer.writerow(['Multi-Objective Optimization Results'])
        writer.writerow(['Items to assign', opt.items_to_assign])
        writer.writerow(['Available positions', opt.positions_to_assign])
        writer.writerow([])
        
        # Results header
        header = ['Rank', 'Items', 'Positions'] + objective_names + ['Combined Total']
        writer.writerow(header)
        
        # Sort by combined score for ranking
        pareto_with_combined = [(sum(objectives), mapping_array, objectives) 
                               for mapping_array, objectives in pareto_front]
        pareto_with_combined.sort(key=lambda x: x[0], reverse=True)
        
        # Write results
        for rank, (combined_score, mapping_array, objectives) in enumerate(pareto_with_combined, 1):
            # Build complete mapping
            item_mapping = {item: positions_list[pos] for item, pos in 
                           zip(items_list, mapping_array) if pos >= 0}
            complete_mapping = {}
            if opt.items_assigned:
                complete_mapping.update(dict(zip(opt.items_assigned, opt.positions_assigned.upper())))
            complete_mapping.update({k: v.upper() for k, v in item_mapping.items()})
            
            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            
            row = ([rank, all_items, all_positions] + 
                   [f"{obj:.9f}" for obj in objectives] +
                   [f"{combined_score:.9f}"])
            writer.writerow(row)
    
    return output_path

#-----------------------------------------------------------------------------
# Progress reporting
#-----------------------------------------------------------------------------
def print_optimization_progress(nodes_processed: int, nodes_pruned: int, 
                              solutions_found: int, elapsed_time: float,
                              search_space_size: Optional[int] = None) -> None:
    """Print optimization progress statistics."""
    print(f"\nOptimization Statistics:")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Nodes pruned: {nodes_pruned:,}")
    print(f"  Solutions found: {solutions_found:,}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")
    
    if nodes_processed > 0:
        prune_rate = 100 * nodes_pruned / (nodes_pruned + nodes_processed)
        print(f"  Pruning efficiency: {prune_rate:.1f}%")
    
    if search_space_size:
        explored_pct = (nodes_processed / search_space_size) * 100
        print(f"  Search space explored: {explored_pct:.4f}%")