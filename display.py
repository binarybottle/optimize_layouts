# display.py
"""
Display, visualization, and output formatting for layout optimization.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import Config
from scoring import LayoutScorer, calculate_complete_layout_score_direct, apply_default_combination

#-----------------------------------------------------------------------------
# Keyboard visualization
#-----------------------------------------------------------------------------
def visualize_keyboard_layout(mapping: Optional[Dict[str, str]] = None, 
                            title: str = "Layout", 
                            config: Config = None,
                            items_to_display: Optional[str] = None,
                            positions_to_display: Optional[str] = None) -> None:
    """
    Print ASCII visual representation of keyboard layout.
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
                     score_dicts: Tuple,
                     n_display: int = 5,
                     verbose: bool = False) -> None:
    """
    Print single-objective optimization results with complete layout scores.
    
    Args:
        results: List of (score, mapping, metadata) tuples (optimization scores)
        config: Configuration object
        scorer: Scorer used for optimization  
        score_dicts: Tuple of score dictionaries for complete scoring
        n_display: Number of solutions to display
        verbose: Whether to show detailed breakdown
    """
    opt = config.optimization
    n_display = min(len(results), n_display)
    
    for i, (opt_score, mapping, _) in enumerate(results[:n_display], 1):
        # Build complete item->position mapping
        complete_mapping = {}
        if opt.items_assigned:
            complete_mapping.update(dict(zip(opt.items_assigned, [pos.upper() for pos in opt.positions_assigned])))

        complete_mapping.update({k: v.upper() for k, v in mapping.items()})
        
        # Calculate TRUE complete layout score using centralized function
        try:
            complete_total, complete_item, complete_item_pair = calculate_complete_layout_score_direct(
                complete_mapping, score_dicts)
        except Exception as e:
            print(f"Warning: Could not calculate complete score: {e}")
            complete_total = opt_score
            complete_item = complete_item_pair = 0.0
        
        print(f"\n#{i}: Optimization Score = {opt_score:.9f}")
        print(f"     Complete Layout Score = {complete_total:.9f}")
                
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
            print(f"  Complete Score Breakdown:")
            print(f"    Item component:  {complete_item:.6f}")
            print(f"    Pair component:  {complete_item_pair:.6f}")
            print(f"    Total:           {complete_total:.6f}")
        
        # Keyboard visualization
        if config.visualization.print_keyboard:
            visualize_keyboard_layout(complete_mapping, f"Solution #{i}", config)

def print_moo_results(pareto_front, config, normalized_scores, objective_names, top_n=5):
    """Print multi-objective optimization results."""
    
    print(f"\nPareto Front: {len(pareto_front)} non-dominated solutions")
    
    if not pareto_front:
        return
    
    # Print header
    print(f"\nFirst {min(top_n, len(pareto_front))} Pareto solutions:")
    print("=" * 80)
    
    # Sort by first objective for display
    sorted_front = sorted(pareto_front, key=lambda x: x['objectives'][0], reverse=True)
    
    for i, solution in enumerate(sorted_front[:top_n]):
        # Extract the correct data structure
        mapping = solution['mapping']  # This is already a dict
        objectives = solution['objectives']  # This is a list of objective values
        
        print(f"\nSolution {i+1}:")
        print(f"  {objective_names[0]}: {objectives[0]:.6f}")
        print(f"  {objective_names[1]}: {objectives[1]:.6f}")
        print(f"  Combined Score: {solution.get('score', sum(objectives)):.6f}")
        
        # Print mapping
        print("  Layout mapping:")
        for item, position in mapping.items():
            print(f"    {item} -> {position}")
    
    # Print objective ranges
    print(f"\n📊 Objective Ranges:")
    obj_values = [[sol['objectives'][i] for sol in pareto_front] for i in range(len(objective_names))]
    
    for i, name in enumerate(objective_names):
        min_val = min(obj_values[i])
        max_val = max(obj_values[i])
        print(f"  {name}: [{min_val:.6f}, {max_val:.6f}]")

#-----------------------------------------------------------------------------
# CSV output
#-----------------------------------------------------------------------------
def save_soo_results_to_csv(results: List[Tuple[float, Dict[str, str], Dict]], 
                           config: Config, normalized_scores: Tuple) -> str:
    """
    Save single-objective results to CSV file with complete layout scores.
    
    Args:
        results: Optimization results
        config: Configuration object
        normalized_scores: Tuple of score dictionaries for complete scoring
        
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
            'Optimized Items', 'Optimized Positions', 
            'Optimization Score', 'Complete Layout Score',
            'Complete Item Score', 'Complete Pair Score'
        ])
        
        # Results data
        for rank, (opt_score, mapping, _) in enumerate(results, 1):
            # Build complete mappings
            complete_mapping = {}
            if opt.items_assigned:
                complete_mapping.update(dict(zip(opt.items_assigned, [pos.upper() for pos in opt.positions_assigned])))
            complete_mapping.update({k: v.upper() for k, v in mapping.items()})
            
            print(f"DEBUG: Complete mapping construction")
            print(f"  Pre-assigned items: {list(opt.items_assigned) if opt.items_assigned else 'None'}")
            print(f"  Pre-assigned positions: {list(opt.positions_assigned) if opt.positions_assigned else 'None'}")
            print(f"  Optimized mapping: {mapping}")
            print(f"  Complete mapping: {complete_mapping}")

            complete_items = ''.join(complete_mapping.keys())
            complete_positions = ''.join(complete_mapping.values())
            opt_items = ''.join(mapping.keys())
            opt_positions = ''.join(v.upper() for v in mapping.values())
            
            # Calculate complete score using centralized function
            try:
                complete_total, complete_item, complete_item_pair = calculate_complete_layout_score_direct(
                    complete_mapping, normalized_scores)
            except Exception:
                complete_total = complete_item = complete_item_pair = 0.0
            
            writer.writerow([
                rank, complete_items, complete_positions,
                opt_items, opt_positions, f"{opt_score:.9f}",
                f"{complete_total:.9f}", f"{complete_item:.6f}", f"{complete_item_pair:.6f}"
            ])
    
    return output_path

def save_moo_results_to_csv(pareto_front: List[Dict], 
                           config: Config, normalized_scores: Tuple,
                           objective_names: List[str] = None) -> str:
    """
    Save multi-objective results to CSV file with complete layout scores.
    
    Args:
        pareto_front: Pareto front solutions (list of dictionaries)
        config: Configuration object
        normalized_scores: Tuple of score dictionaries for complete scoring
        objective_names: Names for objectives
        
    Returns:
        Path to saved CSV file
    """
    if objective_names is None:
        objective_names = ['Item Score', 'Item-Pair Score']
    
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
        header = (['Rank', 'Items', 'Positions'] + 
                 [f'Opt {name}' for name in objective_names] + 
                 ['Opt Combined', 'Complete Layout Score',
                  'Complete Item', 'Complete Pair'])
        writer.writerow(header)
        
        # Sort by combined score
        pareto_with_combined = []
        for solution in pareto_front:
            # Extract data from dictionary format
            mapping = solution['mapping']  # This is already a dict
            objectives = solution['objectives']  # This is a list
            
            # Calculate combined score
            combined_score = apply_default_combination(objectives[0], objectives[1])
            pareto_with_combined.append((combined_score, mapping, objectives))

        pareto_with_combined.sort(key=lambda x: x[0], reverse=True)
        
        # Write results
        for rank, (opt_combined_score, mapping, objectives) in enumerate(pareto_with_combined, 1):
            # Build complete mapping
            complete_mapping = {}
            if opt.items_assigned:
                complete_mapping.update(dict(zip(opt.items_assigned, [pos.upper() for pos in opt.positions_assigned])))
            complete_mapping.update({k: v.upper() for k, v in mapping.items()})

            print(f"DEBUG: Complete mapping construction")
            print(f"  Pre-assigned items: {list(opt.items_assigned) if opt.items_assigned else 'None'}")
            print(f"  Pre-assigned positions: {list(opt.positions_assigned) if opt.positions_assigned else 'None'}")
            print(f"  Optimized mapping: {mapping}")
            print(f"  Complete mapping: {complete_mapping}")

            all_items = ''.join(complete_mapping.keys())
            all_positions = ''.join(complete_mapping.values())
            
            # Calculate complete score using centralized function
            try:
                complete_total, complete_item, complete_item_pair = calculate_complete_layout_score_direct(
                    complete_mapping, normalized_scores)
            except Exception:
                complete_total = complete_item = complete_item_pair = 0.0
            
            row = ([rank, all_items, all_positions] + 
                   [f"{obj:.9f}" for obj in objectives] +
                   [f"{opt_combined_score:.9f}", f"{complete_total:.9f}",
                    f"{complete_item:.6f}", f"{complete_item_pair:.6f}"])
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