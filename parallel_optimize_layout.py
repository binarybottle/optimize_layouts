# parallel_optimize_layout.py
"""
High-performance keyboard layout optimizer designed for HPC clusters.
Leverages extreme memory and parallel processing capabilities.

- Parallel chunk processing across many cores
- Preloaded data structures
- Distributed Pareto front computation
"""

import argparse
import time
import pandas as pd
import numpy as np
import multiprocessing as mp
from typing import Dict, Tuple, List, Optional
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import math

# Import your existing modules
from config import Config, load_config, print_config_summary
from scoring import LayoutScorer, prepare_scoring_arrays
from search import single_objective_search, multi_objective_search
from display import (print_optimization_header, print_search_space_info, 
                    print_soo_results, print_moo_results, visualize_keyboard_layout,
                    save_soo_results_to_csv, save_moo_results_to_csv)

class HPCDataLoader:
    """High-performance data loader optimized for HPC environments."""
    
    def __init__(self, config: Config, use_memory_mapping: bool = True):
        self.config = config
        self.use_memory_mapping = use_memory_mapping
        self.cached_data = {}
        
    def preload_all_data(self) -> Dict:
        """Preload all CSV data into memory with optional memory mapping."""
        print("🚀 Preloading all data into memory...")
        
        data = {}
        file_paths = {
            'item_scores': self.config.paths.item_scores_file,
            'item_pair_scores': self.config.paths.item_pair_scores_file,
            'position_scores': self.config.paths.position_scores_file,
            'position_pair_scores': self.config.paths.position_pair_scores_file
        }
        
        for name, path in file_paths.items():
            if self.use_memory_mapping and os.path.exists(path):
                data[name] = self._load_with_memory_mapping(path)
            else:
                data[name] = pd.read_csv(path)
            print(f"  Loaded {name}: {len(data[name])} records")
        
        return data
    
    def _load_with_memory_mapping(self, filepath: str) -> pd.DataFrame:
        """Load CSV using memory mapping for faster access."""
        # For very large files, consider using memory mapping
        # For now, we'll use regular pandas but with optimized dtypes
        df = pd.read_csv(filepath, low_memory=False)
        
        # Optimize dtypes to reduce memory usage
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except:
                    pass  # Keep as object if conversion fails
        
        return df

class ParallelMOOManager:
    """Manages parallel multi-objective optimization across multiple processes."""
    
    def __init__(self, config: Config, n_processes: Optional[int] = None):
        self.config = config
        self.n_processes = n_processes or min(mp.cpu_count(), 64)
        print(f"🔥 Initializing parallel MOO with {self.n_processes} processes")
    
    def parallel_multi_objective_search(self, scorer, normalized_scores: Tuple, 
                                      max_solutions: int = None, time_limit: float = None) -> Tuple[List, int, int]:
        """Run multi-objective search in parallel chunks with Pareto front merging."""
        from itertools import permutations
        import numpy as np
        from parallel_moo_workers import process_moo_chunk_worker, compute_pareto_front
        
        # Get all possible assignments
        items = list(self.config.optimization.items_to_assign)
        positions = list(self.config.optimization.positions_to_assign)
        
        # Estimate search space
        total_perms = math.factorial(len(items))
        print(f"  Total search space: {total_perms:,} permutations")
        
        if total_perms < 10000:
            # Small search space - use single process with original function
            print("  Small search space - using single-threaded MOO")
            from search import multi_objective_search
            return multi_objective_search(self.config, scorer, max_solutions, time_limit, None, True)
        
        # Large search space - use parallel approach
        chunk_size = max(1000, total_perms // (self.n_processes * 4))  # 4 chunks per process
        chunks = self._create_moo_chunks(items, positions, chunk_size)
        print(f"  Created {len(chunks)} chunks for parallel processing")
        
        # Prepare scorer data for workers
        scorer_data = self._prepare_scorer_data(normalized_scores, positions)
        
        # Process chunks in parallel
        all_solutions = []
        total_nodes_processed = 0
        total_nodes_pruned = 0
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_moo_chunk_worker, (chunk, scorer_data, max_solutions)): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                
                # Check time limit
                if time_limit and (time.time() - start_time) > time_limit:
                    print(f"    Time limit reached, stopping search...")
                    break
                
                try:
                    chunk_solutions, nodes_proc, nodes_pruned = future.result()
                    all_solutions.extend(chunk_solutions)
                    total_nodes_processed += nodes_proc
                    total_nodes_pruned += nodes_pruned
                    
                    if chunk_idx % 10 == 0:
                        print(f"    Completed chunk {chunk_idx}/{len(chunks)}")
                        
                    # Periodically merge and prune to manage memory
                    if len(all_solutions) > (max_solutions or 1000) * 10:
                        all_solutions = compute_pareto_front(all_solutions)
                        if max_solutions:
                            all_solutions = all_solutions[:max_solutions]
                        
                except Exception as exc:
                    print(f"    Chunk {chunk_idx} generated exception: {exc}")
        
        # Final Pareto front computation
        print(f"  Merging {len(all_solutions)} solutions into final Pareto front...")
        pareto_front = compute_pareto_front(all_solutions)
        
        if max_solutions and len(pareto_front) > max_solutions:
            # If still too many, select diverse subset
            pareto_front = self._select_diverse_solutions(pareto_front, max_solutions)
        
        return pareto_front, total_nodes_processed, total_nodes_pruned
    
    def _create_moo_chunks(self, items: List, positions: List, chunk_size: int) -> List[List]:
        """Create chunks of permutations for MOO processing."""
        from itertools import permutations, islice
        
        def chunked_permutations(iterable, chunk_size):
            iterator = permutations(iterable)
            while True:
                chunk = list(islice(iterator, chunk_size))
                if not chunk:
                    break
                yield chunk
        
        return list(chunked_permutations(items, chunk_size))
    
    def _prepare_scorer_data(self, normalized_scores: Tuple, positions: List) -> Dict:
        """Prepare data structure for worker processes."""
        return {
            'item_scores': normalized_scores[0],
            'item_pair_scores': normalized_scores[1], 
            'position_scores': normalized_scores[2],
            'position_pair_scores': normalized_scores[3],
            'positions': positions
        }
    
    def _select_diverse_solutions(self, pareto_front: List[Dict], max_solutions: int) -> List[Dict]:
        """Select a diverse subset of solutions from the Pareto front."""
        if len(pareto_front) <= max_solutions:
            return pareto_front
        
        # Simple diversity selection: select solutions spread across objective space
        pareto_front.sort(key=lambda x: x['objectives'][0])  # Sort by first objective
        
        # Select evenly spaced solutions
        indices = []
        step = len(pareto_front) / max_solutions
        for i in range(max_solutions):
            idx = int(i * step)
            indices.append(min(idx, len(pareto_front) - 1))
        
        return [pareto_front[i] for i in indices]

class ParallelSearchManager:
    """Manages parallel search across multiple processes."""
    
    def __init__(self, config: Config, n_processes: Optional[int] = None):
        self.config = config
        self.n_processes = n_processes or min(mp.cpu_count(), 64)  # Cap at 64 for efficiency
        print(f"🔥 Initializing parallel search with {self.n_processes} processes")
    
    def parallel_single_objective_search(self, scorer, n_solutions: int, 
                                      chunk_size: int = 1000) -> Tuple[List, int, int]:
        """Run single-objective search in parallel chunks."""
        from itertools import permutations
        
        # Get all possible assignments
        items = list(self.config.optimization.items_to_assign)
        positions = list(self.config.optimization.positions_to_assign)
        
        # Estimate search space
        total_perms = np.math.factorial(len(items))
        print(f"  Total search space: {total_perms:,} permutations")
        
        if total_perms < 10000:
            # Small search space - use single process
            return single_objective_search(self.config, scorer, n_solutions)
        
        # Split search space into chunks
        chunks = self._create_search_chunks(items, positions, chunk_size)
        print(f"  Created {len(chunks)} search chunks")
        
        # Process chunks in parallel
        best_solutions = []
        total_nodes_processed = 0
        total_nodes_pruned = 0
        
        with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk, scorer, n_solutions): i 
                for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    chunk_results, nodes_proc, nodes_pruned = future.result()
                    best_solutions.extend(chunk_results)
                    total_nodes_processed += nodes_proc
                    total_nodes_pruned += nodes_pruned
                    
                    if chunk_idx % 10 == 0:
                        print(f"    Completed chunk {chunk_idx}/{len(chunks)}")
                        
                except Exception as exc:
                    print(f"    Chunk {chunk_idx} generated exception: {exc}")
        
        # Merge and return top solutions
        best_solutions.sort(key=lambda x: x['score'], reverse=True)
        return best_solutions[:n_solutions], total_nodes_processed, total_nodes_pruned
    
    def _create_search_chunks(self, items: List, positions: List, 
                            chunk_size: int) -> List[List]:
        """Create chunks of the search space for parallel processing."""
        from itertools import permutations, islice
        
        def chunked_permutations(iterable, chunk_size):
            """Generate permutations in chunks."""
            iterator = permutations(iterable)
            while True:
                chunk = list(islice(iterator, chunk_size))
                if not chunk:
                    break
                yield chunk
        
        return list(chunked_permutations(items, chunk_size))
    
    def _process_chunk(self, chunk: List, scorer, n_solutions: int) -> Tuple[List, int, int]:
        """Process a chunk of permutations."""
        best_solutions = []
        nodes_processed = 0
        
        for perm in chunk:
            # Create mapping
            mapping = dict(zip(perm, self.config.optimization.positions_to_assign))
            
            # Score the layout
            score = scorer.score_layout(mapping)
            nodes_processed += 1
            
            # Keep track of best solutions
            best_solutions.append({
                'mapping': mapping,
                'score': score
            })
        
        # Sort and keep top solutions from this chunk
        best_solutions.sort(key=lambda x: x['score'], reverse=True)
        return best_solutions[:n_solutions], nodes_processed, 0

class MemoryOptimizedArrays:
    """Optimized array structures for extreme memory configurations."""
    
    def __init__(self, data_dict: Dict):
        print("🧠 Creating memory-optimized arrays...")
        self.arrays = self._create_optimized_arrays(data_dict)
        
    def _create_optimized_arrays(self, data_dict: Dict) -> Dict:
        """Create memory-efficient numpy arrays from loaded data."""
        arrays = {}
        
        # Convert DataFrames to optimized numpy arrays
        for name, df in data_dict.items():
            if 'pair' in name:
                # For pair data, create lookup dictionaries
                arrays[name] = self._create_pair_lookup(df, name)
            else:
                # For single item data, create direct arrays
                arrays[name] = self._create_item_lookup(df, name)
        
        return arrays
    
    def _create_pair_lookup(self, df: pd.DataFrame, name: str) -> Dict:
        """Create optimized lookup for pair scores."""
        if 'item_pair' in name:
            pair_col = 'item_pair'
        else:
            pair_col = 'position_pair'
        
        lookup = {}
        for _, row in df.iterrows():
            pair_str = str(row[pair_col])
            if len(pair_str) == 2:
                key = (pair_str[0].lower(), pair_str[1].lower())
                lookup[key] = np.float32(row['score'])  # Use float32 to save memory
        
        return lookup
    
    def _create_item_lookup(self, df: pd.DataFrame, name: str) -> Dict:
        """Create optimized lookup for item scores."""
        if 'item' in name:
            key_col = 'item'
        else:
            key_col = 'position'
        
        return {
            row[key_col].lower(): np.float32(row['score']) 
            for _, row in df.iterrows()
        }

def run_hpc_single_objective_optimization(config: Config, n_solutions: int = 5, 
                                        verbose: bool = False, n_processes: int = None) -> None:
    """
    High-performance single-objective optimization for HPC clusters.
    """
    print_optimization_header("HPC SOO", config)
    print_config_summary(config)
    print_search_space_info(config)
    
    # Memory info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"🖥️  Available memory: {memory_gb:.1f} GB")
    
    # Load data efficiently
    data_loader = HPCDataLoader(config, use_memory_mapping=True)
    data_dict = data_loader.preload_all_data()
    
    # Create memory-optimized arrays
    optimized_arrays = MemoryOptimizedArrays(data_dict)
    
    # Convert to the format your existing scorer expects
    print("Preparing optimization arrays...")
    normalized_scores = (
        optimized_arrays.arrays['item_scores'],
        optimized_arrays.arrays['item_pair_scores'],
        optimized_arrays.arrays['position_scores'],
        optimized_arrays.arrays['position_pair_scores']
    )
    
    arrays = prepare_scoring_arrays(
        items_to_assign=list(config.optimization.items_to_assign),
        positions_to_assign=list(config.optimization.positions_to_assign),
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3],
        items_assigned=list(config.optimization.items_assigned) if config.optimization.items_assigned else None,
        positions_assigned=list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
    )
    
    scorer = LayoutScorer(arrays, mode='combined')
    print(f"Scorer initialized: {scorer.mode_name}")
    
    # Run parallel optimization
    print(f"\n🚀 Searching for top {n_solutions} solutions using parallel processing...")
    start_time = time.time()
    
    search_manager = ParallelSearchManager(config, n_processes)
    results, nodes_processed, nodes_pruned = search_manager.parallel_single_objective_search(
        scorer, n_solutions
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results
    if results:
        print_soo_results(results, config, scorer, normalized_scores, n_solutions, verbose)
        csv_path = save_soo_results_to_csv(results, config, normalized_scores)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo solutions found!")
    
    # Performance summary
    print(f"\n🏁 HPC Optimization Summary:")
    print(f"  Solutions found: {len(results)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Processing rate: {nodes_processed/elapsed_time:.0f} nodes/sec")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Processes used: {search_manager.n_processes}")

def run_distributed_moo(config: Config, max_solutions: int = None, 
                       time_limit: float = None, n_processes: int = None) -> None:
    """
    Distributed multi-objective optimization leveraging extreme memory.
    """
    print("🌟 Distributed Multi-Objective Optimization")
    print_config_summary(config)
    print_search_space_info(config)
    
    # Memory info
    memory_gb = psutil.virtual_memory().total / (1024**3)
    print(f"🖥️  Available memory: {memory_gb:.1f} GB")
    n_processes = n_processes or min(mp.cpu_count(), 64)  # Cap at 64 for efficiency
    print(f"🔥 Using {n_processes} processes for parallel MOO")
    
    # Load data efficiently
    data_loader = HPCDataLoader(config, use_memory_mapping=True)
    data_dict = data_loader.preload_all_data()
    
    # Create memory-optimized arrays
    optimized_arrays = MemoryOptimizedArrays(data_dict)
    normalized_scores = (
        optimized_arrays.arrays['item_scores'],
        optimized_arrays.arrays['item_pair_scores'],
        optimized_arrays.arrays['position_scores'],
        optimized_arrays.arrays['position_pair_scores']
    )
    
    # Prepare MOO arrays
    arrays = prepare_scoring_arrays(
        items_to_assign=list(config.optimization.items_to_assign),
        positions_to_assign=list(config.optimization.positions_to_assign),
        norm_item_scores=normalized_scores[0],
        norm_item_pair_scores=normalized_scores[1],
        norm_position_scores=normalized_scores[2],
        norm_position_pair_scores=normalized_scores[3],
        items_assigned=list(config.optimization.items_assigned) if config.optimization.items_assigned else None,
        positions_assigned=list(config.optimization.positions_assigned) if config.optimization.positions_assigned else None
    )
    
    scorer = LayoutScorer(arrays, mode='multi_objective')
    print(f"Multi-objective scorer initialized")
    
    # Run parallel MOO
    print(f"\n🚀 Searching for Pareto-optimal solutions using {n_processes} processes...")
    if max_solutions:
        print(f"  Maximum solutions: {max_solutions}")
    if time_limit:
        print(f"  Time limit: {time_limit}s")
    
    start_time = time.time()
    
    moo_manager = ParallelMOOManager(config, n_processes)
    pareto_front, nodes_processed, nodes_pruned = moo_manager.parallel_multi_objective_search(
        scorer, normalized_scores, max_solutions, time_limit
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results
    if pareto_front:
        objective_names = ['Item Score', 'Item-Pair Score']
        print_moo_results(pareto_front, config, normalized_scores, objective_names)
        csv_path = save_moo_results_to_csv(pareto_front, config, normalized_scores, objective_names)
        print(f"\nResults saved to: {csv_path}")
    else:
        print("\nNo Pareto solutions found!")
    
    # Performance summary
    print(f"\n🏁 Distributed MOO Summary:")
    print(f"  Pareto solutions: {len(pareto_front)}")
    print(f"  Nodes processed: {nodes_processed:,}")
    print(f"  Processing rate: {nodes_processed/elapsed_time:.0f} nodes/sec")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Processes used: {n_processes}")
    if nodes_pruned > 0:
        prune_rate = 100 * nodes_pruned / (nodes_processed + nodes_pruned)
        print(f"  Nodes pruned: {nodes_pruned:,} ({prune_rate:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description='High-performance keyboard layout optimizer for HPC clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
HPC-Optimized Examples:
  # Single-objective with automatic process detection
  python hpc_optimize_layout.py --config config.yaml --n-solutions 10
  
  # Force specific number of processes
  python hpc_optimize_layout.py --config config.yaml --processes 32
  
  # Multi-objective optimization (distributed)
  python hpc_optimize_layout.py --config config.yaml --moo --processes 64
  
  # Memory profiling mode
  python hpc_optimize_layout.py --config config.yaml --profile-memory
        """
    )
    
    # Basic options
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed scoring breakdown')
    
    # HPC-specific options
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of processes (default: auto-detect)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Chunk size for parallel processing')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Enable memory profiling')
    
    # Optimization mode
    parser.add_argument('--moo', action='store_true',
                       help='Use distributed multi-objective optimization')
    
    # SOO options
    parser.add_argument('--n-solutions', type=int, default=5,
                       help='Number of top solutions to find')
    
    # MOO options
    parser.add_argument('--max-solutions', type=int, default=None,
                       help='Maximum Pareto solutions')
    parser.add_argument('--time-limit', type=float, default=None,
                       help='Time limit in seconds')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Memory profiling
    if args.profile_memory:
        import tracemalloc
        tracemalloc.start()
    
    try:
        if args.moo:
            run_distributed_moo(
                config=config,
                max_solutions=args.max_solutions,
                time_limit=args.time_limit,
                n_processes=args.processes
            )
        else:
            run_hpc_single_objective_optimization(
                config=config,
                n_solutions=args.n_solutions,
                verbose=args.verbose,
                n_processes=args.processes
            )
    finally:
        if args.profile_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f"\n📊 Memory Usage:")
            print(f"  Current: {current / 1024**2:.1f} MB")
            print(f"  Peak: {peak / 1024**2:.1f} MB")

if __name__ == "__main__":
    main()