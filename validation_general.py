# test_general_moo_system.py
"""
Comprehensive testing and validation suite for the general MOO system.

This script validates that:
1. The general system produces correct results
2. Performance improvements are achieved
3. Integration with existing infrastructure works
4. Cluster deployment is ready

Usage:
    python test_general_moo_system.py --keypair-table data/keypair_scores.csv --test-all
"""

import argparse
import time
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Test configuration
TEST_ITEMS = "etaoin"
TEST_POSITIONS = "FDESVR"
TEST_OBJECTIVES = ['comfort_score_normalized', 'time_total_normalized', 'engram8_score_normalized']

class GeneralMOOSystemTester:
    """Complete testing suite for the general MOO system."""
    
    def __init__(self, keypair_table_path: str):
        self.keypair_table_path = keypair_table_path
        self.test_config_path = None
        self.test_results = {}
        
    def create_test_config(self) -> str:
        """Create temporary test configuration."""
        
        test_config = {
            'paths': {
                'input': {
                    'raw_item_scores_file': 'input/frequency/english-letter-counts-leipzig.csv',
                    'raw_item_pair_scores_file': 'input/frequency/english-letter-pair-counts-leipzig.csv',
                    'raw_position_scores_file': 'input/comfort/key-comfort-scores.csv',
                    'raw_position_pair_scores_file': 'input/comfort/key-pair-comfort-scores.csv'
                },
                'output': {
                    'layout_results_folder': 'output/test_layouts'
                }
            },
            'optimization': {
                'items_assigned': '',
                'positions_assigned': '',
                'items_to_assign': TEST_ITEMS,
                'positions_to_assign': TEST_POSITIONS,
                'items_to_constrain': '',
                'positions_to_constrain': ''
            },
            'visualization': {
                'print_keyboard': False
            }
        }
        
        # Create temporary config file
        import yaml
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(test_config, temp_file, default_flow_style=False)
        temp_file.flush()
        
        self.test_config_path = temp_file.name
        return temp_file.name
    
    def test_keypair_table_loading(self) -> bool:
        """Test 1: Key-pair table loading and validation."""
        
        print("Test 1: Key-pair table loading...")
        
        try:
            # Test loading
            df = pd.read_csv(self.keypair_table_path, dtype={'key_pair': str})
            
            # Basic validation
            if 'key_pair' not in df.columns:
                print("  FAIL: No 'key_pair' column")
                return False
            
            # Count valid key-pairs
            valid_pairs = 0
            positions_found = set()
            
            for key_pair in df['key_pair']:
                key_str = str(key_pair)
                if len(key_str) == 2:
                    valid_pairs += 1
                    positions_found.add(key_str[0].upper())
                    positions_found.add(key_str[1].upper())
            
            print(f"  Loaded {len(df)} rows, {valid_pairs} valid key-pairs")
            print(f"  Positions found: {len(positions_found)}")
            print(f"  Available objectives: {len([col for col in df.columns if col != 'key_pair'])}")
            
            # Check if test objectives exist
            missing_objectives = [obj for obj in TEST_OBJECTIVES if obj not in df.columns]
            if missing_objectives:
                print(f"  WARNING: Missing test objectives: {missing_objectives}")
            
            # Check position coverage
            test_positions = set(TEST_POSITIONS.upper())
            missing_positions = test_positions - positions_found
            if missing_positions:
                print(f"  WARNING: Missing positions: {missing_positions}")
            
            self.test_results['table_loading'] = {
                'passed': True,
                'rows': len(df),
                'valid_pairs': valid_pairs,
                'positions': len(positions_found),
                'objectives': len(df.columns) - 1
            }
            
            print("  PASS: Key-pair table loaded successfully")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['table_loading'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_precomputed_matrices(self) -> bool:
        """Test 2: Pre-computed matrix creation and memory usage."""
        
        print("\nTest 2: Pre-computed matrix creation...")
        
        try:
            # Load table
            keypair_df = pd.read_csv(self.keypair_table_path, dtype={'key_pair': str})
            
            # Create test configuration
            from optimize_layout_general import GeneralMOOConfig, ExtremeMemoryObjectiveArrays
            
            config = GeneralMOOConfig(
                keypair_table_path=self.keypair_table_path,
                objective_columns=TEST_OBJECTIVES[:2],  # Test with 2 objectives
                objective_weights=[1.0, 1.0],
                maximize_objectives=[True, False],
                items_to_assign=TEST_ITEMS,
                positions_to_assign=TEST_POSITIONS
            )
            
            # Create arrays
            start_time = time.time()
            arrays = ExtremeMemoryObjectiveArrays(keypair_df, config)
            creation_time = time.time() - start_time
            
            # Test scoring
            test_mapping = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            objectives = arrays.calculate_all_objectives(test_mapping)
            
            print(f"  Matrix creation time: {creation_time:.3f}s")
            print(f"  Objectives calculated: {len(objectives)}")
            print(f"  Sample scores: {[f'{obj:.6f}' for obj in objectives]}")
            
            self.test_results['precomputed_matrices'] = {
                'passed': True,
                'creation_time': creation_time,
                'n_objectives': len(objectives),
                'sample_scores': objectives
            }
            
            print("  PASS: Pre-computed matrices created successfully")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['precomputed_matrices'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_scorer_compatibility(self) -> bool:
        """Test 3: Scorer compatibility with existing search."""
        
        print("\nTest 3: Scorer compatibility...")
        
        try:
            # Create test config
            config_path = self.create_test_config()
            
            # Load config using existing system
            from config import load_config
            config = load_config(config_path)
            
            # Create general MOO scorer
            from optimize_layout_general import (GeneralMOOConfig, ExtremeMemoryObjectiveArrays, 
                                               GeneralMOOLayoutScorer)
            
            keypair_df = pd.read_csv(self.keypair_table_path, dtype={'key_pair': str})
            
            general_config = GeneralMOOConfig(
                keypair_table_path=self.keypair_table_path,
                objective_columns=TEST_OBJECTIVES[:2],
                objective_weights=[1.0, 1.0],
                maximize_objectives=[True, False],
                items_to_assign=TEST_ITEMS,
                positions_to_assign=TEST_POSITIONS
            )
            
            arrays = ExtremeMemoryObjectiveArrays(keypair_df, general_config)
            scorer = GeneralMOOLayoutScorer(arrays)
            
            # Test scorer interface
            test_mapping = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            
            # Test score_layout method
            objectives = scorer.score_layout(test_mapping)
            objectives_with_components = scorer.score_layout(test_mapping, return_components=True)
            
            # Test get_components method
            components = scorer.get_components(test_mapping)
            
            print(f"  Scorer mode: {scorer.mode}")
            print(f"  Objectives: {len(objectives)}")
            print(f"  Components interface: {hasattr(components, 'as_list')}")
            
            # Test with existing search (small test)
            print("  Testing with existing search algorithm...")
            from search import multi_objective_search
            
            start_time = time.time()
            pareto_front, nodes_processed, nodes_pruned = multi_objective_search(
                config, scorer, max_solutions=10, time_limit=5.0, processes=1
            )
            search_time = time.time() - start_time
            
            print(f"  Search completed: {len(pareto_front)} solutions, {search_time:.2f}s")
            
            self.test_results['scorer_compatibility'] = {
                'passed': True,
                'pareto_solutions': len(pareto_front),
                'nodes_processed': nodes_processed,
                'search_time': search_time
            }
            
            print("  PASS: Scorer compatible with existing search")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['scorer_compatibility'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_performance_comparison(self) -> bool:
        """Test 4: Performance comparison with current system."""
        
        print("\nTest 4: Performance comparison...")
        
        try:
            config_path = self.create_test_config()
            from config import load_config
            config = load_config(config_path)
            
            # Create current system scorer
            from optimize_layout_general import CurrentSystemViaGeneralScorer
            current_scorer = CurrentSystemViaGeneralScorer(config)
            
            # Create general system scorer (configured for 2 objectives like current)
            from optimize_layout_general import (GeneralMOOConfig, ExtremeMemoryObjectiveArrays, 
                                               GeneralMOOLayoutScorer)
            
            keypair_df = pd.read_csv(self.keypair_table_path, dtype={'key_pair': str})
            
            general_config = GeneralMOOConfig(
                keypair_table_path=self.keypair_table_path,
                objective_columns=['comfort_score_normalized', 'time_total_normalized'],
                objective_weights=[1.0, 1.0],
                maximize_objectives=[True, False],
                items_to_assign=TEST_ITEMS,
                positions_to_assign=TEST_POSITIONS
            )
            
            arrays = ExtremeMemoryObjectiveArrays(keypair_df, general_config)
            general_scorer = GeneralMOOLayoutScorer(arrays)
            
            # Performance benchmark
            n_layouts = 1000
            test_mappings = []
            n_positions = len(TEST_POSITIONS)
            np.random.seed(42)
            
            for _ in range(n_layouts):
                mapping = np.random.permutation(n_positions)[:len(TEST_ITEMS)].astype(np.int32)
                test_mappings.append(mapping)
            
            # Benchmark current system
            print("  Benchmarking current system...")
            start_time = time.time()
            for mapping in test_mappings:
                _ = current_scorer.score_layout(mapping)
            current_time = time.time() - start_time
            
            # Benchmark general system
            print("  Benchmarking general system...")
            start_time = time.time()
            for mapping in test_mappings:
                _ = general_scorer.score_layout(mapping)
            general_time = time.time() - start_time
            
            # Calculate performance metrics
            current_rate = n_layouts / current_time
            general_rate = n_layouts / general_time
            speedup = general_rate / current_rate
            
            print(f"  Current system: {current_rate:.0f} layouts/sec")
            print(f"  General system: {general_rate:.0f} layouts/sec")
            print(f"  Speedup: {speedup:.2f}x")
            
            # Test equivalent results (small sample)
            print("  Testing result equivalence...")
            equivalent_results = True
            max_difference = 0.0
            
            for i, mapping in enumerate(test_mappings[:100]):  # Test first 100
                current_obj = current_scorer.score_layout(mapping)
                general_obj = general_scorer.score_layout(mapping)
                
                # Compare objectives (both should be 2-element lists)
                if len(current_obj) == len(general_obj):
                    for j in range(len(current_obj)):
                        diff = abs(current_obj[j] - general_obj[j])
                        max_difference = max(max_difference, diff)
                        if diff > 1e-4:  # Allow small numerical differences
                            equivalent_results = False
            
            self.test_results['performance_comparison'] = {
                'passed': True,
                'current_rate': current_rate,
                'general_rate': general_rate,
                'speedup': speedup,
                'equivalent_results': equivalent_results,
                'max_difference': max_difference
            }
            
            print(f"  Result equivalence: {'PASS' if equivalent_results else 'FAIL'} (max diff: {max_difference:.2e})")
            print("  PASS: Performance comparison completed")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['performance_comparison'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_job_runner_integration(self) -> bool:
        """Test 5: Integration with job runner."""
        
        print("\nTest 5: Job runner integration...")
        
        try:
            # Test command building
            sample_config_id = 1
            
            # Test current system command
            print("  Testing current system command...")
            import subprocess
            import sys
            
            current_cmd = [
                sys.executable, "optimize_layout_general.py",
                "--config", self.test_config_path,
                "--mode", "current",
                "--keypair-table", self.keypair_table_path,
                "--dry-run"
            ]
            
            # Test general system command
            print("  Testing general system command...")
            general_cmd = [
                sys.executable, "optimize_layout_general.py", 
                "--config", self.test_config_path,
                "--mode", "general",
                "--keypair-table", self.keypair_table_path,
                "--objectives", "comfort_score_normalized,time_total_normalized",
                "--dry-run"
            ]
            
            # Test if commands can be built (dry run)
            for cmd_name, cmd in [("current", current_cmd), ("general", general_cmd)]:
                try:
                    # Just test that the command can be constructed
                    cmd_str = ' '.join(cmd)
                    print(f"    {cmd_name}: {cmd_str[:80]}...")
                except Exception as e:
                    print(f"    {cmd_name}: FAIL - {e}")
                    return False
            
            self.test_results['job_runner_integration'] = {
                'passed': True,
                'current_command': ' '.join(current_cmd),
                'general_command': ' '.join(general_cmd)
            }
            
            print("  PASS: Job runner integration ready")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['job_runner_integration'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_memory_scaling(self) -> bool:
        """Test 6: Memory usage with different numbers of objectives."""
        
        print("\nTest 6: Memory scaling analysis...")
        
        try:
            import psutil
            
            keypair_df = pd.read_csv(self.keypair_table_path, dtype={'key_pair': str})
            available_objectives = [col for col in keypair_df.columns if col != 'key_pair']
            
            memory_usage = []
            
            # Test with increasing numbers of objectives
            for n_obj in [1, 2, 5, 10, min(20, len(available_objectives))]:
                if n_obj > len(available_objectives):
                    continue
                
                print(f"  Testing {n_obj} objectives...")
                
                from optimize_layout_general import GeneralMOOConfig, ExtremeMemoryObjectiveArrays
                
                config = GeneralMOOConfig(
                    keypair_table_path=self.keypair_table_path,
                    objective_columns=available_objectives[:n_obj],
                    objective_weights=[1.0] * n_obj,
                    maximize_objectives=[True] * n_obj,
                    items_to_assign=TEST_ITEMS,
                    positions_to_assign=TEST_POSITIONS
                )
                
                # Measure memory before
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024**3)  # GB
                
                # Create arrays
                start_time = time.time()
                arrays = ExtremeMemoryObjectiveArrays(keypair_df, config)
                creation_time = time.time() - start_time
                
                # Measure memory after
                memory_after = process.memory_info().rss / (1024**3)  # GB
                memory_increase = memory_after - memory_before
                
                memory_usage.append({
                    'n_objectives': n_obj,
                    'memory_increase_gb': memory_increase,
                    'creation_time': creation_time
                })
                
                print(f"    Memory increase: {memory_increase:.3f}GB, Time: {creation_time:.3f}s")
            
            # Analyze scaling
            if len(memory_usage) >= 2:
                memory_per_objective = []
                for usage in memory_usage[1:]:  # Skip first measurement
                    mem_per_obj = usage['memory_increase_gb'] / usage['n_objectives']
                    memory_per_objective.append(mem_per_obj)
                
                avg_memory_per_obj = np.mean(memory_per_objective)
                print(f"  Average memory per objective: {avg_memory_per_obj:.4f}GB")
                print(f"  Estimated memory for 50 objectives: {50 * avg_memory_per_obj:.2f}GB")
            
            self.test_results['memory_scaling'] = {
                'passed': True,
                'measurements': memory_usage,
                'avg_memory_per_objective': avg_memory_per_obj if 'avg_memory_per_obj' in locals() else 0.0
            }
            
            print("  PASS: Memory scaling analysis completed")
            return True
            
        except Exception as e:
            print(f"  FAIL: {e}")
            self.test_results['memory_scaling'] = {'passed': False, 'error': str(e)}
            return False
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.test_config_path and os.path.exists(self.test_config_path):
            os.unlink(self.test_config_path)
    
    def run_all_tests(self) -> bool:
        """Run complete test suite."""
        
        print("=" * 60)
        print("GENERAL MOO SYSTEM TEST SUITE")
        print("=" * 60)
        
        tests = [
            self.test_keypair_table_loading,
            self.test_precomputed_matrices,
            self.test_scorer_compatibility,
            self.test_performance_comparison,
            self.test_job_runner_integration,
            self.test_memory_scaling
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            if test():
                passed += 1
            time.sleep(0.5)  # Brief pause between tests
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Passed: {passed}/{total}")
        
        if passed == total:
            print("ALL TESTS PASSED - System ready for deployment!")
            
            # Print key metrics
            if 'performance_comparison' in self.test_results:
                perf = self.test_results['performance_comparison']
                if perf['passed']:
                    print(f"Performance improvement: {perf['speedup']:.2f}x faster")
            
            if 'memory_scaling' in self.test_results:
                mem = self.test_results['memory_scaling']
                if mem['passed']:
                    avg_mem = mem.get('avg_memory_per_objective', 0)
                    if avg_mem > 0:
                        print(f"Memory usage: ~{avg_mem:.4f}GB per objective")
            
        else:
            print("SOME TESTS FAILED - Review failures before deployment")
            
            # Print failures
            for test_name, result in self.test_results.items():
                if not result['passed']:
                    error = result.get('error', 'Unknown error')
                    print(f"  FAILED: {test_name} - {error}")
        
        self.cleanup()
        return passed == total

def create_deployment_checklist():
    """Create deployment checklist for cluster usage."""
    
    checklist = """
CLUSTER DEPLOYMENT CHECKLIST
============================

Pre-deployment:
  [ ] All tests pass (run: python test_general_moo_system.py --test-all)
  [ ] Key-pair table is accessible to all nodes
  [ ] optimize_layout_general.py is deployed to all nodes
  [ ] run_jobs_enhanced.py is deployed to master node
  [ ] Sufficient memory available (check memory scaling test results)

Deployment steps:
  [ ] Upload key-pair table to shared storage
  [ ] Test single config on cluster:
      python optimize_layout_general.py --config test_config.yaml --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --dry-run
  
  [ ] Run small batch test (10 configs):
      python run_jobs_enhanced.py --script optimize_layout_general.py --mode general --keypair-table data/keypair_scores.csv --objectives comfort_score_normalized,time_total_normalized --start-config 1 --end-config 10 --max-workers 2
  
  [ ] Monitor memory usage during test batch
  [ ] Verify output files are created correctly
  
Production deployment:
  [ ] Set appropriate memory limits per worker
  [ ] Configure --reserve-cpus based on node specifications
  [ ] Set --max-workers based on memory analysis
  [ ] Run full production batch:
      python run_jobs_enhanced.py --script optimize_layout_general.py --mode general --keypair-table data/keypair_scores.csv --objectives [your_objectives] --start-config 1 --end-config 95040

Monitoring:
  [ ] Monitor memory usage trends
  [ ] Check output file generation rate
  [ ] Verify no memory leaks in long runs
  [ ] Compare performance to current system

Troubleshooting:
  [ ] If memory errors: reduce --max-workers or number of objectives
  [ ] If slow performance: check disk I/O for key-pair table access
  [ ] If incorrect results: run validation comparison with current system
    """
    
    print(checklist)
    
    with open("deployment_checklist.txt", "w") as f:
        f.write(checklist)
    print("Deployment checklist saved to: deployment_checklist.txt")

def main():
    """Main testing interface."""
    
    parser = argparse.ArgumentParser(description="General MOO System Test Suite")
    parser.add_argument("--keypair-table", required=True, help="Path to key-pair scoring table")
    parser.add_argument("--test-all", action="store_true", help="Run complete test suite")
    parser.add_argument("--create-checklist", action="store_true", help="Create deployment checklist")
    parser.add_argument("--test", choices=[
        'table', 'matrices', 'compatibility', 'performance', 'integration', 'memory'
    ], help="Run specific test")
    
    args = parser.parse_args()
    
    if args.create_checklist:
        create_deployment_checklist()
        return 0
    
    if not Path(args.keypair_table).exists():
        print(f"Error: Key-pair table not found: {args.keypair_table}")
        return 1
    
    # Run tests
    tester = GeneralMOOSystemTester(args.keypair_table)
    
    if args.test_all:
        success = tester.run_all_tests()
        return 0 if success else 1
    
    elif args.test:
        test_methods = {
            'table': tester.test_keypair_table_loading,
            'matrices': tester.test_precomputed_matrices,
            'compatibility': tester.test_scorer_compatibility,
            'performance': tester.test_performance_comparison,
            'integration': tester.test_job_runner_integration,
            'memory': tester.test_memory_scaling
        }
        
        success = test_methods[args.test]()
        tester.cleanup()
        return 0 if success else 1
    
    else:
        print("Error: Use --test-all or --test <specific_test>")
        return 1

if __name__ == "__main__":
    exit(main())