#!/usr/bin/env python3
"""
Run sequential jobs for MOO layout optimization using configuration files.

Usage:
``python3 run_jobs.py --start-config 1 --end-config 1000``    # ascending
``python3 run_jobs.py --start-config 1000 --end-config 500``    # descending
``python3 run_jobs.py --reserve-cpus 2``    # reserve 2 CPUs for system
"""

import os
import sys
import glob
import subprocess
import psutil
import time
import threading
import queue
from pathlib import Path
from collections import deque
import argparse

# Configuration
CONFIG_PREFIX = "output/configs1/config_"
CONFIG_SUFFIX = ".yaml"
OUTPUT_DIR = "output/layouts"
TOTAL_CONFIGS = 1000
SCRIPT_PATH = "optimize_moo.py" 
OBJECTIVES = "engram_key_preference,engram_row_separation,engram_same_row,engram_same_finger"
# Adaptive scaling parameters
MAX_MEMORY_PERCENT = 90  # Scale down if memory exceeds this
MAX_CPU_PERCENT = 95     # Scale down if CPU exceeds this
MIN_FREE_MEMORY_GB = 1.0 # Scale down if free memory below this
TARGET_MEMORY_PERCENT = 80  # Ideal memory usage target
TARGET_CPU_PERCENT = 90     # Ideal CPU usage target

SCALE_UP_THRESHOLD = 2    # Scale up after 2 successful completions
SCALE_DOWN_IMMEDIATE = True  # Scale down immediately on overload
MIN_WORKERS = 1
MAX_WORKERS = 1  # Conservative default

class AdaptiveOptimizer:
    def __init__(self, config_ids, reserved_cpus=1, max_workers_override=None, show_output=False):
        self.config_ids = deque(config_ids)
        self.reserved_cpus = reserved_cpus
        self.show_output = show_output
        
        self.active_processes = {}  # pid -> (config_id, process, start_time)
        self.results_queue = queue.Queue()
        
        # Calculate max workers based on reserved CPUs
        total_cpus = psutil.cpu_count()
        available_cpus = max(1, total_cpus - self.reserved_cpus)
        
        # Adaptive scaling state - conservative for MOO
        self.current_workers = 1  # Start with single worker for MOO
        self.max_workers = max_workers_override or 1  # Conservative default
        self.successful_completions = 0
        self.last_scale_time = time.time()
        self.recent_completion_times = deque(maxlen=10)
        
        # Statistics
        self.total_processed = 0
        self.success_count = 0
        self.skip_count = 0
        self.error_count = 0
        
        print(f"🚀 Starting adaptive optimizer")
        print(f"   Workers: {self.current_workers} (conservative for MOO)")
        print(f"   Reserved CPUs: {self.reserved_cpus} (out of {total_cpus} total)")
        print(f"   Available CPUs for processing: {available_cpus}")
        print(f"   Configs to process: {len(self.config_ids)}")
        print()
    
    def get_system_resources(self):
        """Get current system resource usage"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent,
            'total_memory_gb': memory.total / (1024**3)
        }
    
    def config_already_completed(self, config_id):
        """Check if a config has already been processed."""
        pattern = f"{OUTPUT_DIR}/moo_results_config_{config_id}_*.csv"
        return len(glob.glob(pattern)) > 0
    
    def start_config_process(self, config_id):
        """Start processing a single config"""
        config_file = f"{CONFIG_PREFIX}{config_id}{CONFIG_SUFFIX}"
        
        if not os.path.exists(config_file):
            return None, "Config file not found"
        
        if self.config_already_completed(config_id):
            return None, "Already completed"
        
        try:
            cmd = [sys.executable, SCRIPT_PATH, "--config", config_file,
                "--objectives", OBJECTIVES]
            
            print(" ".join(cmd))
            
            if self.show_output:
                stdout_redirect = None
                stderr_redirect = None
            else:
                # Create log directory if it doesn't exist
                log_dir = Path("output/logs")
                log_dir.mkdir(exist_ok=True)
                
                # Redirect to log files
                log_file = log_dir / f"config_{config_id}.log"
                stdout_redirect = open(log_file, 'w')
                stderr_redirect = subprocess.STDOUT  # Combine stderr with stdout
                
            process = subprocess.Popen(
                cmd,
                stdout=stdout_redirect,
                stderr=stderr_redirect,
                text=True
            )
            
            # Store the file handle so we can close it later
            if not self.show_output and stdout_redirect:
                process._log_file = stdout_redirect
            
            return process, "Started"
        except Exception as e:
            return None, f"Failed to start: {str(e)}"

    def check_completed_processes(self):
        """Check for completed processes and collect results"""
        completed_pids = []
        
        for pid, (config_id, process, start_time) in self.active_processes.items():
            if process.poll() is not None:  # Process has finished
                completed_pids.append(pid)
                
                elapsed_time = time.time() - start_time
                self.recent_completion_times.append(elapsed_time)
                
                try:
                    # Close log file if we opened one
                    if hasattr(process, '_log_file') and process._log_file:
                        process._log_file.close()
                    
                    if process.returncode == 0:
                        print(f"✅ Config {config_id} completed in {elapsed_time:.1f}s")
                        self.success_count += 1
                        self.successful_completions += 1
                        result = {"id": config_id, "status": "success", "time": elapsed_time}
                    else:
                        print(f"❌ Config {config_id} failed (code {process.returncode}) after {elapsed_time:.1f}s")
                        self.error_count += 1
                        result = {"id": config_id, "status": "failed", "time": elapsed_time}
                    
                    self.results_queue.put(result)
                    
                except Exception as e:
                    print(f"💥 Error collecting results for config {config_id}: {str(e)}")
                    self.error_count += 1
        
        # Remove completed processes
        for pid in completed_pids:
            del self.active_processes[pid]

    def run(self):
        """Main processing loop with adaptive scaling"""
        last_status_time = time.time()
        
        print("🔄 Starting adaptive processing...")
        print("Press Ctrl+C to stop gracefully")
        print()
        
        try:
            while self.config_ids or self.active_processes:
                # Check for completed processes
                self.check_completed_processes()
                
                # Start new processes if we have capacity and configs
                while (len(self.active_processes) < self.current_workers and 
                       self.config_ids):
                    
                    config_id = self.config_ids.popleft()
                    process, message = self.start_config_process(config_id)
                    
                    if process is None:
                        if "Already completed" in message:
                            self.skip_count += 1
                            print(f"⭐️  Config {config_id}: {message}")
                        else:
                            self.error_count += 1
                            print(f"❌ Config {config_id}: {message}")
                    else:
                        self.active_processes[process.pid] = (config_id, process, time.time())
                        print(f"🔄 Started config {config_id} (worker {len(self.active_processes)}/{self.current_workers})")
                        self.show_status()
                
                # Show status periodically (every 60 seconds)
                #if time.time() - last_status_time > 60:
                #    self.show_status()
                #    last_status_time = time.time()
                
                # Small sleep to prevent busy waiting
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user, stopping gracefully...")
            
            # Terminate active processes
            for config_id, process, start_time in self.active_processes.values():
                print(f"🛑 Terminating config {config_id}...")
                process.terminate()
            
            # Wait a bit for graceful termination
            time.sleep(5)
            
            # Force kill if necessary
            for config_id, process, start_time in self.active_processes.values():
                if process.poll() is None:
                    print(f"💀 Force killing config {config_id}...")
                    process.kill()
    
    def show_status(self):
        """Show current status and system resources"""
        resources = self.get_system_resources()
        total_cpus = psutil.cpu_count()
        available_cpus = total_cpus - self.reserved_cpus
        
        print(f"\n📊 Status Report:")
        print(f"   Workers: {len(self.active_processes)}/{self.current_workers} active")
        print(f"   CPUs: {available_cpus} available, {self.reserved_cpus} reserved, {total_cpus} total")
        print(f"   Progress: ✅{self.success_count} ⭐️{self.skip_count} ❌{self.error_count}")
        print(f"   Remaining: {len(self.config_ids)} configs")
        print(f"   Resources: {resources['memory_percent']:.1f}% RAM, {resources['cpu_percent']:.1f}% CPU")
        print(f"   Free Memory: {resources['memory_available_gb']:.1f}GB")
        
        if self.recent_completion_times:
            avg_time = sum(self.recent_completion_times) / len(self.recent_completion_times)
            print(f"   Recent avg time: {avg_time:.1f}s per config")
        print()

def main():
    parser = argparse.ArgumentParser(description="Run MOO optimization jobs locally")
    parser.add_argument("--start-config", type=int, default=1,
                       help="Starting config ID (default: 1)")
    parser.add_argument("--end-config", type=int, default=TOTAL_CONFIGS,
                       help="Ending config ID (default: highest)")
    parser.add_argument("--reserve-cpus", type=int, default=1,
                       help="Number of CPU cores to keep free (default: 1)")
    parser.add_argument("--show-output", action="store_true",
                       help="Show optimize_moo.py output in real-time")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without running")
    
    args = parser.parse_args()
    
    # Validate reserve-cpus argument
    cpu_count = psutil.cpu_count()
    if args.reserve_cpus >= cpu_count:
        print(f"❌ Error: Cannot reserve {args.reserve_cpus} CPUs when system only has {cpu_count}")
        print(f"   Please use --reserve-cpus with a value less than {cpu_count}")
        sys.exit(1)
    
    if args.reserve_cpus < 0:
        print(f"❌ Error: --reserve-cpus must be 0 or greater")
        sys.exit(1)
    
    # Setup
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    print("=== MOO Layout Optimization Job Runner ===")
    
    # Show system info
    memory = psutil.virtual_memory()
    available_cpus = max(1, cpu_count - args.reserve_cpus)
    
    print(f"System: {cpu_count} CPUs, {memory.total / (1024**3):.1f}GB RAM")
    print(f"Reserved CPUs: {args.reserve_cpus}")
    print(f"Available CPUs: {available_cpus}")
    print(f"Script: {SCRIPT_PATH}")
    print(f"Objectives: {OBJECTIVES}")
    
    # Determine if we're going ascending or descending and generate config list accordingly
    if args.start_config <= args.end_config:
        # Ascending order
        config_ids = list(range(args.start_config, args.end_config + 1))
        print(f"Config range: {args.start_config} to {args.end_config} (ascending order)")
    else:
        # Descending order
        config_ids = list(range(args.start_config, args.end_config - 1, -1))
        print(f"Config range: {args.start_config} down to {args.end_config} (descending order)")
    
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if optimize_moo.py exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"❌ Error: {SCRIPT_PATH} not found in current directory")
        sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN - Would process configs:")
        for cid in config_ids[:10]:
            completed = len(glob.glob(f"{OUTPUT_DIR}/moo_results_config_{cid}_*.csv")) > 0
            status = "✅ completed" if completed else "⏳ pending"
            print(f"  Config {cid}: {status}")
        if len(config_ids) > 10:
            print(f"  ... and {len(config_ids) - 10} more")
        return
    
    # Run the adaptive optimizer
    optimizer = AdaptiveOptimizer(
        config_ids, 
        reserved_cpus=args.reserve_cpus,
        max_workers_override=1,  # Conservative for MOO
        show_output=args.show_output
    )
    optimizer.run()
    
    # Final summary
    print("\n=== Final Summary ===")
    print(f"✅ Successfully processed: {optimizer.success_count}")
    print(f"⭐️ Skipped (already done): {optimizer.skip_count}")
    print(f"❌ Errors/failures: {optimizer.error_count}")
    
    # Count total completed outputs
    total_outputs = len(glob.glob(f"{OUTPUT_DIR}/moo_results_config_*.csv"))
    print(f"📊 Total completed outputs: {total_outputs} / {TOTAL_CONFIGS}")
    
    print(f"\n📁 Output files are in: {output_dir}")

if __name__ == "__main__":
    main()