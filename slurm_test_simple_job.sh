#!/bin/bash
# slurm_test_simple_job.sh
# Simple SLURM job to test compute node environment

#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --job-name=test_env
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err
#SBATCH -p RM-shared
#SBATCH -A med250002p

echo "🧪 Testing Compute Node Environment"
echo "===================================="

# Basic info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Test module loading
echo -e "\nTesting module system on compute node:"
module purge
module load anaconda3

if [ $? -eq 0 ]; then
    echo "✅ anaconda3 module loaded successfully"
    echo "Python version: $(python3 --version)"
    echo "Python path: $(which python3)"
else
    echo "❌ Failed to load anaconda3 module"
    exit 1
fi

# Test Python packages
echo -e "\nTesting Python packages:"
python3 -c "
import sys
print(f'Python executable: {sys.executable}')

packages = ['numpy', 'pandas', 'yaml', 'multiprocessing', 'tqdm', 'psutil']
failed = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        failed.append(pkg)

if failed:
    print(f'Failed packages: {failed}')
    sys.exit(1)
else:
    print('✅ All required packages available')
"

# Test multiprocessing
echo -e "\nTesting multiprocessing:"
python3 -c "
import multiprocessing as mp
import os

print(f'CPU count (mp): {mp.cpu_count()}')
print(f'CPU count (os): {os.cpu_count()}')
print(f'SLURM CPUs: {os.environ.get(\"SLURM_CPUS_PER_TASK\", \"Not set\")}')

# Test parallel computation
def square(x):
    return x * x

with mp.Pool(processes=2) as pool:
    result = pool.map(square, [1, 2, 3, 4])
    print(f'Parallel test result: {result}')
    print('✅ Multiprocessing works')
"

# Test file I/O
echo -e "\nTesting file operations:"
TEST_FILE="test_compute_node_${SLURM_JOB_ID}.txt"
echo "Test data" > $TEST_FILE
if [ -f "$TEST_FILE" ]; then
    echo "✅ File creation works"
    rm $TEST_FILE
else
    echo "❌ File creation failed"
fi

# Test optimization imports
echo -e "\nTesting optimization script imports:"
python3 -c "
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

try:
    # Test if our modules can be imported
    modules_to_test = [
        'config',
        'scoring', 
        'search',
        'display'
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f'✅ {module} module imported')
        except ImportError as e:
            print(f'⚠️  {module} module not found: {e}')
    
    print('Module import test complete')
    
except Exception as e:
    print(f'Import test error: {e}')
"

echo -e "\n🏁 Compute Node Test Complete"
echo "End time: $(date)"
echo "If you see this message, the compute node environment is working!"