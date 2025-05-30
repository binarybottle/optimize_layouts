#!/bin/bash
# slurm_test_environment.sh
# Comprehensive test of HPC environment for keyboard optimization

echo "🧪 Testing HPC Environment for Keyboard Optimization"
echo "=================================================="

# Test 1: Basic system info
echo "1. System Information:"
echo "   Hostname: $(hostname)"
echo "   User: $USER"
echo "   Date: $(date)"
echo "   PWD: $PWD"

# Test 2: Module system
echo -e "\n2. Testing Module System:"
if command -v module &> /dev/null; then
    echo "   ✅ Module command available"
    echo "   Available Python modules:"
    module avail python 2>&1 | grep -i python | head -5 | sed 's/^/      /'
    module avail anaconda 2>&1 | grep -i anaconda | head -3 | sed 's/^/      /'
else
    echo "   ❌ Module command not found"
    exit 1
fi

# Test 3: Load anaconda3 and test Python
echo -e "\n3. Testing Python Environment:"
module purge
module load anaconda3 2>/dev/null

if [ $? -eq 0 ]; then
    echo "   ✅ anaconda3 module loaded successfully"
    echo "   Python version: $(python3 --version)"
    echo "   Python location: $(which python3)"
else
    echo "   ❌ Failed to load anaconda3 module"
    exit 1
fi

# Test 4: Required packages
echo -e "\n4. Testing Required Packages:"
PACKAGES=("numpy" "pandas" "yaml" "tqdm" "multiprocessing" "psutil")
ALL_GOOD=true

for pkg in "${PACKAGES[@]}"; do
    if python3 -c "import $pkg" 2>/dev/null; then
        echo "   ✅ $pkg available"
    else
        echo "   ❌ $pkg NOT available"
        ALL_GOOD=false
    fi
done

# Test 5: SLURM commands
echo -e "\n5. Testing SLURM Commands:"
SLURM_COMMANDS=("sbatch" "squeue" "scancel" "sinfo")
for cmd in "${SLURM_COMMANDS[@]}"; do
    if command -v $cmd &> /dev/null; then
        echo "   ✅ $cmd available"
    else
        echo "   ❌ $cmd NOT available"
        ALL_GOOD=false
    fi
done

# Test 6: Check allocation (if provided)
echo -e "\n6. Testing SLURM Allocation:"
if [ ! -z "$SLURM_ACCOUNT" ]; then
    echo "   Current allocation: $SLURM_ACCOUNT"
elif [ ! -z "$1" ]; then
    echo "   Testing allocation: $1"
    sacctmgr show assoc user=$USER account=$1 format=account,user,partition -n 2>/dev/null | head -3
else
    echo "   No allocation specified. Usage: $0 [allocation_id]"
    echo "   Example: $0 med250002p"
fi

# Test 7: Disk space
echo -e "\n7. Testing Disk Space:"
echo "   Home directory: $(df -h $HOME | tail -1 | awk '{print $4}') available"
echo "   Current directory: $(df -h . | tail -1 | awk '{print $4}') available"

# Test 8: Directory structure
echo -e "\n8. Testing Directory Structure:"
REQUIRED_DIRS=("output" "output/layouts" "output/outputs" "output/errors" "output/configs1")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir exists"
    else
        echo "   ⚠️  $dir missing (will be created)"
        mkdir -p "$dir"
    fi
done

# Test 9: File permissions
echo -e "\n9. Testing File Permissions:"
SCRIPTS=("slurm_array_processor.sh" "slurm_quota_smart_array_submit.sh" "slurm_optimize_layout.py")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "   ✅ $script is executable"
        else
            echo "   ⚠️  $script not executable (fixing)"
            chmod +x "$script"
        fi
    else
        echo "   ❌ $script not found"
        ALL_GOOD=false
    fi
done

# Test 10: Memory and CPU info
echo -e "\n10. System Resources:"
if command -v nproc &> /dev/null; then
    echo "    CPU cores: $(nproc)"
fi
if [ -f /proc/meminfo ]; then
    MEM_GB=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)
    echo "    Total memory: ${MEM_GB} GB"
fi

# Summary
echo -e "\n🏁 Environment Test Summary:"
if [ "$ALL_GOOD" = true ]; then
    echo "   ✅ All tests passed! Environment looks good."
    echo "   You can proceed with running the optimization scripts."
else
    echo "   ❌ Some tests failed. Please fix the issues above."
    echo "   Contact your system administrator if you need help."
fi

echo -e "\n📋 Next Steps:"
echo "   1. Test simple job: sbatch slurm_test_simple_job.sh"
echo "   2. Generate test config: python3 slurm_test_generate_config.py"
echo "   3. Run single optimization: sbatch --export=CONFIG_FILE=test_single.txt --array=0-0 slurm_array_processor.sh"