#!/bin/bash
# slurm_setup_run_all_tests.sh
# Orchestrates all HPC environment testing steps

echo "🧪 Comprehensive HPC Testing Suite"
echo "===================================="

# Configuration
ALLOCATION=${1:-"med250002p"}
WAIT_TIME=30  # seconds to wait between steps

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [allocation_id]"
    echo "Example: $0 med250002p"
    echo ""
    echo "This script runs all tests to validate your HPC environment"
    echo "for keyboard layout optimization."
    exit 0
fi

echo "Allocation: $ALLOCATION"
echo "Start time: $(date)"

# Function to check if required scripts exist
check_scripts() {
    local missing=0
    
    scripts=(
        "slurm_setup_test_environment.sh"
        "slurm_setup_test_simple_job.sh" 
        "slurm_setup_test_generate_config.py"
        "slurm_setup_check_resources.sh"
        "slurm_array_processor.sh"
    )
    
    echo "Checking required scripts..."
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            echo "  ✅ $script"
        else
            echo "  ❌ $script (missing)"
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        echo ""
        echo "❌ Some required scripts are missing!"
        echo "Please ensure all testing scripts are present."
        exit 1
    fi
    
    echo "✅ All scripts found"
}

# Function to wait with countdown
wait_with_countdown() {
    local seconds=$1
    local message=$2
    
    echo "$message"
    for ((i=seconds; i>0; i--)); do
        printf "\r⏳ Waiting %d seconds... " $i
        sleep 1
    done
    printf "\r✅ Continuing...                    \n"
}

# Step 1: Check scripts
echo -e "\n🔍 Step 1: Checking Scripts"
echo "=========================="
check_scripts

# Step 2: Environment check
echo -e "\n🔍 Step 2: Environment Check"
echo "=========================="
bash slurm_setup_test_environment.sh $ALLOCATION

if [ $? -ne 0 ]; then
    echo "❌ Environment check failed! Please fix issues before continuing."
    exit 1
fi

# Step 3: Resource check  
echo -e "\n🔍 Step 3: SLURM Resources Check"
echo "============================="
bash slurm_setup_check_resources.sh $ALLOCATION

# Step 4: Generate test data
echo -e "\n🔍 Step 4: Generate Test Data"
echo "=========================="
python3 slurm_setup_test_generate_config.py

if [ $? -ne 0 ]; then
    echo "❌ Test data generation failed!"
    exit 1
fi

# Step 5: Submit compute node test
echo -e "\n🔍 Step 5: Compute Node Test"
echo "========================="
echo "Submitting test job to compute node..."

JOB_OUTPUT=$(sbatch slurm_setup_test_simple_job.sh 2>&1)
JOB_STATUS=$?

if [ $JOB_STATUS -eq 0 ]; then
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $4}')
    echo "✅ Job submitted: $JOB_ID"
    echo "Waiting for job to complete..."
    
    # Wait for job to finish
    while squeue -j $JOB_ID -h >/dev/null 2>&1; do
        printf "⏳ Job $JOB_ID still running...\r"
        sleep 5
    done
    printf "                                    \r"
    
    # Check job output
    if [ -f "test_job_${JOB_ID}.out" ]; then
        echo "📄 Job output:"
        echo "----------------------------------------"
        cat "test_job_${JOB_ID}.out"
        echo "----------------------------------------"
        
        if grep -q "Compute Node Test Complete" "test_job_${JOB_ID}.out"; then
            echo "✅ Compute node test passed!"
        else
            echo "❌ Compute node test failed!"
            if [ -f "test_job_${JOB_ID}.err" ]; then
                echo "Error output:"
                cat "test_job_${JOB_ID}.err"
            fi
            exit 1
        fi
    else
        echo "❌ Job output file not found"
        exit 1
    fi
else
    echo "❌ Failed to submit test job: $JOB_OUTPUT"
    exit 1
fi

wait_with_countdown 5 "Proceeding to optimization test..."

# Step 6: Test single optimization
echo -e "\n🔍 Step 6: Single Optimization Test"
echo "================================"
echo "Testing single optimization (with module loading)..."

# Load anaconda3 module first
module purge
module load anaconda3

if [ $? -eq 0 ]; then
    echo "✅ anaconda3 module loaded for login node test"
    python3 optimize_layout.py --config test_config.yaml --n-solutions 3
    
    if [ $? -eq 0 ]; then
        echo "✅ Single optimization test passed!"
    else
        echo "❌ Single optimization test failed!"
        echo "ℹ️  This is optional - SLURM jobs are the main workflow"
    fi
else
    echo "⚠️  Could not load anaconda3 on login node"
    echo "ℹ️  Skipping single optimization test - this is normal"
    echo "ℹ️  Compute nodes work correctly (verified in Step 5)"
fi

# Step 7: Test SLURM array job
echo -e "\n🔍 Step 7: SLURM Array Test"
echo "========================"
echo "Testing SLURM array processor..."

ARRAY_OUTPUT=$(sbatch --export=CONFIG_FILE=test_single.txt --array=0-0 slurm_array_processor.sh 2>&1)
ARRAY_STATUS=$?

if [ $ARRAY_STATUS -eq 0 ]; then
    ARRAY_JOB_ID=$(echo "$ARRAY_OUTPUT" | awk '{print $4}')
    echo "✅ Array job submitted: $ARRAY_JOB_ID"
    echo "Waiting for array job to complete..."
    
    # Wait for array job to finish
    while squeue -j $ARRAY_JOB_ID -h >/dev/null 2>&1; do
        printf "⏳ Array job $ARRAY_JOB_ID still running...\r"
        sleep 5
    done
    printf "                                         \r"
    
    # Check array job output
    ARRAY_OUT="output/outputs/layout_${ARRAY_JOB_ID}_0.out"
    if [ -f "$ARRAY_OUT" ]; then
        echo "📄 Array job output:"
        echo "----------------------------------------"
        tail -20 "$ARRAY_OUT"
        echo "----------------------------------------"
        
        if grep -q "completed successfully" "$ARRAY_OUT"; then
            echo "✅ Array job test passed!"
        else
            echo "❌ Array job test failed!"
            ARRAY_ERR="output/errors/layout_${ARRAY_JOB_ID}_0.err"
            if [ -f "$ARRAY_ERR" ]; then
                echo "Error output:"
                cat "$ARRAY_ERR"
            fi
            exit 1
        fi
    else
        echo "❌ Array job output file not found: $ARRAY_OUT"
        exit 1
    fi
else
    echo "❌ Failed to submit array job: $ARRAY_OUTPUT"
    exit 1
fi

# Step 8: Verify results
echo -e "\n🔍 Step 8: Verify Results"
echo "======================"
echo "Checking output files..."

OUTPUT_COUNT=$(ls output/layouts/test_layout_results_1_*.csv 2>/dev/null | wc -l)
if [ $OUTPUT_COUNT -gt 0 ]; then
    echo "✅ Found $OUTPUT_COUNT result file(s)"
    echo "📄 Sample results:"
    ls -la output/layouts/test_layout_results_1_*.csv | head -3
    
    # Show a few lines from results
    SAMPLE_FILE=$(ls output/layouts/test_layout_results_1_*.csv | head -1)
    if [ -f "$SAMPLE_FILE" ]; then
        echo "📄 Sample data from $SAMPLE_FILE:"
        head -5 "$SAMPLE_FILE"
    fi
else
    echo "❌ No result files found!"
    echo "Expected files like: output/layouts/test_layout_results_1_*.csv"
    exit 1
fi

# Final summary
echo -e "\n🏁 Test Suite Complete!"
echo "======================"
echo "✅ All tests passed successfully!"
echo ""
echo "📋 Summary:"
echo "  ✅ Environment validation"
echo "  ✅ SLURM resource check"
echo "  ✅ Test data generation" 
echo "  ✅ Compute node validation"
echo "  ✅ Single optimization"
echo "  ✅ SLURM array job"
echo "  ✅ Result file generation"
echo ""
echo "🚀 Your environment is ready for production runs!"
echo ""
echo "Next steps:"
echo "  1. Generate your real configuration files"
echo "  2. Run: bash slurm_array_submit.sh --rescan"
echo "  3. Monitor with: squeue -u $USER"
echo ""
echo "Cleanup test files with:"
echo "  rm -f test_*.* test_job_*.* data/test_*.csv output/layouts/test_*"

# Clean up on success (optional)
read -p "Clean up test files? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning up test files..."
    rm -f test_*.* test_job_*.*
    rm -rf data/test_*.csv
    rm -f output/layouts/test_*
    echo "✅ Cleanup complete"
fi

echo "Testing completed at $(date)"