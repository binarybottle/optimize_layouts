#!/bin/bash
# slurm_check_resources.sh
# Check SLURM resources, quotas, and system status

echo "📊 SLURM Resource and Quota Check"
echo "=================================="

# Get allocation from command line or environment
ALLOCATION=${1:-$SLURM_ACCOUNT}
if [ -z "$ALLOCATION" ]; then
    echo "Usage: $0 <allocation_id>"
    echo "Example: $0 med250002p"
    echo ""
fi

echo "User: $USER"
echo "Date: $(date)"
if [ ! -z "$ALLOCATION" ]; then
    echo "Allocation: $ALLOCATION"
fi

# 1. Check current jobs
echo -e "\n1. Current Job Status:"
RUNNING_JOBS=$(squeue -u $USER -h | wc -l)
if [ $RUNNING_JOBS -eq 0 ]; then
    echo "   ✅ No jobs currently running"
else
    echo "   📊 $RUNNING_JOBS jobs found:"
    squeue -u $USER -o "   %8A %12j %8T %10M %4C %8m" | head -10
    if [ $RUNNING_JOBS -gt 10 ]; then
        echo "   ... and $((RUNNING_JOBS - 10)) more jobs"
    fi
fi

# 2. Partition information
echo -e "\n2. Available Partitions:"
sinfo -o "%15P %5a %10l %6D %6c %8z %10m %25f" | head -10

# 3. Node availability
echo -e "\n3. Node Availability (RM-shared partition):"
sinfo -p RM-shared -o "%10P %5a %4D %4c %8z %10m %6t"

# 4. Check allocation status
if [ ! -z "$ALLOCATION" ]; then
    echo -e "\n4. Allocation Status:"
    
    # Check if user has access to allocation
    ASSOC_CHECK=$(sacctmgr show assoc user=$USER account=$ALLOCATION -n 2>/dev/null | wc -l)
    if [ $ASSOC_CHECK -gt 0 ]; then
        echo "   ✅ Access to allocation $ALLOCATION confirmed"
        sacctmgr show assoc user=$USER account=$ALLOCATION format=account,user,partition,qos -n | sed 's/^/   /'
    else
        echo "   ❌ No access to allocation $ALLOCATION"
        echo "   Available allocations:"
        sacctmgr show assoc user=$USER format=account -n | sort -u | sed 's/^/      /'
    fi
    
    # Show recent usage
    echo -e "\n5. Recent Usage (last 7 days):"
    sacct -S $(date -d '7 days ago' +%Y-%m-%d) -u $USER -A $ALLOCATION --format=JobID,JobName,State,CPUTime,Elapsed,MaxRSS | head -10
else
    echo -e "\n4. Your Allocations:"
    sacctmgr show assoc user=$USER format=account,partition,qos -n | sort -u | sed 's/^/   /'
fi

# 5. Resource limits
echo -e "\n6. Current Resource Limits:"

# Check for any active limits
LIMITS=$(scontrol show config | grep -E "(MaxJobsAccrue|MaxJobsSubmit|MaxNodes|MaxCPUs)" | head -5)
if [ ! -z "$LIMITS" ]; then
    echo "   System limits:"
    echo "$LIMITS" | sed 's/^/      /'
else
    echo "   No specific system limits found"
fi

# 7. Recommended settings for this system
echo -e "\n7. Recommended SLURM Settings for RM-shared:"
echo "   --partition=RM-shared"
echo "   --cpus-per-task=1-8 (max cores per task)"
echo "   --mem=1GB-15GB (max ~1.9GB per CPU)"
echo "   --time=1:00:00-48:00:00 (max walltime)"
echo "   --ntasks-per-node=1 (for single-node jobs)"

# 8. Check disk space
echo -e "\n8. Disk Space:"
echo "   Home directory:"
df -h $HOME | tail -1 | awk '{print "      Total: " $2 ", Used: " $3 ", Available: " $4 " (" $5 " used)"}'

echo "   Current directory:"
df -h . | tail -1 | awk '{print "      Total: " $2 ", Used: " $3 ", Available: " $4 " (" $5 " used)"}'

# 9. Estimate job capacity
echo -e "\n9. Job Capacity Estimates:"
if [ ! -z "$ALLOCATION" ]; then
    # Get CPU hours available (this is approximate)
    echo "   For 8-CPU, 2-hour jobs: ~125 jobs per 1000 CPU-hours"
    echo "   For 8-CPU, 4-hour jobs: ~62 jobs per 1000 CPU-hours"
    echo "   For batch size 500: ~1-2 array jobs (depending on resources)"
else
    echo "   Provide allocation ID to see capacity estimates"
fi

# 10. System status
echo -e "\n10. Current System Status:"
NODES_DOWN=$(sinfo -h -p RM-shared -t down | wc -l)
NODES_TOTAL=$(sinfo -h -p RM-shared | wc -l)
echo "    Nodes down: $NODES_DOWN / $NODES_TOTAL"

QUEUE_LENGTH=$(squeue -p RM-shared -h | wc -l)
echo "    Jobs in RM-shared queue: $QUEUE_LENGTH"

# Summary
echo -e "\n🏁 Quick Start Recommendations:"
echo "   1. Use allocation: ${ALLOCATION:-'[specify your allocation]'}"
echo "   2. Start with small test: 1-5 jobs"
echo "   3. Monitor with: squeue -u $USER"
echo "   4. Scale up gradually based on queue times"

if [ $RUNNING_JOBS -gt 50 ]; then
    echo -e "\n⚠️  Warning: You have many jobs running ($RUNNING_JOBS)"
    echo "   Consider reducing concurrent jobs to be courteous to other users"
fi