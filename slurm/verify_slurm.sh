#!/bin/bash
# Verify the SLURM jobs are working correctly

echo "=== Current Job Status ==="
squeue -u $USER --format="%.10i %.9P %.15j %.8u %.2t %.10M %.6D %R"

echo ""
echo "=== Batch File Verification ==="
echo "First few batch files:"
for i in {1..3}; do
    if [ -f "output/batches/batch_${i}.txt" ]; then
        lines=$(wc -l < "output/batches/batch_${i}.txt")
        first=$(head -1 "output/batches/batch_${i}.txt")
        last=$(tail -1 "output/batches/batch_${i}.txt")
        echo "  batch_${i}.txt: $lines lines, configs $first-$last"
    fi
done

echo ""
echo "Last batch file:"
last_batch=$(ls output/batches/batch_*.txt | sort -V | tail -1)
if [ -f "$last_batch" ]; then
    batch_num=$(basename "$last_batch" .txt | sed 's/batch_//')
    lines=$(wc -l < "$last_batch")
    first=$(head -1 "$last_batch")
    last=$(tail -1 "$last_batch")
    echo "  batch_${batch_num}.txt: $lines lines, configs $first-$last"
fi

echo ""
echo "=== Array Range Verification ==="
echo "All jobs should have [0-499%3] or similar 0-based ranges:"
squeue -u $USER -h --format="%.18i %.20E" | while read job_id array_spec; do
    # Handle both formats: with and without array specification visible
    if [[ $job_id =~ ([0-9]+)_\[([0-9]+)-([0-9]+) ]]; then
        base_job=${BASH_REMATCH[1]}
        start=${BASH_REMATCH[2]}
        end=${BASH_REMATCH[3]}
        if [ $start -eq 0 ]; then
            echo "  ✓ Job $base_job: Array [$start-$end] - CORRECT"
        else
            echo "  ✗ Job $base_job: Array [$start-$end] - WRONG (should start at 0)"
        fi
    else
        echo "  ? Job $job_id: Array format not detected (may be truncated)"
    fi
done

echo ""
echo "Full job details (should show [0-499%3] arrays):"
squeue -u $USER

echo ""
echo "=== Running Task Check ==="
echo "When jobs start running, check individual tasks:"
echo "  squeue -u $USER -t R --format=\"%.15i %.10T %.15E\""
echo ""
echo "Check specific task logs:"
echo "  tail -f output/errors/layout_32930045_0.err"
echo "  tail -f output/outputs/layout_32930045_0.out"