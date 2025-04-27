# This script checks for each config ID whether there's a job_completed.txt file
# For those that completed, checks if there's a corresponding layout file.
# If a job completed but has no layout file, adds it to the list of configs to rerun.

# Loop through all potential config IDs
for i in $(seq 1 62250); do
  # Check if job completed
  if [ -f "output/layouts/config_$i/job_completed.txt" ]; then
    # Check if layout file exists (using both potential naming patterns)
    if ! ls output/layouts/layout_results_${i}_*.csv &>/dev/null && \
       ! ls output/layouts/layout_results_config_${i}_*.csv &>/dev/null; then
      # No layout file found for this completed job
      echo $i >> configs_to_rerun.txt
      echo "Config $i has completed job but no layout file"
    fi
  fi
done
