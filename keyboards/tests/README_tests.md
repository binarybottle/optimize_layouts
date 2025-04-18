# Keyboard layout optimization tests run on SLURM
The following are tests to process a single config file 
that optimizes 14 or 13 letter assignments to fill 18 keys.
12 letter assignments and above require Extreme Memory nodes.
Up to 11 letter assignments work fine on Regular Memory nodes.

RESULTS: 
All tests failed to complete within the required time,
and would exceed the allocated 75,000 compute-hours.

SOLUTION:
To fill 16 keys, we can use Regular Memory nodes by arranging the top 5
letters in 16 keys, and the next 11 letters in the remaining keys.

ACKNOWLEDGMENTS:
These tests are part of a study that used Bridges-2 resources 
at Pittsburgh Supercomputing Center through allocation MED250010 
from the Advanced Cyberinfrastructure Coordination Ecosystem: 
Services & Support (ACCESS) program, which is supported by 
National Science Foundation grants #2138259, #2138286, #2138307, 
#2137603, and #2138296. Citation:

  Brown, ST, Buitrago, P, Hanna, E, Sanielevici, S, Scibek, R, 
  Nystrom, NA (2021). Bridges-2: A Platform for Rapidly-Evolving 
  and Data Intensive Research. In Practice and Experience in 
  Advanced Research Computing (pp 1-4). doi:10.1145/3437359.3465593

———————————————————————————————————————————————————————————————————————————————
Setup
———————————————————————————————————————————————————————————————————————————————
# Set slurm_batchmaking.sh parameters for Extreme Memory nodes on PSC Bridges-2
#SBATCH --time=10:00:00       
#SBATCH --array=0-999%1000       
#SBATCH --ntasks-per-node=1   
#SBATCH --cpus-per-task=24    # Replacing 2,      
#SBATCH --mem=512GB           # Replacing 2GB, for Extreme Memory nodes on Bridges-2
#SBATCH --job-name=layouts     
#SBATCH --output=output/outputs/layouts_%A_%a.out
#SBATCH --error=output/errors/layouts_%A_%a.err
#SBATCH -p EM                 # Replacing RM-shared, for Extreme Memory nodes on Bridges-2   
#SBATCH -A <YOUR_ALLOCATION_ID>   
TOTAL_CONFIGS=<YOUR_TOTAL_CONFIGS> 
BATCH_SIZE=1000       

# Set slurm_submit_batches.sh parameters for Extreme Memory nodes on PSC Bridges-2
TOTAL_CONFIGS=<YOUR_TOTAL_CONFIGS>
BATCH_SIZE=1000                   
CHUNK_SIZE=3                        

```bash
#———————————————————————————————————————————————————————————————————————————————
# Arrange top 4 letters in 16 keys, optimize next 14 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
mv output/tests/configs1_assign_4_14_per_73440_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
mv output/configs output/tests/configs1_assign_4_14_per_73440_files

# TEST RESULT: Did not complete within 10 hours
# (Each config file would have to complete within 1 hour 
# not to exceed the allocated 75,000 compute-hours)

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 3 letters in 16 keys, next 14 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
mv output/tests/configs1_assign_1_3_14_per_8160_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
mv output/configs output/tests/configs1_assign_1_3_14_per_8160_files

# TEST RESULT: Did not complete within 10 hours
# (Each config file would have to complete within 10 hours 
# not to exceed the allocated 75,000 compute-hours)

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 4 letters in 16 keys, next 13 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
mv output/tests/configs1_assign_1_4_13_per_65520_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
mv output/configs output/tests/configs1_assign_1_4_13_per_65520_files

# TEST RESULT: Did not complete within 10 hours
# (Each config file would have to complete in less than about 1 hour 
# not to exceed the allocated 75,000 compute-hours)
```

#———————————————————————————————————————————————————————————————————————————————
# Arrange top 5 letters in 16 keys, optimize next 11 in remaining 16 keys
#———————————————————————————————————————————————————————————————————————————————

53569 jobs completed; 0 jobs failed

  ```bash
  for i in {1..10}; do 
    find output/layouts/config* | wc -l
  done | awk '{sum+=$1} END {print sum}'
  ```
  53569

  [aklein1@bridges2-login014 optimize_layouts]$ ls -d output/layouts/config* | wc -l
52459
[aklein1@bridges2-login014 optimize_layouts]$ ls output/layouts/layout* | wc -l
8313
