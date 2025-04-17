# Keyboard layout optimization tests run on SLURM
The following are tests to process a single config file that 
optimizes 14, 13, 12, then 11 letter assignments.

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
ssh aklein1@bridges2.psc.edu

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

# Reattach screen [replace "submission"]
screen -r submission

```bash
#———————————————————————————————————————————————————————————————————————————————
# Arrange top 4 letters in 16 keys, optimize next 14 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
rm -rf output/errors/* output/outputs/* output/layouts/*
mv output/tests/configs1_assign_4_14_per_73440_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
# Move configs back to output/tests/
mv output/configs output/tests/configs1_assign_4_14_per_73440_files

# TEST RESULT: Did not complete within 10 hours
# (Each config file would have to complete within 1 hour 
# not to exceed the allocated 75K compute-hours)

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 3 letters in 16 keys, next 14 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
rm -rf output/errors/* output/outputs/* output/layouts/*
mv output/tests/configs1_assign_1_3_14_per_8160_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
# Move configs back to output/tests/
mv output/configs output/tests/configs1_assign_1_3_14_per_8160_files

# TEST RESULT: Would not complete within 10 hours (see above)
# (Each config file would have to complete within 10 hours 
# not to exceed the allocated 75K compute-hours)

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 4 letters in 16 keys, next 13 in remaining 18 keys
#———————————————————————————————————————————————————————————————————————————————
rm -rf output/errors/* output/outputs/* output/layouts/*
mv output/tests/configs1_assign_1_4_13_per_65520_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
# Move configs back to output/tests/
mv output/configs output/tests/configs1_assign_1_4_13_per_65520_files

# TEST RESULT: Did not complete within 10 hours
# (Each config file would have to complete within 1-2 hours 
# not to exceed the allocated 75K compute-hours)

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 3 letters in 16 keys, next 12 in remaining 16 keys
#———————————————————————————————————————————————————————————————————————————————
rm -rf output/errors/* output/outputs/* output/layouts/*
mv output/tests/configs1_assign_1_3_12_per_5460_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
# Move configs back to output/tests/
mv output/configs output/tests/configs1_assign_1_3_12_per_5460_files

 configs1_assign_4_12_per_73440_files ???

#———————————————————————————————————————————————————————————————————————————————
# Arrange 'e' in 2 keys, next 4 letters in 16 keys, next 11 in remaining 16 keys
#———————————————————————————————————————————————————————————————————————————————
rm -rf output/errors/* output/outputs/* output/layouts/*
mv output/tests/configs1_assign_1_4_11_per_65520_files output/configs
sbatch --export=BATCH_NUM=0 --array=0 slurm_batchmaking.sh
# Move configs back to output/tests/
mv output/configs output/tests/configs1_assign_1_4_11_per_65520_files
```

 RM vs EM ???

