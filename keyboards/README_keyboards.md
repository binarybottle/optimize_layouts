# README_keyboards

Optimize an English language keyboard layout

**Repository**: https://github.com/binarybottle/optimize_layouts.git  
**Author**: Arno Klein (arnoklein.info)  
**License**: MIT License (see LICENSE)

## Introduction
Let's apply the optimize_layouts software to the challenge of optimizing the 
arrangement of letters on a computer keyboard for typing (examples in English). 

For the following, we:
  - Assume bilateral symmetry in left- and right-hand ergonomics.
  - Focus on the 24 keys in, above, and below each home row ("home blocks")
    to reduce lateral stretching (the two middle columns are removed in figures):
    FDRSEVAWCQXZJKULIM;O,P./ (qwerty keys, excluding ' and [).
  - Focus on the 24 most frequent letters in English to assign to the 24 keys:
    etaoinsrhldcumfpgwybvkxj (q or z are dealt with at the end).
  - Refer to the "comfort" of a key (and key-pairs) based on typing preference 
    research data, with comfort ranking 1 (high) to 12 (low) for each hand:

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │  10 │  7  │  3  │  6  │    │  6  │  3  │  7  │  10 │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  8  │  4  │  2  │  1  │    │  1  │  2  │  4  │  8  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │  11 │  12 │  9  │  5  │    │  5  │  9  │  12 │  11 │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

## Overview of steps
1. Optimally arrange the 16 most frequent letters in the 16 most comfortable keys.
  (The remaining 8 letters have negligible interaction with the top 16 letters,
  and the remaining 8 keys have much lower comfort scores than the top 16 keys.)
  Use multi-objective optimization (MOO, with item- and item-pair objectives) 
  to find a Pareto front of equally viable, 16-letter/key layout candidates.
    1. To parallelize the branch-and-bound, depth-first search, 
       fix the 5 most frequent letters in every possible arrangement (95,040) 
       within the 12 most comfortable keys (half of our 24 keys in the home blocks).
    2. For each of the 95,040 possible arrangements of 5 letters, 
       use MOO to optimally arrange 12 letters in the 12 available of 16 keys. 
       The result is 95,040 Pareto fronts, each with 20-30 solutions.
    3. Select the global Pareto front of MOO solutions from the 16-letter/key layouts.
2. Optimally arrange the remaining letters.
    1. Remove 2 letters from the 2 least comfortable of the 16 keys 
       in the selected layouts (to explore a broader solution space).
    2. For each resulting 14-letter/key layout, optimally arrange the 10 
       (out of 24) remaining letters in the 10 remaining keys, again using MOO.
    3. Select the global Pareto front of MOO solutions from the 24-letter/key layouts.
3. Select the final layout.
    1. For each MOO objective, replace scores with their rankings.
    2. Sum the rankings for each layout.
    3. Sort layouts by these sums.
    4. Select layout(s) with the lowest sum (lowest is best).
4. Arrange periphery of the 12-letter/key home blocks.
    1. Assign the two least frequent letters (q & z in English) 
       to the two upper-right corner keys.
    2. Assign punctuation to the two middle columns between the home blocks. 

## Commands and visuals for steps 1 and 2.

### Step 1. Optimally arrange the 16 most frequent letters in the 16 most comfortable keys.

  **1.1. Fix 5 letters in every possible arrangement within 12 keys.**

  There are 95,040 permutations of 5 letters in 12 keys.
  Command for generating the 95,040 configuration files:

  ```bash
    cd keyboards; python3 generate_configs1.py
  ```

  In the example below, the 5 most frequent letters (etaoi in English)
  are assigned to 5 of the 12 most comfortable keys in the home blocks
  (available keys are empty; restricted keys are represented by "|||||"):

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │||||||     │  o  │     │    │     │     │     │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │     │  i  │  e  │  a  │    │     │  t  │     │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│     │    │     │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **1.2. Optimally arrange 12 letters in the 12 available of the top 16 keys.**

  Command for optimizing layouts, with constraints specified per configuration file:

  ```bash
    bash run_jobs_local.sh
  ```

  This optimally arranges 12 letters in 12 available keys 
  (479,001,600 permutations) for each of the 95,040 configurations above:

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │||||||  f  │  o  │  u  │    │  l  │  d  │  m  │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  i  │  e  │  a  │    │  h  │  t  │  s  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│  p  │    │  r  │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **1.3. Select the global Pareto front from the 16-letter/key layouts.**

  ```bash
    python3 select_global_moo_solutions.py
    python3 analyze_global_moo_solutions.py output/global_moo_solutions.csv
  ```

### Step 2. Optimally arrange the remaining letters.
The following steps act on each of Step 1's output files. 

  **2.1. Remove 2 letters from the 2 least comfortable of the 16 keys.**

  Command for generating the second set of configuration files:

  ```bash
    cd keyboards
    python3 generate_configs2.py --input-file ../output/global_moo_solutions.csv --remove-positions "A;"
  ```

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │     |  f  │  o  │  u  │    │  l  │  d  │  m  │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │     │  i  │  e  │  a  │    │  h  │  t  │  s  │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │     |     |     │  p  │    │  r  │     |     |     │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **2.2. Optimally arrange the 10 remaining letters in the 10 remaining keys.**

  There are 3,628,800 permutations of 10 letters in 10 keys.

  Rerun the command in 1.2 above (after renaming the output folders): 
  ```bash
  bash run_jobs_local.sh
  ```

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │  b  |  f  │  o  │  u  │    │  l  │  d  │  m  │  v  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  i  │  e  │  a  │    │  h  │  t  │  s  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │  k  │  x  │  j  │  p  │    │  r  │  m  │  f  │  p  │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **2.3. Select the global Pareto front from the 24-letter/key layouts.**
  
  Rerun the commands in 1.3 above (after renaming the output files):
  ```bash
    python3 select_global_moo_solutions.py
    python3 analyze_global_moo_solutions.py output/global_moo_solutions.csv
  ```
