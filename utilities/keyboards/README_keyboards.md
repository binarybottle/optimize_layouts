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
1. Optimally arrange the 14 most frequent letters in the 14 most comfortable keys.
  (The remaining 10 letters have negligible interaction with the top 14 letters,
  and the remaining 10 keys have much lower comfort scores than the top 14 keys.)
  Use multi-objective optimization (MOO, with Engram-7 scoring objectives) 
  to find a Pareto front of equally viable, 14-letter/key layout candidates.
    1. To parallelize the branch-and-bound, depth-first search:
       constrain the most frequent letter to the right side of the keyboard while 
       constraining the next 3 letters to the remaining 13 of the top 14 keys on both sides.
       Create a configuration file for each of the (12,012) possible arrangements of 4 in 14. 
       ``generate_configs1.py``
    2. For each of the 12,012 configuration files, use MOO to optimally arrange 10 letters 
       in the 10 available of 14 keys. The result is 12,012 Pareto fronts, each with many solutions.
       ``run_jobs.py``
    3. Select the global Pareto front of MOO solutions from the 14-letter/key layouts.
       ``select_global_moo_solutions.py``
2. Optimally arrange the remaining letters.
    1. For each selected 14-letter/key layout, generate a new configuration file.
       ``generate_configs2.py``
    2. For each configuration file, optimally arrange the 10 (out of 24) 
       remaining letters in the 10 remaining keys, again using MOO.
       ``run_jobs.py``
    3. Select the global Pareto front of MOO solutions from the 24-letter/key layouts.
       ``select_global_moo_solutions.py``
3. Select the final layout: ``analyze_global_moo_solutions.py``
    1. For each MOO objective, replace scores with their rankings.
    2. Sum the rankings for each layout.
    3. Sort layouts by these sums.
    4. Select layout(s) with the lowest sum (lowest is best).
4. Arrange periphery of the 24-letter/key home blocks.
    1. Assign the two least frequent letters (q & z in English) 
       to the two upper-right corner keys.
    2. Assign punctuation to the two middle columns between the home blocks. 

## Commands and visuals for steps 1 and 2.

### Step 1. Optimally arrange the 14 most frequent letters in the 14 most comfortable keys.

  **1.1. Constrain the top 4 letters to the top 14 keys, with the top letter on the right side.**

    Number of permutations per config file, with some of the 14 items fixed per file: 
        14 items (0 fixed):  87,178,291,200 permutations
        13 items (1 fixed):   6,227,020,800 permutations
        12 items (2 fixed):     479,001,600 permutations
        11 items (3 fixed):      39,916,800 permutations
        10 items (4 fixed):       3,628,800 permutations
    Number of permutations for 3 fixed items (after constraining the top item to one side):
        3 fixed in 13 positions:      1,716 permutations
    Number of configuration files:
        1,716 3-fixed-in-13 permutations x 7 1-fixed-in-7 permutations = 12,012 configuration files

  Command for generating the 12,012 configuration files:

  ```bash
    cd keyboards; python generate_configs1.py
  ```

  In the example below, the 4 most frequent letters (etao in English)
  are assigned to 4 of the 14 most comfortable keys in the home blocks
  (available keys are empty; restricted keys are represented by "|||||"):

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │|||||||||||│  o  │     │    │     │     │|||||│|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │     │     │  t  │  a  │    │  e  │     │     │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│     │    │     │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **1.2. Optimally arrange 10 letters in the 10 available of the top 14 keys.**

  Command for optimizing layouts, with constraints specified per configuration file:

  ```bash
    bash run_jobs.sh
  ```

  This optimally arranges 10 letters in 10 available keys 
  (3,628,800 permutations) for each of the 12,012 configurations above:

  ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │|||||||||||│  o  │  u  │    │  l  │  d  │|||||│|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  i  │  t  │  a  │    │  e  │  h  │  s  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│  m  │    │  r  │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
  ```

  **1.3. Select the global Pareto front of 14-letter/key layouts.**

  ```bash
    python select_global_moo_solutions.py
  ```

### Step 2. Optimally arrange the remaining letters.

  **2.1. Optimally arrange the 10 remaining letters in the 10 remaining keys.**

  Rerun the command in 1.2 above (after renaming the output folders): 
  ```bash
  bash run_jobs.sh
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

  **2.2. Select the global Pareto front from the 24-letter/key layouts.**

  Rerun the commands in 1.3 above (after renaming the output files):
  ```bash
    python select_global_moo_solutions.py
    python analyze_global_moo_solutions.py output/global_moo_solutions.csv
  ```
