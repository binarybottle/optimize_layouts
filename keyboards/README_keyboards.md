# README_keyboards
Optimize an English language keyboard layout
===============================================================================
https://github.com/binarybottle/optimize_layouts.git
Author: Arno Klein (arnoklein.info)
MIT License

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
    
    ╭───────────────────────╮    ╭───────────────────────╮
    │  10 │  7  │  3  │  6  │    │  6  │  3  │  7  │  10 │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  8  │  4  │  2  │  1  │    │  1  │  2  │  4  │  8  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │  11 │  12 │  9  │  5  │    │  5  │  9  │  12 │  11 │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯

## Overview of Steps
1. Optimally arrange the 16 most frequent letters in the 16 most comfortable keys.
    (The remaining 8 letters have negligible interaction with the top 16 letters,
    and the remaining 8 keys have much lower comfort scores than the top 16 keys.)
    Use multi-objective optimization (MOO, with item- and item-pair objectives) 
    to find a Pareto front of equally viable, 16-letter/key layout candidates.
  1. To parallelize the branch-and-bound, depth-first search, 
      fix the 4 most frequent letters in every possible arrangement (11,880) 
      within the 12 most comfortable keys (half of our 24 keys in the home blocks).
  2. For each 4-letter arrangement, use MOO to optimally arrange 12 letters 
      in the 12 available of the 16 keys.
  3. Select the highest-scoring 16-letter/key layouts.
    1. For each MOO objective, replace scores with their rankings.
    2. Sum the rankings for each layout.
    3. Sort layouts by these sums.
    4. Select layouts whose sums are less than ?????? 

2. Optimally arrange the remaining letters.
  1. Remove 2 letters from the 2 least comfortable of the 16 keys 
      in the selected layouts (to explore a broader solution space).
  2. For each resulting 14-letter/key layout, optimally arrange the 10 
      (out of 24) remaining letters in the 10 remaining keys.
  3. Select the highest-scoring 24-letter/key layouts.
    1. Repeat Step 2 for 24-letter/key layouts.
    2. Select the final layout by ?????? 

3. Arrange periphery of the 12-letter/key home blocks.
  1. Assign the two least frequent letters (q & z in English) 
      to the two upper-right corner keys.
  2. Assign punctuation to the two middle columns between the home blocks. 


### Step 1. Optimally arrange the 16 most frequent letters in the 16 most comfortable keys.

  ##### 1.1. Fix 4 of the letters in every possible arrangement within 12 keys.
  There are 11,880 permutations of 4 letters in 12 keys (see example below):
    - 1,365 ways to choose 4 keys from 12 keys
    - 24 ways to arrange 4 letters in those 4 keys

  Command for generating the 11,880 configuration files:

    ```bash
    cd keyboards; python generate_configs1.py
    ```

  In the example below, the 4 most frequent letters (etao in English)
  are assigned to 4 of the 12 most comfortable keys in the home blocks
  (available keys are empty; restricted keys are represented by "|||||"):

    ╭───────────────────────╮    ╭───────────────────────╮
    │||||||     │  o  │     │    │     │     │     │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │     │     │  e  │  a  │    │     │  t  │     │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│     │    │     │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯

##### 1.2. Optimally arrange 12 letters in the 12 available of the top 16 keys.

Command for optimizing layouts with constraints specified in a configuration file:

    ```bash
    python optimize_layout.py

    # Script to parallelize across the 11,880 configuration files:
    bash slurm_array_submit.sh --moo --rescan
    ```

  Following the example, 12 letters are optimally arranged in 12 available keys:

    ╭───────────────────────╮    ╭───────────────────────╮
    │||||||  f  │  o  │  u  │    │  l  │  d  │  m  │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  i  │  e  │  a  │    │  h  │  t  │  s  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│  p  │    │  r  │|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯


### Step 2. Optimally arrange the remaining letters.
The following steps act on each of Step 1's 11,880 configuration files. 

  ##### 2.1. Remove 2 letters from the 2 least comfortable of the 16 keys.

  Command for generating the second set of configuration files:

    ```bash
    cd keyboards; python generate_configs2.py
    ```

    ╭───────────────────────╮    ╭───────────────────────╮
    │     |  f  │  o  │  u  │    │  l  │  d  │  m  │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │     │  i  │  e  │  a  │    │  h  │  t  │  s  │     │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │     |     |     │  p  │    │  r  │     |     |     │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯

  ##### 2.2. Optimally arrange the 10 remaining letters in the 10 remaining keys.
  There are XXXX permutations of 10 letters in 10 keys:

  Run the same command as above: `python optimize_layout.py`

    ╭───────────────────────╮    ╭───────────────────────╮
    │  b  |  f  │  o  │  u  │    │  l  │  d  │  m  │  v  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  i  │  e  │  a  │    │  h  │  t  │  s  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │  k  │  x  │  j  │  p  │    │  r  │  m  │  f  │  p  │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯

### Step 3. Arrange periphery of the 12-letter/key home blocks.
We assign the two least frequent letters (q & z in English) 
to the two upper-right corner keys:

    ╭───────────────────────╮    ╭─────────────────────────────╮
    │  b  |  f  │  o  │  u  │    │  l  │  d  │  m  │  v  │  z  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┼─────┤
    │  c  │  i  │  e  │  a  │    │  h  │  t  │  s  │  n  │  q  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤─────╯
    │  k  │  x  │  j  │  p  │    │  r  │  m  │  f  │  p  │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯

Finally, we assign punctuation to the two middle columns between the home blocks. 


## Acknowledgments
NSF and the Pittsburgh Supercomputing Center (PSC) generously provided 
computing resources for a keyboard layout optimization study. 
The study used Bridges-2 at PSC through allocation MED250010 from the 
Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support 
(ACCESS) program, which is supported by National Science Foundation grants 
#2138259, #2138286, #2138307, #2137603, and #2138296. Citation:

  Brown, ST, Buitrago, P, Hanna, E, Sanielevici, S, Scibek, R, 
  Nystrom, NA (2021). Bridges-2: A Platform for Rapidly-Evolving 
  and Data Intensive Research. In Practice and Experience in 
  Advanced Research Computing (pp 1-4). doi:10.1145/3437359.3465593
