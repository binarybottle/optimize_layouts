# README_keyboards
Optimize an English keyboard layout
===============================================================================
https://github.com/binarybottle/optimize_layouts.git
Author: Arno Klein (arnoklein.info)
License: MIT License (see LICENSE)

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

## Introduction
Let's apply this software to the challenge of optimizing the 
arrangement of letters on a computer keyboard for typing in English. 

For the following, we:
  - Assume bilateral symmetry in left- and right-hand ergonomics.
  - Focus on the 24 most frequent letters in English:
    etaoinsrhldcumfpgwybvkxj (not q or z)
  - Focus on the 24 keys in, above, and below the home row.
  - Arrange letters in stages (more detail below).
  - Refer to the "comfort" of a key based on typing research data.
    The rank order of estimated key comfort is:
    ╭───────────────────────────────────────────────╮
    │  10 │  8  │  5  │  3  ║  3  │  5  │  8  │  10 │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  7  │  4  │  2  │  1  ║  1  │  2  │  4  │  7  │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  12 │  11 │  9  │  6  ║  6  │  9  │  11 │  12 │
    ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

## Overview of Steps
1. Generate 65,520 configuration files.
2. Optimally arrange frequent letters for each configuration. 
3. Generate a second set of configuration files, removing letters.
4. Optimally arrange remaining letters.
5. Select the layout with the highest score.

### Step 1. Generate 65,520 configuration files 

  `cd keyboards; python generate_keyboard_configs1.py`

  ##### Assign the most frequent letter to the 2 most comfortable left keys.
  We will start by constraining the placement of the most frequent letter 
  (`e` in English, by a wide margin) to one of the 2 most comfortable keys
  (again, by a considerable margin, according to typing data). 
  There are 2 permutations (1 letter in either of 2 keys): 
  ╭───────────────────────────────────────────────╮
  │     │     │     │     ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  -  │  -  ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │     ║     │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  We'll choose one key for our example going forward:
  ╭───────────────────────────────────────────────╮
  │     │     │     │     ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  e  │     ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │     ║     │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  #### Assign the next 4 letters to any available top-16 keys
  We then allow the next 4 letters (`taoi` in English) 
  to be placed in any available of the 16 most comfortable keys.
  There are 32,760 permutations:
    - 1,365 ways to choose 4 keys from 15 keys
    - 24 ways to arrange 4 letters in those 4 keys

  ╭───────────────────────────────────────────────╮
  │     │  -  │  -  │  -  ║  -  │  -  │  -  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  -  │  -  │  e  │  -  ║  -  │  -  │  -  │  -  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  -  ║  -  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
  ╭───────────────────────────────────────────────╮
  │     │     │  o  │     ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │  i  │  e  │  a  ║     │  t  │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │     ║     │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 2. Optimally arrange frequent letters for each configuration

  `python optimize_layout.py`

  #### Assign the next 11 letters to the 11 remaining top-16 keys
  For each of Step 1's 65,520 configuration files, a branch-and-bound algorithm 
  optimally arranges the next set of letters for that configuration. 

  If we choose 11 letters (`nsrhldcumfp`), then there are more than
  39.9 million (11!) possible permutations, resulting in 16 filled positions:

  ╭───────────────────────────────────────────────╮
  │     │  -  │  o  │  -  ║  -  │  -  │  -  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  -  │  i  │  e  │  a  ║  -  │  t  │  -  │  -  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  -  ║  -  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
  ╭───────────────────────────────────────────────╮
  │     │  f  │  o  │  u  ║  l  │  d  │  m  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  p  ║  r  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  If we choose 13 letters (`nsrhldcumfpgw`), then there are more than
  6.2 billion (13!) possible permutations, resulting in 18 filled positions:

  ╭───────────────────────────────────────────────╮
  │     │  -  │  o  │  -  ║  -  │  -  │  -  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  -  │  i  │  e  │  a  ║  -  │  t  │  -  │  -  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  -  │  -  ║  -  │  -  │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
  ╭───────────────────────────────────────────────╮
  │     │  f  │  o  │  u  ║  l  │  d  │  w  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  p  │  g  ║  r  │  m  │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 3. Generate a 2nd set of configuration files, removing letters 

  ```shell
  cd keyboards; python generate_keyboard_configs2.py
  # Per-config approach (default: 1 layout per config file):
  python generate_configs2.py --layouts-per-config 100
  # Across-all approach (top 1,000 layouts across all config files):
  python generate_configs2.py --top-across-all 1000
  # Both approaches together:
  python generate_configs2.py --layouts-per-config 100 --top-across-all 1000
  ```

  Generate a new configuration file from each optimal layout from Step 2, 
  removing the letters from the layout's least comfortable keys. 
  This will promote greater exploration in Step 4.

  If we leave 14 letters, then we will need to fill 10 keys in Step 4:

  ╭───────────────────────────────────────────────╮
  │     │     │  o  │  u  ║  l  │  d  │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  g  ║  r  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  If we leave 12 letters, then we will need to fill 12 keys in Step 4:

  ╭───────────────────────────────────────────────╮
  │     │     │  o  │  u  ║  l  │  d  │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │  i  │  e  │  a  ║  h  │  t  │  s  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  g  ║  r  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

If we leave 10 letters, then we will need to fill 14 keys in Step 4:

  ╭───────────────────────────────────────────────╮
  │     │     │  o  │  u  ║  l  │  d  │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │  i  │  e  │  a  ║  h  │  t  │  s  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │     ║     │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯


### Step 4. Optimally arrange remaining letters

  ```shell
  # Per-config approach (default: 1 layout per config file):
  python generate_configs2.py --layouts-per-config 100
  # Across-all approach (top 1,000 layouts across all config files):
  python generate_configs2.py --top-across-all 1000
  # Both approaches together:
  python generate_configs2.py --layouts-per-config 100 --top-across-all 1000
  ```

  We run optimize_layout.py again on each new unique configuration file to 
  optimally arrange the remaining letters in the least comfortable keys
  (in our example assigning 12 letters, byckxjwvnmfp):

  ╭───────────────────────────────────────────────╮
  │  -  │  -  │  o  │  u  ║  l  │  d  │  -  │  -  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  -  │  i  │  e  │  a  ║  h  │  t  │  s  │  -  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  -  │  -  │  -  │  g  ║  r  │  -  │  -  │  -  │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  ╭───────────────────────────────────────────────╮
  │  b  │  y  │  o  │  u  ║  l  │  d  │  w  │  v  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  k  │  x  │  j  │  g  ║  r  │  m  │  f  │  p  │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯


### Step 5. Select the layout with the highest score

  Each of the 65,520 configuration files leads to an optimal layout 
  based on its initial constraints. A score is computed for each layout,
  based on the same scoring logic used to evaluate layouts during the
  optimization process. For this final step, we select the layout 
  with the highest score for a 24-key layout, 
  and add the 2 least frequent letters q and z: 

  ╭─────────────────────────────────────────────────────╮
  │  b  │  y  │  o  │  u  ║  l  │  d  │  w  │  v  │  z  │ 
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤─────┤
  │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │  q  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤─────╯
  │  k  │  x  │  j  │  g  ║  r  │  m  │  f  │  p  │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
