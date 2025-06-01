# README_keyboards
Optimize an English language keyboard layout
===============================================================================
https://github.com/binarybottle/optimize_layouts.git
Author: Arno Klein (arnoklein.info)
MIT License

## Introduction
Let's apply this software to the challenge of optimizing the 
arrangement of letters on a computer keyboard for typing in English. 

For the following, we:
  - Assume bilateral symmetry in left- and right-hand ergonomics.
  - Focus on the 24 keys in, above, and below the home row:
    FDRSEVAWCQXZJKULIM;O,P./ (qwerty keys, excluding ' and [).
  - Focus on the 24 most frequent letters in English to assign to the 24 keys:
    etaoinsrhldcumfpgwybvkxj (not q or z).
  - Refer to the "comfort" of a key based on typing preference research data,
    with resulting key comfort ranking 1 (high) to 12 (low) for each hand:
    ╭───────────────────────────────────────────────╮
    │  10 │  7  │  3  │  6  ║  6  │  3  │  7  │  10 │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  8  │  4  │  2  │  1  ║  1  │  2  │  4  │  8  │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  11 │  12 │  9  │  5  ║  5  │  9  │  12 │  11 │
    ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

## Overview of Steps
1. Optimally arrange the 16 most frequent letters in the 16 most comfortable keys.
   1. Create every possible arrangement (11,880) of the 4 most frequent letters 
      to the 12 most comfortable keys.
   2. For each 
   Generate 11,880 configuration files to set up letters and keys.
2. Optimally arrange frequent letters for each configuration. 
3. Generate a second set of configuration files.
4. Optimally arrange remaining letters.
5. Select the layout with the highest score.

### Step 1. Generate 11,880 configuration files 

  `cd keyboards; python generate_configs1.py`

  ##### Assign the 4 most frequent letters to the 12 most comfortable keys.
  We will start by allowing the 4 most frequent letters (etao in English) 
  to be assigned to any of the 12 most comfortable keys (half the 24 keys).
  There are 11,880 permutations:
    - 1,365 ways to choose 4 keys from 12 keys
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
  bash slurm_array_submit.sh --moo --rescan

  #### Assign the next 11 letters to the 11 remaining top-16 keys
  For each of Step 1's 65,520 configuration files, a branch-and-bound algorithm 
  optimally arranges the next set of letters for that configuration.
  A score is computed for each layout (see "Layout scoring" in README.md).

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

  (More than 11 letters exceeds time and memory limits for the 
  supercomputer center compute-hours -- see tests/README_tests.md.)

### Step 3. Generate a 2nd set of configuration files, removing letters 

  ```shell
  cd keyboards; python generate_keyboard_configs2.py
  python generate_keyboard_configs2.py # default: 1 layout per config file
  ```

  Generate a new configuration file from each optimal layout from Step 2, 
  removing the letters from the layout's least comfortable keys. 
  This will promote greater exploration in Step 4.
  
    ╭───────────────────────────────────────────────╮
    │  10 │  7  │  3  │  6  ║  6  │  3  │  7  │  10 │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  8  │  4  │  2  │  1  ║  1  │  2  │  4  │  8  │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │  11 │  12 │  9  │  5  ║  5  │  9  │  12 │  11 │
    ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  If we leave 14 letters, then we will need to fill 10 keys in Step 4:

    ╭───────────────────────────────────────────────╮
    │     │  f  │  o  │  u  ║  l  │  d  │  m  │     │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │     │  i  │  e  │  a  ║  h  │  t  │  s  │     │
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
    │     │     │  o  │     ║     │  d  │     │     │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │     │  i  │  e  │  a  ║  h  │  t  │  s  │     │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
    │     │     │     │  g  ║  r  │     │     │     │
    ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 4. Optimally arrange remaining letters

  `python optimize_layout.py`

  Run optimize_layout.py again on each new unique configuration file to 
  optimally arrange the remaining letters in the least comfortable keys
  -- in our example assigning 10 letters, ckxjwvnmfp

    ╭───────────────────────────────────────────────╮
    │  -  │  f  │  o  │  u  ║  l  │  d  │  m  │  -  │
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

  Finally, we select the layout with the highest score for a 24-key layout, 
  and add the 2 least frequent letters (q and z): 

    ╭─────────────────────────────────────────────────────╮
    │  b  │  y  │  o  │  u  ║  l  │  d  │  w  │  v  │  z  │ 
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤─────┤
    │  c  │  i  │  e  │  a  ║  h  │  t  │  s  │  n  │  q  │
    ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤─────╯
    │  k  │  x  │  j  │  g  ║  r  │  m  │  f  │  p  │
    ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

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
