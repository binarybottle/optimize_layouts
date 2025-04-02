# README_keyboards

# Optimizing an English keyboard layout
===================================================================
Let's apply this software to the challenge of optimizing the 
arrangement of letters on a computer keyboard for typing in English. 

For the following, we:
  - Assume bilateral symmetry in left- and right-hand ergonomics.
  - Focus on the 24 most frequent letters in English (not q or z).
  - Focus on the 24 keys in, above, and below the home row.
  - Refer to each key position by its capitalized Qwerty letter (J).
  - Refer to the "comfort" of a key based on typing research data.
  - Arrange letters in stages (more detail below):
    1. Restrict the most frequent letter to the most comfortable keys.
    2. Restrict the next-most frequent letters to a larger set of keys.
    3. Optimize the arrangement of remaining characters keys.

## Overview of Steps
1. Generate 720,720 configuration files corresponding to arrangements 
   of the 6 most frequent letters in the 16 most comfortable keys.
2. Optimally arrange 12 more letters to fill 18 keys for each configuration.
3. Generate a new configuration file from each optimal layout,
   removing letters from the 8 least comfortable keys.
4. Optimally arrange the 12 least frequent letters in remaining keys.
5. Select the highest scoring layout from the 720,720 complete layouts.

### Step 1. Generate 720,720 configuration files 

  ``python generate_configs1.py``

  ##### Assign the most frequent letter to the 2 most comfortable left keys.
  We will start by constraining the placement of the most frequent letter 
  ("e" in English, by a wide margin) to one of the 2 most comfortable keys.
  There are 2 permutations (1 letter in either of 2 keys): 

  ╭───────────────────────────────────────────────╮
  │     │     │     │     ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  D  │  F  ║     │     │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │     ║     │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

  #### Assign the next 5 letters to any available top-16 keys
  We then allow the next 5 letters (t, a, o, i, n) to be placed in any
  available key of the 16 most comfortable keys.
  There are 360,360 permutations (5 letters in any of 15 available keys):
    - 3,003 ways to choose 5 keys from 15 keys
    - 120 ways to arrange 5 letters in those 5 keys

  ╭───────────────────────────────────────────────╮
  │     │  W  │  E  │  R  ║  U  │  I  │  O  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  V  ║  M  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 2. Optimize layouts for each configuration file 

  ``python optimize_layout.py``

  #### Assign the next 12 letters to the 12 remaining top-18 keys
  For each of Step 1's 720,720 configuration files, a branch-and-bound 
  algorithm efficiently probes the 479 million (12!) possible permutations, 
  to optimally arrange the next 12 letters for that configuration.
  This results in 18 filled positions:

  ╭───────────────────────────────────────────────╮
  │     │  W  │  E  │  R  ║  U  │  I  │  O  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │  C  │  V  ║  M  │  ,  │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 3. Generate a second set of configuration files 

  ``python generate_configs2.py``

  Generate a new configuration file from each optimal layout
  from Step 2, removing letters from the 8 least comfortable keys.
  This will promote exploration of a wide range of permutations 
  for the remaining 12 of the 24 keys in the next step.

  ╭───────────────────────────────────────────────╮
  │     │     │  E  │  R  ║  U  │  I  │     │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │  S  │  D  │  F  ║  J  │  K  │  L  │     │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │     │     │     │  V  ║  M  │     │     │     │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 4. Optimally arrange the least frequent letters

  Per-config approach (default: 1 layout per config file):
  ``python generate_configs2.py --layouts-per-config 100``
  Across-all approach (top 1,000 layouts across all config files):
  ``python generate_configs2.py --top-across-all 1000``
  Both approaches together:
  ``python generate_configs2.py --layouts-per-config 100 --top-across-all 1000``

  We run optimize_layout.py again on each new unique configuration 
  file to optimally arrange the 12 least frequent letters in the 
  12 remaining least comfortable keys:

  ╭───────────────────────────────────────────────╮
  │  Q  │  W  │     │     ║     │     │  O  │  P  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  A  │     │     │     ║     │     │     │  ;  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  Z  │  X  │  C  │     ║     │  ,  │  .  │  /  │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯

### Step 5. Select the layout with the highest score

  Each of the 720,720 configuration files leads to an optimal layout 
  based on its initial constraints. A score is computed for each layout,
  based on the same scoring logic used to evaluate layouts during the
  optimization process. For this final step, we select the layout 
  with the highest score for a 24-key layout: 

  ╭───────────────────────────────────────────────╮
  │  Q  │  W  │  E  │  R  ║  U  │  I  │  O  │  P  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  A  │  S  │  D  │  F  ║  J  │  K  │  L  │  ;  │
  ├─────┼─────┼─────┼─────╫─────┼─────┼─────┼─────┤
  │  Z  │  X  │  C  │  V  ║  M  │  ,  │  .  │  /  │
  ╰─────┴─────┴─────┴─────╨─────┴─────┴─────┴─────╯
