# README_keyboards

Example application: Optimize an English language keyboard layout

Repository: https://github.com/binarybottle/optimize_layouts.git  
Author: Arno Klein (arnoklein.info)  
License: MIT License (see LICENSE)

## Introduction
Let's apply the optimize_layouts software to the challenge of optimizing the 
arrangement of letters on a computer keyboard for typing (examples in English). 

For the following, we:
  - Assume bilateral symmetry in left- and right-hand ergonomics.
  - Focus on the 24 keys in, above, and below each home row ("home blocks")
    to reduce lateral stretching (the two middle columns are removed in figures):
    FDRSEVAWCQXZJKULIM;O,P./ (qwerty keys).
  - Focus on the 24 most frequent letters in English to assign to the 24 keys:
    etaoinsrhldcumfpgwybvkxj (q or z are dealt with at the end).

## Overview of steps
1. Optimally arrange the most frequent half of the letters in the alphabet
  to the 16 most preferred keys (as determined by the Bigram Typing Preference Study).
  (The remaining letters have negligible interaction with the top 13 letters,
  and the remaining keys have very low preference scores.)
  Use multi-objective optimization (MOO, with Engram scoring objectives -- see below) 
  to exhaustively search for a Pareto front of equally viable, 13-letter layout candidates.
    1. To parallelize the exhaustive search:
       - Constrain the four most frequent letters to the eight most preferred keys,
         with the most frequent letter constrained to one side (removes mirror layouts). 
       - Constrain the next four most frequent letters to the remaining of the top 16 keys.
       - Create a configuration file for each of the (840) possible arrangements of 4 letters. 
       ``generate_configs1.py``
    2. For each of the 840 configuration files, use MOO to optimally arrange 9 letters 
       in the 12 available of the top 16 keys. Each file entails 19,958,400 permutations
       (16,765,056,000 total permutations across all 840 configuration files),
       and the result is 840 Pareto fronts, each with many solutions.
       ``run_jobs.py``
    3. Select the global Pareto front of MOO solutions from the 13-letter layouts.
       ``layouts_consolidate.py``
2. Optimally arrange the remaining letters.
    1. For each selected 13-letter layout, generate a new configuration file.
       ``generate_configs2.py``
    2. For each configuration file, optimally arrange the 11 (out of 24) 
       remaining letters in the 11 remaining keys, again using MOO.
       ``run_jobs.py``
    3. Select the global Pareto front of MOO solutions from the 24-letter/key layouts.
       ``layouts_consolidate.py``
3. Select the final layout: ``layouts_filter.py``
4. Arrange periphery of the 24-letter/key home blocks.
    1. Assign the two least frequent letters (q & z in English) 
       to the two upper-right corner keys.
    2. Assign punctuation to the two middle columns between the home blocks. 

## Commands for steps 1 and 2.

### Step 1. Optimally arrange the 13 top letters in the 18 top keys.

  **1.1. Constrain the top 5 letters to the top 8 keys, with the topmost letter on the right side.**
  Command for generating the configuration files: `python generate_configs1.py`

  In the example below, the 4 most frequent letters (etao of etaoinsrhldcumfpgwybvkxj
  in English) are assigned to any 4 of the top 8 keys 
  (available keys are empty; restricted keys are represented by "|||||"):

    ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │|||||||||||│     │|||||│    │|||||│  o  │|||||│|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │|||||│     │  t  │     │    │     │  e  │  a  │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │|||||||||||||||||│|||||│    │|||||│|||||||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
    ```

  **1.2. Optimally arrange 9 letters in the 12 available of the top 16 keys.**
  Command for optimizing layouts, with constraints specified per configuration file: `bash run_jobs.sh`

    ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │||||||     │  p  │  u  │    │  l  │  o  │     │|||||│
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  s  │  t  │  h  │    │  i  │  e  │  a  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │||||||||||||     │     │    │  r  │     ||||||||||||│
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
    ```

  **1.3. Select the global Pareto front of 13-letter layouts.**
  `python layouts_consolidate.py`

### Step 2. Optimally arrange the remaining letters.

  **2.1. Optimally arrange the 11 remaining letters in the 11 remaining keys.**
  Rerun the commands above (after renaming variables and folders): 
  `python generate_configs2.py; bash run_jobs.sh`

    ```
    ╭───────────────────────╮    ╭───────────────────────╮
    │  b  |  f  │  p  │  u  │    │  l  │  d  │  r  │  v  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┼─────┤
    │  c  │  s  │  t  │  h  │    │  i  │  e  │  a  │  n  │
    ├─────┼─────┼─────┼─────┤    ├─────┼─────┼─────┤─────┤
    │  k  │  x  │  j  │  m  │    │  r  │  g  │  y  │  w  │
    ╰─────┴─────┴─────┴─────╯    ╰─────┴─────┴─────┴─────╯
    ```

  **2.2. Select the global Pareto front from the 24-letter/key layouts.**

  Rerun the commands in 1.3 above (after renaming the output files):
  `python optimize_layouts.py; python layouts_filter.py output/global_moo_solutions.csv`


## Scoring keyboard layouts
Keyboard layouts are defined by letters assigned to keys.
While letters and letter-pairs and their frequencies are language-dependent, 
keys and key-pairs are not. For each of the MOO objectives, 
an average base score is calculated over all possible key-pairs:

    1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    3. Same-row finger order and column separation
       - (empirical analysis of left key-pairs toward vs. away from the thumb)
       - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 
    4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns)
    5. Outside reach (empirical analysis of left-hand lateral stretches outside finger-columns)

For this study, #5 (engram_outside) is excluded, as only finger-column keys are considered.
A layout's score is the average product of each key-pair's base score 
and the corresponding letter-pair's frequency. 

Below is a code excerpt of the first four key-pair scoring criteria from 
https://github.com/binarybottle/keyboard_layout_scorers/blob/main/prep/prep_keypair_engram_scores.py
  
```python
    #----------------------------------------------------------------------------------
    # Engram's bigram scoring criteria
    #----------------------------------------------------------------------------------    
    # 1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    # 2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    # 3. Same-row finger order and column separation
    #    - (empirical analysis of left key-pairs toward vs. away from the thumb)
    #    - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 
    # 4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns)
    #----------------------------------------------------------------------------------    
   
    # 1. Key preferences (empirical Bradley-Terry tiers inside left finger-columns)
    #    0.137 - 1.000: keys inside the 8 finger-columns
    #    0.000: keys outside the 8 finger-columns 
    tier_values = {
        'F': 1.000, 'J': 1.000,
        'D': 0.870, 'K': 0.870,
        'E': 0.646, 'I': 0.646,
        'S': 0.646, 'L': 0.646,
        'V': 0.568, 'M': 0.568,
        'R': 0.568, 'U': 0.568,
        'W': 0.472, 'O': 0.472,
        'A': 0.410, ';': 0.410,
        'C': 0.410, ',': 0.410,
        'Z': 0.137, '/': 0.137,
        'Q': 0.137, 'P': 0.137,
        'X': 0.137, '.': 0.137
    }

    key_score = 0
    for key in [char1, char2]:
        key_score += tier_values.get(key, 0)  # Get tier value or 0 if not found

    scores['key_preference'] = key_score / 2.0  # Average over 2 keys

    # 2. Row separation (empirical meta-analysis of left same-row, reach, and hurdle key-pairs) 
    #    1.000: 2 hands
    #    1.000: 2 keys in the same row
    #    0.588: 2 keys in adjacent rows (reach)
    #    0.000: 2 keys straddling home row (hurdle)
    if hand1 != hand2:
        scores['row_separation'] = 1.0        # Two hands
    else:
        if row_gap == 0:
            scores['row_separation'] = 1.0    # Same row
        elif row_gap == 1:
            scores['row_separation'] = 0.588  # Adjacent row (reach)
        else:
            scores['row_separation'] = 0.0    # Skip row (hurdle)

    # 3. Same-row finger order and column separation
    #    - (empirical analysis of left key-pairs toward vs. away from the thumb)
    #    - (empirical meta-analysis of left key-pairs in adjacent vs. remote columns) 
    #    1.000: 2 hands
    #    1.000: adjacent columns, inward roll, in the same row
    #    0.779: adjacent columns, outward roll, in the same row
    #    0.750: remote columns, inward roll, in the same row
    #    0.779 x 0.750: remote columns, outward roll, in the same row
    #    0.500: different rows, different fingers
    #    0.000: same finger
    if hand1 != hand2:
        scores['same_row'] = 1.0        # Two hands
    elif finger1 == finger2:
        scores['same_row'] = 0.0        # Same finger
    elif row_gap == 0:  # Same row logic

        # Apply same-row finger order/direction (stronger effect)
        if finger2 > finger1:           # Inward
            scores['same_row'] = 1.0
        elif finger2 < finger1:         # Outward  
            scores['same_row'] = 0.779
        
        # Apply column separation penalty (weaker effect)
        if column_gap >= 2:             # Remote columns
            scores['same_row'] *= 0.750

    else:
        scores['same_row'] = 0.5        # Different rows, different fingers

    # 4. Same-finger (empirical analysis of left same- vs. different finger inside finger-columns)
    #    1.0: 2 hands
    #    1.0: 2 fingers
    #    0.0: 1 finger
    if hand1 != hand2:
        scores['same_finger'] = 1.0          # Two hands
    elif finger1 != finger2:
        scores['same_finger'] = 1.0          # Two fingers
    else:
        scores['same_finger'] = 0.0          # Same finger
```