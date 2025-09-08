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
    FDRSEVAWCQXZJKULIM;O,P./ (qwerty keys).
  - Focus on the 24 most frequent letters in English to assign to the 24 keys:
    etaoinsrhldcumfpgwybvkxj (q or z are dealt with at the end).

## Overview of steps
1. Optimally arrange the 14 most frequent letters in the 14 most comfortable keys.
  (The remaining 10 letters have negligible interaction with the top 14 letters,
  and the remaining 10 keys have much lower comfort scores than the top 14 keys.)
  Use multi-objective optimization (MOO, with Engram-6 scoring objectives -- see below) 
  to find a Pareto front of equally viable, 14-letter/key layout candidates.
    1. To parallelize the exhaustive search:
       constrain the most frequent letter to the two strongest keys (JK) while 
       constraining the next 3 letters to the remaining 13 of the top 14 keys.
       Create a configuration file for each of the (3,432) possible arrangements of 4 in 14. 
       ``generate_configs1.py``
    2. For each of the 3,432 configuration files, use MOO to optimally arrange 10 letters 
       in the 10 available of 14 keys. The result is 3,432 Pareto fronts, each with many solutions.
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

## Commands for steps 1 and 2.

### Step 1. Optimally arrange the 14 most frequent letters in the 14 most comfortable keys.

  **1.1. Constrain the top 4 letters to the top 14 keys, with the topmost letter on the right side.**
    - 14 keys are prioritized, based on statistical tests and cumulative coverage of 99% of bigrams by frequency.
    - 1 letter is assigned to any one of 7 of these keys on the right side of the keyboard.
    - 3 more letters are assigned to any of the remaining 13 of these 14 keys (1,716 permutations).
    - Each arrangement of these 4 letters is saved as a config file: 1,716 x 7 permutations = 3,432 config files.

  Command for generating the 3,432 configuration files: `python generate_configs1.py`

  In the example below, the 4 most frequent letters (etao in English)
  are assigned to 4 of the 14 keys in the home blocks
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

  Command for optimizing layouts, with constraints specified per configuration file: bash run_jobs.sh

  This optimally arranges 10 letters in 10 available keys 
  (3,628,800 permutations) for each of the 3,432 configurations above:

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

  `python select_global_moo_solutions.py`

### Step 2. Optimally arrange the remaining letters.

  **2.1. Optimally arrange the 10 remaining letters in the 10 remaining keys.**

  Rerun the commands above (after renaming variables and folders): 
  `python generate_configs2.py; bash run_jobs.sh`

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
  `python select_global_moo_solutions.py; python analyze_global_moo_solutions.py output/global_moo_solutions.csv`


## Scoring keyboard layouts
Keyboard layouts are defined by letters assigned to keys.
While letters and letter-pairs and their frequencies are language-dependent, 
keys, key-pairs, and key-triples are not. For each of the MOO objectives, 
an average base score is calculated over all possible key-pairs/triples:
  1.  Finger strength: Typing with the stronger two fingers
  2.  Finger stretch: Typing within the 8 finger columns (NOT USED IN THIS STUDY)
  3.  Finger curl: Typing within the 8 home keys, or preferred alternate keys
  4.  Row span: Same row, reaches, and hurdles
  5.  Column span: Adjacent columns in the same row
  6.  Finger order: Finger sequence toward the thumb

For this study, #2 is excluded, as only finger-column keys are considered,
and #6 has two variants: one for 2 keys, and the other for 3 key combinations.
A layout's score is the average product of each key-pair/triple's base score 
and the corresponding letter-pair/triple's frequency. 

Below is a code excerpt from https://github.com/binarybottle/keyboard_layout_scorers.git
that is responsible for precomputing scores for all possible key-pairs/triples.
  
1. Finger strength: Typing with the stronger two fingers
  - 1.0: 2 keys typed with strong fingers
  - 0.5: 1 key typed with 1 strong finger
  - 0.0: 0 keys typed with strong finger
  ```python
  strong_count = sum(1 for finger in [finger1, finger2] if finger in STRONG_FINGERS)
  if strong_count == 2:
      scores['strength'] = 1.0      # 2 keys typed with strong fingers
  elif strong_count == 1:
      scores['strength'] = 0.5      # 1 key typed with 1 strong finger
  else:
      scores['strength'] = 0.0      # 0 keys typed with strong finger
  ```

3. Finger curl: Typing within the 8 home keys, or preferred alternate keys
   above/below the home keys: 
   fingers 1,4 prefer row 3; finger 3 prefers row 1; finger 2 no preference
   For each key:
  -  1.0: home key
  -  0.5: alternate key
  -  0.0: any other key (unpreferred, no preference, or stretch)
  ```python
  curl_score = 0
  if homekey1 == 1:
      curl_score += 1
  else:
      if in_column1:
          # UPPER_FINGERS prefer the upper row 1; LOWER_FINGERS prefer lower row 3
          if finger1 in UPPER_FINGERS and row1 == 1 or finger1 in LOWER_FINGERS and row1 == 3:
              curl_score = 0.5
  if homekey2 == 1:
      curl_score += 1
  else:
      if in_column2:
          # UPPER_FINGERS prefer the upper row 1; LOWER_FINGERS prefer lower row 3
          if finger2 in UPPER_FINGERS and row2 == 1 or finger2 in LOWER_FINGERS and row2 == 3:
              curl_score = 0.5
  scores['curl'] = curl_score / 2.0
  ```

4. Row span: Same row, reaches, and hurdles 
  - 1.0: 2 keys in the same row (or 2 hands)
  - 0.5: 2 keys in adjacent rows (reach)
  - 0.0: 2 keys straddling home row (hurdle)
  ```python
  if hand1 != hand2:
      scores['rows'] = 1.0          # opposite hands always score well
  else:
      if row1 == row2:
          scores['rows'] = 1.0      # 2 keys in the same row
      elif abs(row1 - row2) == 1:
          scores['rows'] = 0.5      # 2 keys in adjacent rows (reach)
      else:
          scores['rows'] = 0.0      # 2 keys straddling home row (hurdle)
  ```

5. Column span: Adjacent columns in the same row
  - 1.0: adjacent columns in same row, or non-adjacent columns in different rows (or 2 hands)
  - 0.5: non-adjacent columns in the same row, or adjacent columns in different rows
  - 0.0: same finger
  ```python
  if hand1 != hand2:
      scores['columns'] = 1.0          # opposite hands always score well
  elif finger1 != finger2:
      column_gap = abs(column1 - column2)
      finger_gap = abs(finger1 - finger2)
      if (column_gap == 1 and row1 == row2) or (column_gap > 1 and finger_gap > 1 and row1 != row2):
          scores['columns'] = 1.0      # adjacent columns, same row / non-adjacent, different rows
      else:
          scores['columns'] = 0.5      # non-adjacent columns, same row / adjacent, different rows
  else:
      scores['columns'] = 0.0          # same finger
  ```

6. Finger order: Finger sequence toward the thumb
  - 1.0: inward roll on the same row (or 2 hands)
  - 0.5: outward roll, or inward roll on different rows
  - 0.0: same finger
  ```python
  if hand1 != hand2:
      scores['order'] = 1.0       # opposite hands always score well
  elif finger1 == finger2:
      scores['order'] = 0.0       # same finger scores zero
  elif row1 == row2:
      # Inward roll: toward thumb (finger number increases: 1→2→3→4)
      # Both hands are bilaterally symmetric, so same sequence logic
      scores['order'] = 1.0 if finger1 < finger2 else 0.5
  else:
      scores['order'] = 0.5
  ```

## Engram-6 3-key scoring criterion

  6. Finger order: Finger sequence toward the thumb
    - 1.0: inward roll
    - 0.5: outward roll
    - 0.0: mixed roll, or same finger
  ```python
  scores['order'] = 0.0  # Default for mixed patterns or unhandled cases
  if finger1 == finger2 == finger3:
      scores['order'] = 0.0          # same finger scores zero
  elif hand1 != hand2 and hand1 == hand3:
      scores['order'] = 1.0          # alternating hands
  elif hand1 == hand2 == hand3:
      if finger1 < finger2 < finger3:
          scores['order'] = 1.0      # inward roll
      elif finger1 > finger2 > finger3:
          scores['order'] = 0.5      # outward roll
  elif hand1 == hand2 and hand2 != hand3:
      if finger1 < finger2:
          scores['order'] = 1.0      # inward roll
      elif finger1 > finger2:
          scores['order'] = 0.5      # outward roll
  elif hand1 != hand2 and hand2 == hand3:
      if finger2 < finger3:
          scores['order'] = 1.0      # inward roll
      elif finger2 > finger3:
          scores['order'] = 0.5      # outward roll
  ```