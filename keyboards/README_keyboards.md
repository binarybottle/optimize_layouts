# README_keyboards

Example application: Optimize an English language keyboard layout

Repository: https://github.com/binarybottle/optimize_layouts.git  
Author: Arno Klein (arnoklein.info)  
License: MIT License (see LICENSE)

## Scripts in this folder

**Visualization:**
- `display_layout.py` - Visualize single 32-key layouts (ASCII or HTML output)
- `display_layouts.py` - Display multiple layouts from CSV (auto-detects format)
- `compare_layouts.py` - Compare layouts with parallel coordinates and heatmap plots

**Analysis:**
- `analyze_frequencies.py` - Analyze letter-bigram frequency relationships and cumulative coverage
- `plot_frequencies.py` - Plot letter-bigram statistics from CSV data

All scripts include `--help` for usage details and support multiple input formats.

Example display:

  ```
  poetry run python3 display_layout.py --letters "zplr  diwychts  aeomxvbn  ugkfqj" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./['"

  Letters → Qwerty keys:
  ZPLR  DIWYCHTS  AEOMXVBN  UGKFQJ
  QWERTYUIOPASDFGHJKL;ZXCVBNM,./['

  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │ Z │ P │ L │ R │   │   │ D │ I │ W │ Y │ Q │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  │ C │ H │ T │ S │   │   │ A │ E │ O │ M │ J │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤───┘
  │ X │ V │ B │ N │   │   │ U │ G │ K │ F │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘    
  ```
