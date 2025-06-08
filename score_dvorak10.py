# score_dvorak10.py
"""
Dvorak-10 scoring model implementation for keyboard layout evaluation.

This script implements the 10 evaluation criteria derived from Dvorak's work:

2-HAND CRITERIA (apply to all digraphs):
1. Use both hands equally
2. Alternate between hands  

SAME-HAND CRITERIA (apply only to same-hand digraphs):
3. Don't use the same finger (two fingers)
4. Use non-adjacent fingers (skip fingers)
5. Stay within the home block (home block)
6. Don't skip over the home row (don't skip home)
7. Stay in the same row (same row)
8. Use the home row (include home)
9. Strum inward (roll in, not out)
10. Use strong fingers

Example usage:
    python score_dvorak10.py --items "qwertyuiopasdfghjkl;zxcvbnm,./" --positions "QWERTYUIOPASDFGHJKL;ZXCVBNM,./"
    python score_dvorak10.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW"
    python score_dvorak10.py --items "abc" --positions "FDJ" --details
"""

import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set

# QWERTY keyboard layout definitions with corrected finger numbering
QWERTY_LAYOUT = {
    # Row 1 (top) - finger numbering: 1=index, 2=middle, 3=ring, 4=pinky
    'Q': (1, 4, 'L'), 'W': (1, 3, 'L'), 'E': (1, 2, 'L'), 'R': (1, 1, 'L'), 'T': (1, 1, 'L'),
    'Y': (1, 1, 'R'), 'U': (1, 1, 'R'), 'I': (1, 2, 'R'), 'O': (1, 3, 'R'), 'P': (1, 4, 'R'),
    # Row 2 (home)
    'A': (2, 4, 'L'), 'S': (2, 3, 'L'), 'D': (2, 2, 'L'), 'F': (2, 1, 'L'), 'G': (2, 1, 'L'),
    'H': (2, 1, 'R'), 'J': (2, 1, 'R'), 'K': (2, 2, 'R'), 'L': (2, 3, 'R'), ';': (2, 4, 'R'),
    # Row 3 (bottom)
    'Z': (3, 4, 'L'), 'X': (3, 3, 'L'), 'C': (3, 2, 'L'), 'V': (3, 1, 'L'), 'B': (3, 1, 'L'),
    'N': (3, 1, 'R'), 'M': (3, 1, 'R'), ',': (3, 2, 'R'), '.': (3, 3, 'R'), '/': (3, 4, 'R'),
    # Additional common keys
    '1': (0, 4, 'L'), '2': (0, 3, 'L'), '3': (0, 2, 'L'), '4': (0, 1, 'L'), '5': (0, 1, 'L'),
    '6': (0, 1, 'R'), '7': (0, 1, 'R'), '8': (0, 2, 'R'), '9': (0, 3, 'R'), '0': (0, 4, 'R'),
    # Common punctuation
    "'": (2, 4, 'R'), '[': (1, 4, 'R'), ']': (1, 4, 'R'), '\\': (1, 4, 'R'),
    '-': (0, 4, 'R'), '=': (0, 4, 'R'),
}

# Define home row and home block (24 keys, excluding middle columns)
HOME_ROW = 2
HOME_BLOCK_KEYS = {
    # Left home block (12 keys) - excludes T, G, B
    'q', 'w', 'e', 'r',          # Top left
    'a', 's', 'd', 'f',          # Home left  
    'z', 'x', 'c', 'v',          # Bottom left
    # Right home block (12 keys) - excludes Y, H, N
    'u', 'i', 'o', 'p',          # Top right
    'j', 'k', 'l', ';',          # Home right
    'm', ',', '.', '/',          # Bottom right
}

# Finger strength classification (1=index, 2=middle, 3=ring, 4=pinky)
STRONG_FINGERS = {1, 2}  # Index and middle fingers
WEAK_FINGERS = {3, 4}    # Ring and pinky fingers

def get_key_info(key: str) -> Tuple[int, int, str]:
    """Get (row, finger, hand) for a key."""
    key = key.upper()
    if key in QWERTY_LAYOUT:
        return QWERTY_LAYOUT[key]
    else:
        # Default for unknown keys - assume right hand, index finger, home row
        return (2, 4, 'R')

def is_adjacent_finger(finger1: int, finger2: int, hand1: str, hand2: str) -> bool:
    """Check if two fingers are adjacent (same hand only)."""
    if hand1 != hand2:
        return False
    return abs(finger1 - finger2) == 1

def get_roll_direction(finger1: int, finger2: int, hand1: str, hand2: str) -> str:
    """Determine if sequence rolls inward or outward."""
    if hand1 != hand2:
        return 'alternating'
    
    if finger1 == finger2:
        return 'same'
    
    # Inward roll: from outer (pinky=4) to inner (index=1)
    if finger1 > finger2:
        return 'inward'
    else:
        return 'outward'

def same_hand(pos1: str, pos2: str) -> bool:
    """Check if both positions use the same hand."""
    _, _, hand1 = get_key_info(pos1)
    _, _, hand2 = get_key_info(pos2)
    return hand1 == hand2

def get_finger_from_pos(pos: str) -> int:
    """Get finger number from QWERTY position."""
    _, finger, _ = get_key_info(pos)
    return finger

def get_hand_from_pos(pos: str) -> str:
    """Get hand from QWERTY position."""
    _, _, hand = get_key_info(pos)
    return hand

class Dvorak10Scorer:
    """Implements the Dvorak-10 scoring model."""
    
    def __init__(self, layout_mapping: Dict[str, str], text: str):
        """
        Initialize scorer with layout mapping and optional text.
        
        Args:
            layout_mapping: Dict mapping items to positions (e.g., {'a': 'F', 'b': 'D'})
            text: Optional text to analyze (if None, uses layout_mapping keys)
        """
        self.layout_mapping = layout_mapping
        self.text = text
        
        # Flow-based digraphs (for criteria 1-2) - ignores word boundaries
        self.digraphs = self._extract_digraphs()
        
        # Word-aware digraphs (for criteria 3-10) - respects word boundaries  
        self.word_digraphs = self._extract_word_digraphs()

    def _extract_digraphs(self) -> List[Tuple[str, str]]:
        """Extract digraphs for typing flow analysis (criteria 1-2) - ignores word boundaries."""
        digraphs = []
        
        # Convert text to lowercase and filter to only characters in layout_mapping
        text_lower = self.text.lower()
        filtered_chars = [char for char in text_lower if char in self.layout_mapping]
        
        # Create digraphs from consecutive characters (ignoring spaces)
        for i in range(len(filtered_chars) - 1):
            char1, char2 = filtered_chars[i], filtered_chars[i + 1]
            digraphs.append((char1, char2))
        
        return digraphs

    def _extract_word_digraphs(self) -> List[Tuple[str, str]]:
        """Extract digraphs for finger coordination analysis (criteria 3-10) - respects word boundaries."""
        import re
        
        digraphs = []
        
        # Split text into words (separated by spaces or other word-breaking punctuation)
        words = re.split(r'[\s\n\r\t]+', self.text.lower())
        
        for word in words:
            if not word:  # Skip empty strings
                continue
                
            # Filter characters to only those in our layout mapping
            filtered_chars = [char for char in word if char in self.layout_mapping]
            
            # Create digraphs within this word only
            for i in range(len(filtered_chars) - 1):
                char1, char2 = filtered_chars[i], filtered_chars[i + 1]
                digraphs.append((char1, char2))
        
        return digraphs

    def get_key_counts(self) -> Dict[str, int]:
        """Get frequency count of each character in layout_mapping."""
        counts = defaultdict(int)
        
        # Convert text to lowercase and count only characters in layout_mapping
        text_lower = self.text.lower()
        
        for char in text_lower:
            if char in self.layout_mapping:
                counts[char] += 1
        
        return dict(counts)
    
    def score_1_hand_balance(self) -> Tuple[float, Dict]:
        """Criterion 1: Use both hands equally."""
        key_counts = self.get_key_counts()
        left_count = right_count = 0
        
        details = {'left_keys': [], 'right_keys': [], 'left_count': 0, 'right_count': 0}
        
        for char, count in key_counts.items():
            pos = self.layout_mapping[char]
            _, _, hand = get_key_info(pos)
            
            if hand == 'L':
                left_count += count
                details['left_keys'].append((char, pos, count))
            else:
                right_count += count
                details['right_keys'].append((char, pos, count))
        
        total = left_count + right_count
        if total == 0:
            return 1.0, details
        
        details['left_count'] = left_count
        details['right_count'] = right_count
        details['total_count'] = total
        
        # Score: 1 - 2 * |L - R| / (L + R)
        imbalance = abs(left_count - right_count) / total
        score = 1 - 2 * imbalance
        
        return max(0.0, score), details
    
    def score_2_hand_alternation(self) -> Tuple[float, Dict]:
        """Criterion 2: Alternate between hands."""
        if not self.digraphs:
            return 1.0, {'same_hand': 0, 'different_hand': 0, 'digraphs': []}
        
        same_hand = different_hand = 0
        details = {'same_hand_digraphs': [], 'different_hand_digraphs': []}
        
        for char1, char2 in self.digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            _, _, hand1 = get_key_info(pos1)
            _, _, hand2 = get_key_info(pos2)
            
            if hand1 == hand2:
                same_hand += 1
                details['same_hand_digraphs'].append((char1, char2, pos1, pos2))
            else:
                different_hand += 1
                details['different_hand_digraphs'].append((char1, char2, pos1, pos2))
        
        total = same_hand + different_hand
        details['same_hand'] = same_hand
        details['different_hand'] = different_hand
        details['total_digraphs'] = total
        
        if total == 0:
            return 1.0, details
        
        # Score: 1 - S/(D + S)
        score = 1 - (same_hand / total)
        return score, details
    
    def score_3_different_fingers(self) -> Tuple[float, Dict]:
        """Criterion 3: Don't use the same finger."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'same_finger': 0, 'different_finger': 0, 'digraphs': []}
        
        same_finger = different_finger = 0
        details = {'same_finger_digraphs': [], 'different_finger_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            _, finger1, _ = get_key_info(pos1)
            _, finger2, _ = get_key_info(pos2)
            
            if finger1 == finger2:
                same_finger += 1
                details['same_finger_digraphs'].append((char1, char2, pos1, pos2, finger1))
            else:
                different_finger += 1
                details['different_finger_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
        
        total = same_finger + different_finger
        details['same_finger'] = same_finger
        details['different_finger'] = different_finger
        details['total_same_hand'] = total
        
        if total == 0:
            return 1.0, details
        
        # Score: 1 - S/(D + S)
        score = 1 - (same_finger / total)
        return score, details
    
    def score_4_non_adjacent_fingers(self) -> Tuple[float, Dict]:
        """Criterion 4: Use non-adjacent fingers (same-hand digraphs only)."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'adjacent': 0, 'non_adjacent': 0, 'same_finger': 0, 'digraphs': []}
        
        adjacent = non_adjacent = same_finger_count = 0
        details = {'adjacent_digraphs': [], 'non_adjacent_digraphs': [], 'same_finger_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1 = self.layout_mapping[char1]
            pos2 = self.layout_mapping[char2]
            
            row1, finger1, hand1 = get_key_info(pos1)
            row2, finger2, hand2 = get_key_info(pos2)
            
            if finger1 == finger2:
                same_finger_count += 1
                details['same_finger_digraphs'].append((char1, char2, pos1, pos2, finger1))
            elif is_adjacent_finger(finger1, finger2, hand1, hand2):
                adjacent += 1
                details['adjacent_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
            else:
                non_adjacent += 1
                details['non_adjacent_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
        
        # Only count different-finger digraphs for this score
        different_finger_total = adjacent + non_adjacent
        details['adjacent'] = adjacent
        details['non_adjacent'] = non_adjacent
        details['same_finger'] = same_finger_count
        details['total_same_hand'] = len(same_hand_digraphs)
        details['total_different_finger'] = different_finger_total
        
        if different_finger_total == 0:
            return 0.0, details
        
        # Score: 1 - A/(A + !A) among different-finger digraphs only
        score = 1 - (adjacent / different_finger_total)
        return score, details

    def score_5_home_block(self) -> Tuple[float, Dict]:
        """Criterion 5: Stay within the home block (same-hand digraphs only)."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'outside_home_block': 0, 'inside_home_block': 0, 'digraphs': []}
        
        outside = inside = 0
        details = {'outside_digraphs': [], 'inside_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            
            # Check if both positions are in home block (case-insensitive)
            pos1_in_home = pos1.lower() in HOME_BLOCK_KEYS
            pos2_in_home = pos2.lower() in HOME_BLOCK_KEYS
            
            if not pos1_in_home or not pos2_in_home:
                outside += 1
                details['outside_digraphs'].append((char1, char2, pos1, pos2, pos1_in_home, pos2_in_home))
            else:
                inside += 1
                details['inside_digraphs'].append((char1, char2, pos1, pos2))
        
        total = outside + inside
        details['outside_home_block'] = outside
        details['inside_home_block'] = inside
        details['total_same_hand'] = total
        
        if total == 0:
            return 1.0, details
        
        # Score: 1 - O/(O + I)
        score = 1 - (outside / total)
        return score, details

    def score_6_dont_skip_home(self) -> Tuple[float, Dict]:
        """Criterion 6: Don't skip over the home row."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'jump_home': 0, 'dont_jump_home': 0, 'digraphs': []}
        
        jump = dont_jump = 0
        details = {'jump_digraphs': [], 'no_jump_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            row1, _, _ = get_key_info(pos1)
            row2, _, _ = get_key_info(pos2)
            
            # Check if we skip over home row (row 2)
            if (row1 == 1 and row2 == 3) or (row1 == 3 and row2 == 1):
                jump += 1
                details['jump_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
            else:
                dont_jump += 1
                details['no_jump_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
        
        total = jump + dont_jump
        details['jump_home'] = jump
        details['dont_jump_home'] = dont_jump
        details['total_same_hand'] = total
        
        if total == 0:
            return 1.0, details
        
        # Score: 1 - J/(J + !J)
        score = 1 - (jump / total)
        return score, details
    
    def score_7_same_row(self) -> Tuple[float, Dict]:
        """Criterion 7: Stay in the same row."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'different_row': 0, 'same_row': 0, 'digraphs': []}

        different_row = same_row = 0
        details = {'different_row_digraphs': [], 'same_row_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            row1, _, _ = get_key_info(pos1)
            row2, _, _ = get_key_info(pos2)
            
            if row1 != row2:
                different_row += 1
                details['different_row_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
            else:
                same_row += 1
                details['same_row_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
        
        total = different_row + same_row
        details['different_row'] = different_row
        details['same_row'] = same_row
        details['total_same_hand'] = total
        
        if total == 0:
            return 1.0, details
        
        # Score: 1 - D/(D + S)
        score = 1 - (different_row / total)
        return score, details
    
    def score_8_include_home(self) -> Tuple[float, Dict]:
        """Criterion 8: Use the home row (same-hand digraphs only)."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'outside_home_row': 0, 'include_home_row': 0, 'digraphs': []}

        outside = include = 0
        details = {'outside_digraphs': [], 'include_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            row1, _, _ = get_key_info(pos1)
            row2, _, _ = get_key_info(pos2)
            
            if row1 != HOME_ROW and row2 != HOME_ROW:
                outside += 1
                details['outside_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
            else:
                include += 1
                details['include_digraphs'].append((char1, char2, pos1, pos2, row1, row2))
        
        total = outside + include
        details['outside_home_row'] = outside
        details['include_home_row'] = include
        details['total_same_hand'] = total
        
        # Score: 1 - O/(O + I)
        score = 1 - (outside / total)
        return score, details
    
    def score_9_roll_inward(self) -> Tuple[float, Dict]:
        """Criterion 9: Strum inward (roll in, not out) - same-hand digraphs only."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'outward': 0, 'inward': 0, 'same_finger': 0, 'digraphs': []}

        outward = inward = same_finger_count = 0
        details = {'outward_digraphs': [], 'inward_digraphs': [], 'same_finger_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1 = self.layout_mapping[char1]
            pos2 = self.layout_mapping[char2]
            
            row1, finger1, hand1 = get_key_info(pos1)
            row2, finger2, hand2 = get_key_info(pos2)
            
            roll_dir = get_roll_direction(finger1, finger2, hand1, hand2)
            
            if roll_dir == 'outward':
                outward += 1
                details['outward_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
            elif roll_dir == 'inward':
                inward += 1
                details['inward_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
            else:  # same finger
                same_finger_count += 1
                details['same_finger_digraphs'].append((char1, char2, pos1, pos2, finger1))
        
        # Only count different-finger digraphs for this score
        different_finger_total = outward + inward
        details['outward'] = outward
        details['inward'] = inward
        details['same_finger'] = same_finger_count
        details['total_same_hand'] = len(same_hand_digraphs)
        details['total_different_finger'] = different_finger_total
        
        if different_finger_total == 0:
            return 0.0, details
        
        # Score: 1 - O/(O + I) among different-finger digraphs only
        score = 1 - (outward / different_finger_total)
        return score, details

    def score_10_strong_fingers(self) -> Tuple[float, Dict]:
        """Criterion 10: Use strong fingers (same-hand digraphs only)."""
        same_hand_digraphs = [
            (c1, c2) for c1, c2 in self.word_digraphs  # Use word_digraphs
            if get_key_info(self.layout_mapping[c1])[2] == get_key_info(self.layout_mapping[c2])[2]
        ]
        
        if not same_hand_digraphs:
            return 0.0, {'weak_finger': 0, 'strong_finger': 0, 'digraphs': []}

        weak = strong = 0
        details = {'weak_digraphs': [], 'strong_digraphs': []}
        
        for char1, char2 in same_hand_digraphs:
            pos1, pos2 = self.layout_mapping[char1], self.layout_mapping[char2]
            _, finger1, _ = get_key_info(pos1)
            _, finger2, _ = get_key_info(pos2)
            
            # Digraph involves weak finger if either finger is weak
            if finger1 in WEAK_FINGERS or finger2 in WEAK_FINGERS:
                weak += 1
                details['weak_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
            else:
                strong += 1
                details['strong_digraphs'].append((char1, char2, pos1, pos2, finger1, finger2))
        
        total = weak + strong
        details['weak_finger'] = weak
        details['strong_finger'] = strong
        details['total_same_hand'] = total
        
        # Score: 1 - W/(W + !W)
        score = 1 - (weak / total)
        return score, details
    
    def calculate_all_scores(self) -> Dict:
        """Calculate all 10 Dvorak scores."""
        scores = {}
        
        score_functions = [
            ('1_hand_balance', self.score_1_hand_balance),
            ('2_hand_alternation', self.score_2_hand_alternation),
            ('3_different_fingers', self.score_3_different_fingers),
            ('4_non_adjacent_fingers', self.score_4_non_adjacent_fingers),
            ('5_home_block', self.score_5_home_block),
            ('6_dont_skip_home', self.score_6_dont_skip_home),
            ('7_same_row', self.score_7_same_row),
            ('8_include_home', self.score_8_include_home),
            ('9_roll_inward', self.score_9_roll_inward),
            ('10_strong_fingers', self.score_10_strong_fingers),
        ]
        
        for name, func in score_functions:
            score, details = func()
            scores[name] = {'score': score, 'details': details}
        
        # Calculate total score (equal weights)
        total_score = sum(scores[name]['score'] for name, _ in score_functions)
        scores['total'] = total_score
        
        return scores

def print_detailed_scores(scores: Dict, show_details: bool = False):
    """Print detailed breakdown of all scores."""
    criteria_info = [
        ("1_hand_balance", "1. Use both hands equally"),
        ("2_hand_alternation", "2. Alternate between hands"), 
        ("3_different_fingers", "3. Don't use the same finger"),
        ("4_non_adjacent_fingers", "4. Use non-adjacent fingers"),
        ("5_home_block", "5. Stay within the home block"),
        ("6_dont_skip_home", "6. Don't skip over the home row"),
        ("7_same_row", "7. Stay in the same row"),
        ("8_include_home", "8. Use the home row"),
        ("9_roll_inward", "9. Strum inward (roll in, not out)"),
        ("10_strong_fingers", "10. Use strong fingers")
    ]
    
    print("Dvorak-10 Scoring Results")
    print("=" * 60)
    
    for i, (key, name) in enumerate(criteria_info):
        if key in scores:
            score = scores[key]['score']
            print(f"{name:<40} | {score:6.3f}")
            
            if show_details:
                print_criterion_details(key, scores[key]['details'], i+1)
                print()
    
    print("-" * 60)
    print(f"{'Total Score (sum of all 10)':<40} | {scores['total']:6.3f}")
    print(f"{'Average Score':<40} | {scores['total']/10:6.3f}")

def print_criterion_details(criterion_key: str, details: Dict, criterion_num: int):
    """Print detailed breakdown for a specific criterion."""
    
    def format_digraph_list(digraphs, limit=5):
        """Format a list of digraphs for display."""
        if not digraphs:
            return "None"
        shown = digraphs[:limit]
        formatted = [f"{d[0]}{d[1]}" for d in shown]
        if len(digraphs) > limit:
            formatted.append(f"... (+{len(digraphs) - limit} more)")
        return ", ".join(formatted)
    
    print(f"    Detailed breakdown for criterion {criterion_num}:")
    
    if criterion_num == 1:  # Hand balance
        left_count = details.get('left_count', 0)
        right_count = details.get('right_count', 0)
        total = details.get('total_count', 0)
        if total > 0:
            left_pct = (left_count / total) * 100
            right_pct = (right_count / total) * 100
            print(f"      Left hand: {left_count} keystrokes ({left_pct:.1f}%)")
            print(f"      Right hand: {right_count} keystrokes ({right_pct:.1f}%)")
            print(f"      Imbalance: {abs(left_pct - right_pct):.1f} percentage points")
        
    elif criterion_num == 2:  # Hand alternation
        same = details.get('same_hand', 0)
        diff = details.get('different_hand', 0)
        total = same + diff
        if total > 0:
            same_pct = (same / total) * 100
            print(f"      Same hand digraphs: {same} ({same_pct:.1f}%)")
            print(f"      Different hand digraphs: {diff} ({100-same_pct:.1f}%)")
            print(f"      Examples of same hand: {format_digraph_list(details.get('same_hand_digraphs', []))}")
        
    elif criterion_num == 3:  # Different fingers
        same = details.get('same_finger', 0)
        diff = details.get('different_finger', 0)
        total = same + diff
        if total > 0:
            same_pct = (same / total) * 100
            print(f"      Same finger digraphs: {same} ({same_pct:.1f}%) - BAD")
            print(f"      Different finger digraphs: {diff} ({100-same_pct:.1f}%) - GOOD")
            if same > 0:
                print(f"      Same finger examples: {format_digraph_list(details.get('same_finger_digraphs', []))}")
        
    elif criterion_num == 4:  # Non-adjacent fingers
        adj = details.get('adjacent', 0)
        non_adj = details.get('non_adjacent', 0)
        same_finger = details.get('same_finger', 0)
        diff_finger_total = adj + non_adj
        if diff_finger_total > 0:
            adj_pct = (adj / diff_finger_total) * 100
            print(f"      Adjacent finger digraphs: {adj} ({adj_pct:.1f}%) - BAD")
            print(f"      Non-adjacent finger digraphs: {non_adj} ({100-adj_pct:.1f}%) - GOOD")
            if adj > 0:
                print(f"      Adjacent examples: {format_digraph_list(details.get('adjacent_digraphs', []))}")
        if same_finger > 0:
            print(f"      Same finger digraphs: {same_finger} (excluded from scoring)")
        
    elif criterion_num == 5:  # Home block
        outside = details.get('outside_home_block', 0)
        inside = details.get('inside_home_block', 0)
        total = outside + inside
        if total > 0:
            outside_pct = (outside / total) * 100
            print(f"      Outside home block: {outside} ({outside_pct:.1f}%) - BAD")
            print(f"      Inside home block: {inside} ({100-outside_pct:.1f}%) - GOOD")
            if outside > 0:
                print(f"      Outside examples: {format_digraph_list(details.get('outside_digraphs', []))}")
        
    elif criterion_num == 6:  # Don't skip home
        jump = details.get('jump_home', 0)
        no_jump = details.get('dont_jump_home', 0)
        total = jump + no_jump
        if total > 0:
            jump_pct = (jump / total) * 100
            print(f"      Jump over home row: {jump} ({jump_pct:.1f}%) - BAD")
            print(f"      Don't jump home row: {no_jump} ({100-jump_pct:.1f}%) - GOOD")
            if jump > 0:
                print(f"      Jump examples: {format_digraph_list(details.get('jump_digraphs', []))}")
        
    elif criterion_num == 7:  # Same row
        diff_row = details.get('different_row', 0)
        same_row = details.get('same_row', 0)
        total = diff_row + same_row
        if total > 0:
            diff_pct = (diff_row / total) * 100
            print(f"      Different row digraphs: {diff_row} ({diff_pct:.1f}%) - BAD")
            print(f"      Same row digraphs: {same_row} ({100-diff_pct:.1f}%) - GOOD")
            if diff_row > 0:
                print(f"      Different row examples: {format_digraph_list(details.get('different_row_digraphs', []))}")
        
    elif criterion_num == 8:  # Include home
        outside = details.get('outside_home_row', 0)
        include = details.get('include_home_row', 0)
        total = outside + include
        if total > 0:
            outside_pct = (outside / total) * 100
            print(f"      No home row involvement: {outside} ({outside_pct:.1f}%) - BAD")
            print(f"      Include home row: {include} ({100-outside_pct:.1f}%) - GOOD")
            if outside > 0:
                print(f"      No home row examples: {format_digraph_list(details.get('outside_digraphs', []))}")
        
    elif criterion_num == 9:  # Roll inward
        outward = details.get('outward', 0)
        inward = details.get('inward', 0)
        same_finger = details.get('same_finger', 0)
        diff_finger_total = outward + inward
        if diff_finger_total > 0:
            outward_pct = (outward / diff_finger_total) * 100
            print(f"      Outward rolls: {outward} ({outward_pct:.1f}%) - BAD")
            print(f"      Inward rolls: {inward} ({100-outward_pct:.1f}%) - GOOD")
            if outward > 0:
                print(f"      Outward examples: {format_digraph_list(details.get('outward_digraphs', []))}")
        if same_finger > 0:
            print(f"      Same finger digraphs: {same_finger} (excluded from scoring)")
        
    elif criterion_num == 10:  # Strong fingers
        weak = details.get('weak_finger', 0)
        strong = details.get('strong_finger', 0)
        total = weak + strong
        if total > 0:
            weak_pct = (weak / total) * 100
            print(f"      Weak finger involvement: {weak} ({weak_pct:.1f}%) - BAD")
            print(f"      Strong fingers only: {strong} ({100-weak_pct:.1f}%) - GOOD")
            if weak > 0:
                print(f"      Weak finger examples: {format_digraph_list(details.get('weak_digraphs', []))}")
    
    # Show total applicable cases for context
    if criterion_num in [1]:  # Hand balance uses total_count
        if 'total_count' in details:
            print(f"      Total analyzed: {details['total_count']} keystrokes")
    elif criterion_num in [2]:  # Hand alternation uses all digraphs
        if 'total_digraphs' in details:
            print(f"      Total analyzed: {details['total_digraphs']} digraphs")
    elif criterion_num in [4, 9]:  # Criteria that exclude same-finger digraphs
        if 'total_same_hand' in details:
            total_same_hand = details['total_same_hand']
            total_scored = details.get('total_different_finger', 0)
            print(f"      Total same-hand digraphs: {total_same_hand}")
            print(f"      Scored (different-finger only): {total_scored}")
    else:  # Criteria 3, 5-8, 10 use all same-hand digraphs
        if 'total_same_hand' in details:
            print(f"      Total analyzed: {details['total_same_hand']} same-hand digraphs")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate Dvorak-10 layout scores for keyboard evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Dvorak-10 scoring
  python score_dvorak10.py --items "abc" --positions "FDJ"
  
  # Score full layout with details
  python score_dvorak10.py --items "etaoinsrhldcumfp" --positions "FDESRJKUMIVLA;OW" --details
  
  # Use custom text for analysis
  python score_dvorak10.py --items "abc" --positions "FDJ" --text "abacaba"
        """
    )
    
    parser.add_argument("--items", required=True, 
                       help="String of items (e.g., 'etaoinsrhldcumfp')")
    parser.add_argument("--positions", required=True,
                       help="String of positions (e.g., 'FDESRJKUMIVLA;OW')")
    parser.add_argument("--text", 
                       help="Custom text to analyze (default: uses items string)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed breakdown for each criterion")
    parser.add_argument("--csv", action="store_true",
                       help="Output results in CSV format")
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if len(args.items) != len(args.positions):
            print(f"Error: Item count ({len(args.items)}) != Position count ({len(args.positions)})")
            return
        
        # Create layout mapping
        layout_mapping = dict(zip(args.items.lower(), args.positions.upper()))
        
        # Initialize scorer
        text = args.text if args.text else args.items
        scorer = Dvorak10Scorer(layout_mapping, text)
        
        # Calculate scores
        scores = scorer.calculate_all_scores()
        
        if args.csv:
            # CSV output
            print("criterion,score")
            criteria = ['hand_balance', 'hand_alternation', 'different_fingers', 'non_adjacent_fingers',
                       'home_block', 'dont_skip_home', 'same_row', 'include_home', 'roll_inward', 'strong_fingers']
            for i, criterion in enumerate(criteria, 1):
                key = f"{i}_{criterion}"
                if key in scores:
                    print(f"{criterion},{scores[key]['score']:.6f}")
            print(f"total,{scores['total']:.6f}")
        else:
            # Human-readable output
            print(f"Layout: {args.items} → {args.positions}")
            print(f"Text analyzed: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"Total digraphs: {len(scorer.digraphs)}")
            print()
            
            print_detailed_scores(scores, args.details)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()