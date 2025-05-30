#!/bin/bash

# Permutation & Combination Calculator
# PERMUTATIONS (order matters): P(n,r) = n! / (n-r)!
# COMBINATIONS (order doesn't matter): C(n,r) = n! / (r! × (n-r)!)
# Usage: ./permutation_calc.sh <total_positions> <items_to_arrange> [--combinations]
# Example: ./permutation_calc.sh 16 3  (3 letters in 16 keys, ORDER MATTERS)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    echo -e "${YELLOW}Usage: $0 <total_positions> <items_to_arrange> [--combinations]${NC}"
    echo
    echo -e "${GREEN}Default (PERMUTATIONS - order matters):${NC}"
    echo "  $0 16 3    # 3 letters arranged in 16 key positions (ABC ≠ BAC)"
    echo "  $0 26 12   # 12 letters arranged in 26 positions" 
    echo "  $0 10 5    # 5 items arranged in 10 positions"
    echo
    echo -e "${BLUE}With --combinations flag (order doesn't matter):${NC}"
    echo "  $0 16 3 --combinations    # 3 letters chosen from 16 (ABC = BAC)"
    echo
    echo -e "${YELLOW}Formulas:${NC}"
    echo "  Permutations: P(n,r) = n! / (n-r)!     [order matters]"
    echo "  Combinations: C(n,r) = n! / (r! × (n-r)!)  [order doesn't matter]"
    exit 1
}

# Function to calculate factorial using bc for large numbers
factorial() {
    local n=$1
    if [ $n -eq 0 ] || [ $n -eq 1 ]; then
        echo 1
    else
        echo "define factorial(x) { if (x <= 1) return 1; return x * factorial(x-1) } factorial($n)" | bc -l
    fi
}

# Function to calculate combinations C(n,r) = n! / (r! × (n-r)!)
combination() {
    local n=$1
    local r=$2
    
    if [ $r -eq 0 ] || [ $r -eq $n ]; then
        echo 1
        return
    fi
    
    # Use the more efficient formula: C(n,r) = P(n,r) / r!
    local permutation_result=$(permutation_efficient $n $r)
    local r_factorial=$(factorial $r)
    echo "scale=0; $permutation_result / $r_factorial" | bc
}

# Function to calculate permutations more efficiently
# P(n,r) = n × (n-1) × (n-2) × ... × (n-r+1)
permutation_efficient() {
    local n=$1
    local r=$2
    
    if [ $r -eq 0 ]; then
        echo 1
        return
    fi
    
    local result=1
    local i
    for (( i=0; i<r; i++ )); do
        result=$(echo "$result * ($n - $i)" | bc)
    done
    echo $result
}

# Function to format large numbers with commas (cross-platform)
format_number() {
    local num=$1
    # Pure bash solution - works on all systems
    local formatted=""
    local length=${#num}
    local count=0
    
    # Process digits from right to left
    for (( i=length-1; i>=0; i-- )); do
        if [ $count -eq 3 ]; then
            formatted=",${formatted}"
            count=0
        fi
        formatted="${num:$i:1}${formatted}"
        ((count++))
    done
    
    echo "$formatted"
}

# Function to convert to scientific notation for very large numbers
to_scientific() {
    local num=$1
    if [ ${#num} -gt 15 ]; then
        echo $num | awk '{printf "%.2e", $1}'
    else
        echo $num
    fi
}

# Check arguments
calculate_combinations=false
if [ $# -eq 3 ] && [ "$3" = "--combinations" ]; then
    calculate_combinations=true
elif [ $# -ne 2 ]; then
    echo -e "${RED}Error: 2 arguments required, or 3 with --combinations flag${NC}"
    usage
fi

# Validate inputs
if ! [[ "$1" =~ ^[0-9]+$ ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Arguments must be non-negative integers${NC}"
    usage
fi

n=$1  # total positions
r=$2  # items to arrange

# Validate logical constraints
if [ $r -gt $n ]; then
    echo -e "${RED}Error: Cannot arrange $r items in only $n positions${NC}"
    echo "Items to arrange ($r) must be ≤ total positions ($n)"
    exit 1
fi

if [ $n -eq 0 ]; then
    echo -e "${RED}Error: Total positions cannot be 0${NC}"
    exit 1
fi

# Header
if [ "$calculate_combinations" = true ]; then
    echo -e "${BLUE}🧮 Combination Calculator (Order Doesn't Matter)${NC}"
    echo "=================================================="
else
    echo -e "${BLUE}🧮 Permutation Calculator (Order Matters)${NC}"
    echo "=============================================="
fi
echo

# Show the problem
echo -e "${YELLOW}Problem:${NC}"
if [ "$calculate_combinations" = true ]; then
    echo "  Choosing $r items from $n positions (order doesn't matter)"
    echo "  Formula: C($n,$r) = $n! / ($r! × ($n-$r)!)"
    echo "  Example: Choosing letters A,B,C is same as C,A,B"
else
    echo "  Arranging $r items in $n positions (order matters)"
    echo "  Formula: P($n,$r) = $n! / ($n-$r)!"
    echo "  Example: Arranging letters ABC is different from BAC"
fi
echo

# Calculate step by step
echo -e "${YELLOW}Calculation:${NC}"

if [ "$calculate_combinations" = true ]; then
    # Combinations calculation
    if [ $r -eq 0 ]; then
        echo "  C($n,0) = 1 (by definition)"
        result=1
    elif [ $r -eq $n ]; then
        echo "  C($n,$n) = 1 (only one way to choose all items)"
        result=1
    else
        echo "  C($n,$r) = $n! / ($r! × ($n-$r)!)"
        echo "  C($n,$r) = $n! / ($r! × $(($n-$r))!)"
        result=$(combination $n $r)
    fi
else
    # Permutations calculation (original logic)
    if [ $r -eq 0 ]; then
        echo "  P($n,0) = 1 (by definition)"
        result=1
    elif [ $r -eq $n ]; then
        echo "  P($n,$n) = $n! (all items, all positions)"
        result=$(factorial $n)
    else
        echo "  P($n,$r) = $n! / ($n-$r)!"
        echo "  P($n,$r) = $n! / $(($n-$r))!"
        
        # For efficiency, calculate as n × (n-1) × ... × (n-r+1)
        echo
        echo "  More efficiently:"
        echo -n "  P($n,$r) = "
        
        # Build the multiplication string
        mult_string=""
        for (( i=0; i<r; i++ )); do
            if [ $i -eq 0 ]; then
                mult_string="$n"
            else
                mult_string="$mult_string × $(($n-$i))"
            fi
        done
        echo "$mult_string"
        
        result=$(permutation_efficient $n $r)
    fi
fi

echo

# Display results
echo -e "${GREEN}📊 Results:${NC}"
echo "=================================="

# Raw number
echo -e "Raw result: ${BLUE}$result${NC}"

# Formatted with commas
formatted=$(format_number $result)
echo -e "Formatted:  ${BLUE}$formatted${NC}"

# Scientific notation for large numbers
scientific=$(to_scientific $result)
if [ "$scientific" != "$result" ]; then
    echo -e "Scientific: ${BLUE}$scientific${NC}"
fi

# Context and comparison
echo
echo -e "${YELLOW}📈 Context:${NC}"
if [ "$calculate_combinations" = true ]; then
    echo "  This means there are $formatted different ways"
    echo "  to choose $r items from $n positions (order doesn't matter)."
else
    echo "  This means there are $formatted different ways"
    echo "  to arrange $r items among $n positions (order matters)."
fi

# Show comparison between permutations and combinations
echo
echo -e "${YELLOW}🔄 Comparison (Order Matters vs Doesn't Matter):${NC}"
if [ "$calculate_combinations" = true ]; then
    # Show what permutations would be
    perm_result=$(permutation_efficient $n $r)
    perm_formatted=$(format_number $perm_result)
    echo "  Permutations (order matters): $perm_formatted"
    echo "  Combinations (order doesn't): $formatted"
    if [ $r -gt 1 ]; then
        r_fact=$(factorial $r)
        echo "  Difference factor: ${r}! = $r_fact"
    fi
else
    # Show what combinations would be
    if [ $r -gt 0 ] && [ $r -lt $n ]; then
        comb_result=$(combination $n $r)
        comb_formatted=$(format_number $comb_result)
        echo "  Permutations (order matters): $formatted"
        echo "  Combinations (order doesn't): $comb_formatted"
        if [ $r -gt 1 ]; then
            r_fact=$(factorial $r)
            echo "  Difference factor: ${r}! = $r_fact"
        fi
    fi
fi

echo
