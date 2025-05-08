for type in "Output files:output/layouts/layout_results_*"; do
      # Split the string into label and pattern
      label=${type%%:*}
      pattern=${type#*:}
      
      # Print label
      echo "$label"
      
      # Find files matching pattern across all configs and sum the counts
      find output/layouts/ -path "$pattern" | wc -l
done

