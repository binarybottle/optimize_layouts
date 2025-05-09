# Count the files in a directory
# Ex: sh count_files.sh output/layouts
find $1 -path "$1/*" | wc -l