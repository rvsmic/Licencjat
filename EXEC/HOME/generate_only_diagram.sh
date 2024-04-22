file=$1
test_count=$2

if [ $# -lt 2 ]; then
    echo "Usage: $0 <file> <test_count>"
    exit 1
fi

python3 ../../PYTHON/plot_diagram.py $file $test_count

file_name=$(echo $file | cut -d '.' -f 1)

echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"