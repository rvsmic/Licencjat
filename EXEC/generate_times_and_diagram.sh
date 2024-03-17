file="fiji.tif"
test_count=10
ERROR=0

if [ $# -gt 0 ]; then
    file=$1
fi

if [ $# -gt 1 ]; then
    test_count=$2
fi

if [ ! -e "base" ]; then
    echo "Base not compiled"
    ERROR=1
fi

if [ ! -e "cuda" ]; then
    echo "Cuda not compiled"
    ERROR=1
fi

if [ ! -e "sycl" ]; then
    echo "Sycl not compiled"
    ERROR=1
fi

if [ $ERROR -eq 1 ]; then
    exit 1
fi

# remove old times
if [ -e "../DATA/OUT/base.time" ]; then
    rm ../DATA/OUT/base.time
fi
if [ -e "../DATA/OUT/cuda.time" ]; then
    rm ../DATA/OUT/cuda.time
fi
if [ -e "../DATA/OUT/sycl.time" ]; then
    rm ../DATA/OUT/sycl.time
fi

# base
echo "Base"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./base $file time > /dev/null
done

# cuda
echo "Cuda"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./cuda $file time > /dev/null
done

# sycl
echo "Sycl"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./sycl $file time > /dev/null
done

python3 ../PYTHON/plot_diagram.py $file $test_count

echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file.png"
