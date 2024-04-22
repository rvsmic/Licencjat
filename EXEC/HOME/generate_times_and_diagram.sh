file=$1
test_count=$2
selected_version=$3
ERROR=0

if [ $# -lt 2 ]; then
    echo "Usage: $0 <file> <test_count> <optional: selected_version, default: all>"
    exit 1
fi

if [ -z $selected_version ]; then
    echo "Selected version: all"
elif [ $selected_version == "base" ]; then
    echo "Selected version: base"
    if [ ! -e "base" ]; then
        echo "Base not compiled"
        ERROR=1
    fi
    if [ $ERROR -eq 1 ]; then
        exit 1
    fi
    echo
    echo "Base"
    for i in $(seq 1 $test_count); do
        echo "Test $i"
        ./base $file time > /dev/null
    done
    python3 ../../PYTHON/plot_diagram.py $file $test_count
    file_name=$(echo $file | cut -d '.' -f 1)
    echo
    echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"
    exit 0
elif [ $selected_version == "cuda" ]; then
    echo "Selected version: cuda"
    if [ ! -e "cuda" ]; then
        echo "Cuda not compiled"
        ERROR=1
    fi
    if [ $ERROR -eq 1 ]; then
        exit 1
    fi
    echo
    echo "Cuda"
    for i in $(seq 1 $test_count); do
        echo "Test $i"
        ./cuda $file time > /dev/null
    done
    python3 ../../PYTHON/plot_diagram.py $file $test_count
    file_name=$(echo $file | cut -d '.' -f 1)
    echo
    echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"
    exit 0
elif [ $selected_version == "sycl_cpu" ]; then
    echo "Selected version: sycl_cpu"
    if [ ! -e "sycl_cpu" ]; then
        echo "Sycl CPU not compiled"
        ERROR=1
    fi
    if [ $ERROR -eq 1 ]; then
        exit 1
    fi
    echo
    echo "Sycl CPU"
    for i in $(seq 1 $test_count); do
        echo "Test $i"
        ./sycl_cpu $file time cpu > /dev/null
    done
    python3 ../../PYTHON/plot_diagram.py $file $test_count
    file_name=$(echo $file | cut -d '.' -f 1)
    echo
    echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"
    exit 0
elif [ $selected_version == "sycl_gpu" ]; then
    echo "Selected version: sycl_gpu"
    if [ ! -e "sycl_gpu" ]; then
        echo "Sycl GPU not compiled"
        ERROR=1
    fi
    if [ $ERROR -eq 1 ]; then
        exit 1
    fi
    echo
    echo "Sycl GPU"
    for i in $(seq 1 $test_count); do
        echo "Test $i"
        ./sycl_gpu $file time gpu > /dev/null
    done
    python3 ../../PYTHON/plot_diagram.py $file
    file_name=$(echo $file | cut -d '.' -f 1)
    echo
    echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"
    exit 0
else
    echo "Invalid version"
    exit 1
fi

if [ ! -e "base" ]; then
    echo "Base not compiled"
    ERROR=1
fi

if [ ! -e "cuda" ]; then
    echo "Cuda not compiled"
    ERROR=1
fi

if [ ! -e "sycl_cpu" ]; then
    echo "Sycl CPU not compiled"
    ERROR=1
fi

if [ ! -e "sycl_gpu" ]; then
    echo "Sycl GPU not compiled"
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
if [ -e "../DATA/OUT/sycl_cpu.time" ]; then
    rm ../DATA/OUT/sycl_cpu.time
fi
if [ -e "../DATA/OUT/sycl_gpu.time" ]; then
    rm ../DATA/OUT/sycl_gpu.time
fi

# base
echo
echo "Base"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./base $file time > /dev/null
done

# cuda
echo
echo "Cuda"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./cuda $file time > /dev/null
done

# sycl cpu
echo
echo "Sycl CPU"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./sycl_cpu $file time cpu > /dev/null
done

# sycl gpu
echo
echo "Sycl GPU"
for i in $(seq 1 $test_count); do
    echo "Test $i"
    ./sycl_gpu $file time gpu > /dev/null
done

python3 ../../PYTHON/plot_diagram.py $file $test_count

file_name=$(echo $file | cut -d '.' -f 1)

echo
echo "Generated diagram for $file for average of $test_count times in DOCS/histogram_$file_name.png"
