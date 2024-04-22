echo "Compiling base..."
"./$(dirname "$0")/compile_base.sh"
echo "Compiling cuda..."
"./$(dirname "$0")/compile_cuda.sh"
echo "Compiling sycl cpu..."
"./$(dirname "$0")/compile_sycl_cpu.sh"
echo "Compiling sycl gpu..."
"./$(dirname "$0")/compile_sycl_gpu.sh"