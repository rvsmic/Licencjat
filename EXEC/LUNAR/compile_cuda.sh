nvcc -std=c++17 -o "$(dirname "$0")/cuda" "$(dirname "$0")/../CUDA/main.cu" -lgdal