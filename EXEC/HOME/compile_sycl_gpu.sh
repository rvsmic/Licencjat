icpx -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_52 -o "$(dirname "$0")/sycl_gpu" "$(dirname "$0")/../../SYCL/main.cpp" -lgdal -ljpeg