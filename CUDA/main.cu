#include <iostream>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <jpeglib.h>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 16
#define GDAL_BLOCK_DIM 256

typedef unsigned int uint;
typedef unsigned char uchar;

// Kernel for calculating shade for a single pixel on GPU
__global__ void calculateShadeKernel(float* pixelArray, float* shadeArr, int height, int width, size_t pixelPitch, size_t shadePitch, int azimuth, int altitude) {
    pixelPitch /= sizeof(float);
    shadePitch /= sizeof(float);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int zenithDegree = 90 - altitude;
    float zenithRadian = float(zenithDegree) * (M_PI / 180);
    int azimuthMath = (360 - azimuth + 90) % 360;
    float azimuthRadian = float(azimuthMath) * (M_PI / 180);

    // Cell size for shading - "real" pixel size -> bigger = less accurate shading
    int cellSize = 5;
    // Z factor - height exaggeration -> bigger = more intense shading
    int zFactor = 1;

    if (i>0 && i<height-1 && j>0 && j<width-1) {
        // Change in height in x-axis
        float changeX = (
            (
                pixelArray[(i-1)*pixelPitch + j+1] + (2 * pixelArray[i*pixelPitch + j+1]) + pixelArray[(i+1)*pixelPitch + j+1]
            ) - (
                pixelArray[(i-1)*pixelPitch + j-1] + (2 * pixelArray[i*pixelPitch + j-1]) + pixelArray[(i+1)*pixelPitch + j-1]
            )
        ) / (8 * cellSize);
        // Change in height in y-axis
        float changeY = (
            (
                pixelArray[(i+1)*pixelPitch + j-1] + (2 * pixelArray[(i+1)*pixelPitch + j]) + pixelArray[(i+1)*pixelPitch + j+1]
            ) - (
                pixelArray[(i-1)*pixelPitch + j-1] + (2 * pixelArray[(i-1)*pixelPitch + j]) + pixelArray[(i-1)*pixelPitch + j+1]
            )
        ) / (8 * cellSize);
        // Final slope radian
        float slopeRadian = atan(zFactor * sqrt(pow(changeX, 2) + pow(changeY, 2)));
        // Slope aspect radian
        float aspectRadian;
        if (changeX != 0) {
            aspectRadian = atan2(changeY, -changeX);
            if (aspectRadian < 0) {
                aspectRadian = 2 * M_PI + aspectRadian;
            }
        } else {
            if (changeY > 0) {
                aspectRadian = M_PI / 2;
            } else if (changeY < 0) {
                aspectRadian = 3 * M_PI / 2;
            } else {
                aspectRadian = 0;
            }
        }
        // Shade value for pixel
        float hillShade = 255.0 * (
            (
                cos(zenithRadian) * cos(slopeRadian)
            ) + (
                sin(zenithRadian) * sin(slopeRadian) * cos(azimuthRadian - aspectRadian)
            )
        );
        if (hillShade < 0) {
            hillShade = 0;
        }
        shadeArr[(i-1)*shadePitch + (j-1)] = hillShade;
    }
}

// Calculate shade on GPU
float* calculateShade(float* &pixelArray, const uint height, const uint width, int azimuth=315, int altitude=45) {
    float* shadeArr = new float[(height - 2) * (width - 2)]();
    float *devicePixelArr, *deviceShadeArr;

    // Calculate if arrays fit in GPU memory
    size_t freeGPUBytes, totalGPUBytes;
    cudaMemGetInfo(&freeGPUBytes, &totalGPUBytes);
    size_t totalExpectedSize = (height*width + (height-2) * (width-2)) * sizeof(float);
    
    if(totalExpectedSize > freeGPUBytes) {
        std::cout<<"Too much data at once for GPU memory ("<<totalExpectedSize/1024.0/1024.0<<" MB > "<<freeGPUBytes/1024.0/1024.0<<" MB) - splitting\n";
        // Constant width and varying height
        size_t dataBlockHeight = ((freeGPUBytes*0.9/sizeof(float)) + 2*width - 4) / (2 * width - 2);
        size_t iterationCount = (totalExpectedSize) / (((dataBlockHeight*width)+((dataBlockHeight-2)*(width-2)))*sizeof(float));
        
        // Real row width on gpu
        size_t pixelPitch, shadePitch;

        // Allocate arrays on gpu
        cudaMallocPitch(&devicePixelArr, &pixelPitch, width*sizeof(float), dataBlockHeight);
        cudaMallocPitch(&deviceShadeArr, &shadePitch, (width-2)*sizeof(float), dataBlockHeight-2);

        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize((dataBlockHeight + (BLOCK_DIM-1))/BLOCK_DIM, (width + (BLOCK_DIM-1))/BLOCK_DIM);
        
        for(size_t i = 0; i < iterationCount; ++i) {
            // Copy pixel array values to GPU
            cudaMemcpy2D(devicePixelArr, pixelPitch, &pixelArray[i*width*dataBlockHeight], width*sizeof(float), width*sizeof(float), dataBlockHeight, cudaMemcpyHostToDevice);

            calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, dataBlockHeight, width, pixelPitch, shadePitch, azimuth, altitude);
            
            // Check for error
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cout<<"\nCUDA error: "<<cudaGetErrorString(error)<<"\n\n";
            }

            // Copy shade array values to host
            cudaMemcpy2D(&shadeArr[i*(width-2)*(dataBlockHeight-2)], (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), dataBlockHeight-2, cudaMemcpyDeviceToHost);
        }

        // Copy pixel array values to gpu
        cudaMemcpy2D(devicePixelArr, pixelPitch, &pixelArray[iterationCount*width*dataBlockHeight], width*sizeof(float), width*sizeof(float), dataBlockHeight, cudaMemcpyHostToDevice);

        calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, dataBlockHeight, width, pixelPitch, shadePitch, azimuth, altitude);
        
        // Check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cout<<"\nCUDA error: "<<cudaGetErrorString(error)<<"\n\n";
        }

        // Copy shade array values to host
        cudaMemcpy2D(&shadeArr[iterationCount*(width-2)*(dataBlockHeight-2)], (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), dataBlockHeight-2, cudaMemcpyDeviceToHost);
    } else {
        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize((height + (BLOCK_DIM-1))/BLOCK_DIM, (width + (BLOCK_DIM-1))/BLOCK_DIM); // na odwrot

        // Real row width on gpu
        size_t pixelPitch, shadePitch;

        // Allocate arrays on gpu
        cudaMallocPitch(&devicePixelArr, &pixelPitch, width*sizeof(float), height);
        cudaMallocPitch(&deviceShadeArr, &shadePitch, (width-2)*sizeof(float), height-2);

        // Copy pixel array values to gpu
        cudaMemcpy2D(devicePixelArr, pixelPitch, pixelArray, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);

        calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, height, width, pixelPitch, shadePitch, azimuth, altitude);

        // Check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cout<<"\nCUDA error: "<<cudaGetErrorString(error)<<"\n\n";
        }

        // Copy shade array values to host
        cudaMemcpy2D(shadeArr, (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), height-2, cudaMemcpyDeviceToHost);
    }

    cudaFree(devicePixelArr);
    cudaFree(deviceShadeArr);
    return shadeArr;
}

// Load GeoTIFF file
float* loadGeoTIFF(const std::string &filename, uint &height, uint &width) {
    GDALAllRegister();
    // Path for input data folder
    std::string path = "../DATA/IN/" + filename;
    // Pixel data vector
    float* elevationData = nullptr;

    // Load GeoTIFF
    auto *poDataset = (GDALDataset *) GDALOpen(path.c_str(), GA_ReadOnly);
    bool readFailure = false;

    // If file exists
    if (poDataset != nullptr) {
        int nXSize = poDataset->GetRasterXSize();
        int nYSize = poDataset->GetRasterYSize();

        height = nYSize;
        width = nXSize;

        // Resize for height
        elevationData = new float[height * width]();

        // Read and save to vector
        for (int y = 0; y < nYSize; y += GDAL_BLOCK_DIM) {
            for (int x = 0; x < nXSize; x += GDAL_BLOCK_DIM) {
                // Reading block size
                int readWidth = std::min(GDAL_BLOCK_DIM, nXSize - x);
                int readHeight = std::min(GDAL_BLOCK_DIM, nYSize - y);

                float* blockData = new float[readWidth * readHeight]();

                // Get block values from dataset
                CPLErr err = poDataset->GetRasterBand(1)->RasterIO(
                    GF_Read, x, y, readWidth, readHeight,
                    blockData, readWidth, readHeight, GDT_Float32,
                    0,0
                );

                // Check for error
                if (err == CE_None) {
                    // Copy data from block to the pixel array
                    for (int i = 0; i < readHeight; ++i) {
                        for (int j = 0; j < readWidth; ++j) {
                            elevationData[(y+i)*width + x+j] = blockData[i*readWidth + j];
                        }
                    }
                } else {
                    readFailure = true;
                    std::cerr<<"Block values read failure: ("<<x<<", "<<y<<")"<<std::endl;
                    delete[] blockData;
                    break;
                }

                delete[] blockData;
            }
            if (readFailure) {
                break;
            }
        }

        // Close dataset
        GDALClose(poDataset);
    } else {
        std::cerr<<"Can't open GeoTIFF: "<<path<<std::endl;
    }
    GDALDestroyDriverManager();

    // If there was an error with reading
    if(readFailure && elevationData != nullptr) {
        delete[] elevationData;
        elevationData = nullptr;
    }

    return elevationData;
}

// Apply shade on original image
void applyShade(float* &pixels, float* &shade, const uint height, const uint width) {
    for (int i=1; i < height - 1; ++i) {
        for (int j=1; j < width - 1; ++j) {
            pixels[(i*width) + j] += shade[((i-1)*(width-2)) + j - 1];
        }
    }
}

// Normalise pixel value for preview
uchar* normalisePixels(float* &pixels, const uint height, const uint width) {
    auto* normalisedPixels = new uchar[height * width]();

    // Find min and max values
    float minVal = pixels[0], maxVal = pixels[0];
    for (int i=0;i<height;++i) {
        for (int j=0;j<width;++j) {
            if (pixels[(i*width) + j] < minVal) minVal = pixels[(i*width) + j];
            if (pixels[(i*width) + j] > maxVal) maxVal = pixels[(i*width) + j];
        }
    }

    // Calculate the difference
    float difference = maxVal - minVal;

    // Normalise the values
    for (int i=0; i < height; ++i) {
        for (int j=0; j < width; ++j) {
            normalisedPixels[(i * width) + j] = static_cast<uchar>(((pixels[(i*width) + j] - minVal) / difference) * 255);
        }
    }

    return normalisedPixels;
}

// Generate jpg preview of a tiff file
void saveToJPEG(const std::string& fileName, float* &pixels, const uint height, const uint width) {
    jpeg_compress_struct info{};
    jpeg_error_mgr err{};

    FILE *file;
    std::string filePath = "../DATA/OUT/" + fileName;

    if ((file = fopen(filePath.c_str(), "wb")) == nullptr) {
        std::cerr<<"Can't open file "<<filePath<<std::endl;
        return;
    }

    info.err = jpeg_std_error(&err);
    jpeg_create_compress(&info);
    jpeg_stdio_dest(&info, file);

    info.image_width = width;
    info.image_height = height;
    info.input_components = 1; // Jedna składowa - skala szarości
    info.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, 100, TRUE);

    jpeg_start_compress(&info, TRUE);

    uchar* lpRowBuffer[1];
    auto normalisedPixels = normalisePixels(pixels, height, width);

    while (info.next_scanline < info.image_height) {
        lpRowBuffer[0] = &(normalisedPixels[info.next_scanline * info.image_width]);
        jpeg_write_scanlines(&info, lpRowBuffer, 1);
    }

    jpeg_finish_compress(&info);
    fclose(file);
    jpeg_destroy_compress(&info);
}

int main() {
    uint height = 0, width = 0;
    std::cout<<"Loading image...\n";
    float* pixelArr = loadGeoTIFF("fiji.tif", height, width);
    std::cout<<"Image dimensions: "<<height<<" "<<width<<"\n";
    
    std::cout<<"Saving preview...\n";
    saveToJPEG("pre_shade.jpeg", pixelArr, height, width);

    std::cout<<"Calculating shade...\n";
    auto start = std::chrono::high_resolution_clock::now();
    float* shadeArr = calculateShade(pixelArr, height, width);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout<<"Applying shade...\n";
    applyShade(pixelArr, shadeArr, height, width);

    std::cout<<"Saving preview...\n";
    saveToJPEG("post_shade.jpeg", pixelArr, height, width);

    std::cout<<"Done!\n";
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"Shade function time: "<<duration.count()<<" ms\n";

    return 0;
}
