#include <iostream>
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <jpeglib.h>
#include <cmath>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define BLOCK_DIM 16

typedef unsigned int uint;
typedef unsigned char uchar;

// Kernel for calculating shade for a single pixel on GPU
__global__ void calculateShadeKernel(float* pixelArray, float* shadeArr, int height, int width, float cellSize, size_t pixelPitch, size_t shadePitch, int azimuth, int altitude) {
    pixelPitch /= sizeof(float);
    shadePitch /= sizeof(float);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int zenithDegree = 90 - altitude;
    float zenithRadian = float(zenithDegree) * (M_PI / 180);
    int azimuthMath = (360 - azimuth + 90) % 360;
    float azimuthRadian = float(azimuthMath) * (M_PI / 180);

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
float* calculateShade(float* &pixelArray, const uint height, const uint width, float cellSize, int azimuth=315, int altitude=45) {
    float* shadeArr = new float[(height - 2) * (width - 2)]();
    float *devicePixelArr, *deviceShadeArr;

    // Calculate if arrays fit in GPU memory
    size_t freeGPUBytes, totalGPUBytes;
    cudaMemGetInfo(&freeGPUBytes, &totalGPUBytes);
    size_t totalExpectedSize = (height*width + (height-2) * (width-2)) * sizeof(float);
    
    if(totalExpectedSize > freeGPUBytes) {
        printf("Too much data at once for GPU memory (%.2f MB > %.2f MB) - splitting\n", totalExpectedSize/1024.0/1024.0, freeGPUBytes/1024.0/1024.0);
        // Constant width and varying height
        size_t dataBlockHeight = ((freeGPUBytes*0.9/sizeof(float)) + 2*width - 4) / (2 * width - 2);
        size_t iterationCount = (totalExpectedSize) / (((dataBlockHeight*width)+((dataBlockHeight-2)*(width-2)))*sizeof(float));
        
        // Real row width on GPU
        size_t pixelPitch, shadePitch;

        // Declare cuda streams
        const uint streamCount = 3;
        cudaStream_t streams[streamCount];

        printf("Running %d concurrent streams\n", streamCount);
        
        // Define cuda streams
        for(int i = 0; i < streamCount; ++i) {
            cudaStreamCreate(&streams[i]);
        }
        
        // Allocate arrays on GPU
        cudaMallocPitch(&devicePixelArr, &pixelPitch, width*sizeof(float), dataBlockHeight);
        cudaMallocPitch(&deviceShadeArr, &shadePitch, (width-2)*sizeof(float), dataBlockHeight-2);

        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize((dataBlockHeight + (BLOCK_DIM-1))/BLOCK_DIM, (width + (BLOCK_DIM-1))/BLOCK_DIM);
    
        for(size_t i = 0; i < iterationCount; ++i) {
            // Copy pixel array values to GPU
            cudaMemcpy2DAsync(devicePixelArr, pixelPitch, &pixelArray[i*width*dataBlockHeight], width*sizeof(float), width*sizeof(float), dataBlockHeight, cudaMemcpyHostToDevice, streams[i%streamCount]);
        
            calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, dataBlockHeight, width, cellSize, pixelPitch, shadePitch, azimuth, altitude);
        
            // Check for error
            cudaError_t error = cudaGetLastError();
            if(error != cudaSuccess) {
                printf("\nCUDA error: %s\n\n", cudaGetErrorString(error));
            }
            // Copy shade array values to host
            cudaMemcpy2DAsync(&shadeArr[i*(width-2)*(dataBlockHeight-2)], (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), dataBlockHeight-2, cudaMemcpyDeviceToHost, streams[i%streamCount]);
        }

        size_t heightLeft = (height - iterationCount*dataBlockHeight);
        // Copy pixel array values to GPU
        cudaMemcpy2DAsync(devicePixelArr, pixelPitch, &pixelArray[iterationCount*width*dataBlockHeight], width*sizeof(float), width*sizeof(float), heightLeft, cudaMemcpyHostToDevice, streams[iterationCount%streamCount]);

        calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, dataBlockHeight, width, cellSize, pixelPitch, shadePitch, azimuth, altitude);
    
        // Check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("\nCUDA error: %s\n\n", cudaGetErrorString(error));
        }

        // Copy shade array values to host
        cudaMemcpy2DAsync(&shadeArr[iterationCount*(width-2)*(dataBlockHeight-2)], (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), heightLeft-2, cudaMemcpyDeviceToHost, streams[iterationCount%streamCount]);
        
        // Wait for completion of calculations and then destroy streams
        for(int i = 0; i < streamCount; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    } else {
        dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
        dim3 gridSize((height + (BLOCK_DIM-1))/BLOCK_DIM, (width + (BLOCK_DIM-1))/BLOCK_DIM);

        // Real row width on GPU
        size_t pixelPitch, shadePitch;

        // Allocate arrays on GPU
        cudaMallocPitch(&devicePixelArr, &pixelPitch, width*sizeof(float), height);
        cudaMallocPitch(&deviceShadeArr, &shadePitch, (width-2)*sizeof(float), height-2);

        // Copy pixel array values to GPU
        cudaMemcpy2D(devicePixelArr, pixelPitch, pixelArray, width*sizeof(float), width*sizeof(float), height, cudaMemcpyHostToDevice);

        calculateShadeKernel <<< gridSize, blockSize >>> (devicePixelArr, deviceShadeArr, height, width, cellSize, pixelPitch, shadePitch, azimuth, altitude);

        // Check for error
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("\nCUDA error: %s\n\n", cudaGetErrorString(error));
        }

        // Copy shade array values to host
        cudaMemcpy2D(shadeArr, (width-2)*sizeof(float), deviceShadeArr, shadePitch, (width-2)*sizeof(float), height-2, cudaMemcpyDeviceToHost);
    }

    cudaFree(devicePixelArr);
    cudaFree(deviceShadeArr);
    return shadeArr;
}

// Load GeoTIFF file
float* loadGeoTIFF(const std::string &filename, uint &height, uint &width, float &cellSize) {
    GDALAllRegister();
    // Path for input data folder
    std::string path = "../../DATA/IN/" + filename;
    // Pixel data vector
    float* elevationData = nullptr;

    // Load GeoTIFF
    auto *dataset = (GDALDataset *) GDALOpen(path.c_str(), GA_ReadOnly);

    // If file exists
    if (dataset != nullptr) {
        int nXSize = dataset->GetRasterXSize();
        int nYSize = dataset->GetRasterYSize();

        height = nYSize;
        width = nXSize;

        // Get cellSize
        double adfGeoTransform[6];
        dataset->GetGeoTransform(adfGeoTransform);
        cellSize = adfGeoTransform[1];

        // Allocate memory for the elevation data
        elevationData = new float[height * width]();
        
        // Read data from the dataset
        CPLErr err = dataset->GetRasterBand(1)->RasterIO(
            GF_Read, 0, 0, width, height,
            elevationData, width, height, GDT_Float32, 0, 0
        );

        if (err != CE_None) {
            std::cerr<<"Failed to write data to the new GeoTIFF dataset."<<std::endl;
            height = 0;
            width = 0;
            delete[] elevationData;
        }
        // Close dataset
        GDALClose(dataset);
    } else {
        std::cerr<<"Can't open GeoTIFF: "<<path<<std::endl;
        height = 0;
        width = 0;
    }
    GDALDestroyDriverManager();

    return elevationData;
}

// Save GeoTIFF file
void saveGeoTIFF(const std::string &originalFilename, float* &newData, uint height, uint width) {
    GDALAllRegister();

    // Path for input data folder
    std::string inputPath = "../../DATA/IN/" + originalFilename;

    // Open the original dataset for reading
    auto *sourceDataset = (GDALDataset *) GDALOpen(inputPath.c_str(), GA_ReadOnly);
    if (sourceDataset == nullptr) {
        std::cerr<<"Can't open GeoTIFF: "<<inputPath<<std::endl;
        GDALDestroyDriverManager();
        return;
    }

    // Path for output data folder
    std::string outputPath = "../../DATA/OUT/shaded_" + originalFilename;

    // Create a copy of the original dataset for writing
    auto *destinationDataset = GetGDALDriverManager()->GetDriverByName("GTiff")->CreateCopy(
        outputPath.c_str(), sourceDataset, false, NULL, NULL, NULL
    );

    if (destinationDataset != nullptr) {
        // Write data to the new dataset
        CPLErr err = destinationDataset->GetRasterBand(1)->RasterIO(
            GF_Write, 0, 0, width, height,
            newData, width, height, GDT_Float32, 0, 0
        );

        if (err != CE_None) {
            std::cerr<<"Failed to write data to the new GeoTIFF dataset."<<std::endl;
        }

        // Close the datasets
        GDALClose(sourceDataset);
        GDALClose(destinationDataset);
    } else {
        std::cerr<<"Failed to create a new GeoTIFF dataset for writing."<<std::endl;
    }

    GDALDestroyDriverManager();
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
    std::string filePath = "../../DATA/OUT/" + fileName;

    if ((file = fopen(filePath.c_str(), "wb")) == nullptr) {
        std::cerr<<"Can't open file "<<filePath<<std::endl;
        return;
    }

    info.err = jpeg_std_error(&err);
    jpeg_create_compress(&info);
    jpeg_stdio_dest(&info, file);

    info.image_width = width;
    info.image_height = height;
    info.input_components = 1;
    info.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&info);
    jpeg_set_quality(&info, 90, TRUE);

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

void saveTimeToFile(const std::string& fileName, const std::string& time) {
    std::string filePath = "../../DATA/OUT/" + fileName;
    std::ofstream file;
    file.open(filePath, std::ios::app);
    file<<time<<"\n";
    file.close();
}

int main(int argc, char** argv) {
    std::string tiffFileName = "fuji.tif";
    bool timeMode = false;
    if(argc > 1) {
        tiffFileName = argv[1];
    }
    if(argc > 2) {
        if(std::string(argv[2]) == "time") {
            timeMode = true;
        }
    }
    uint height = 0, width = 0;
    float cellSize;

    // Print device info
    cudaDeviceProp deviceProp;
    cudaSetDevice(0);
    cudaGetDeviceProperties(&deviceProp, 0);
    size_t totalMemory = deviceProp.totalGlobalMem;
    int cpuCount = deviceProp.multiProcessorCount;
    printf("Device: %s\n", deviceProp.name);
    printf("Max compute units: %d\n", cpuCount);
    printf("Total memory: %lu MB\n", totalMemory/1024/1024);

    if(timeMode) {
        float* pixelArr = loadGeoTIFF(tiffFileName, height, width, cellSize);
        if(height == 0 || width == 0) {
            return 1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        float* shadeArr = calculateShade(pixelArr, height, width, cellSize);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        saveTimeToFile("cuda.time", std::to_string(duration.count()));
        return 0;
    }

    printf("Loading image...\n");
    float* pixelArr = loadGeoTIFF(tiffFileName, height, width, cellSize);
    if(height == 0 || width == 0) {
        printf("Failed to load image\n");
        return 1;
    }
    printf("Loaded image: %s\n", tiffFileName.c_str());
    printf("Image dimensions:\n    y: %d\n    x: %d\n", height, width);
    printf("Cell size: %f\n", cellSize);
    
    printf("Saving preview...\n");
    saveToJPEG("pre_shade.jpeg", pixelArr, height, width);

    printf("Calculating shade...\n");
    auto start = std::chrono::high_resolution_clock::now();
    float* shadeArr = calculateShade(pixelArr, height, width, cellSize);
    auto end = std::chrono::high_resolution_clock::now();

    printf("Applying shade...\n");
    applyShade(pixelArr, shadeArr, height, width);

    printf("Saving shaded tiff...\n");
    saveGeoTIFF(tiffFileName, pixelArr, height, width);

    printf("Saving preview...\n");
    saveToJPEG("post_shade.jpeg", pixelArr, height, width);

    printf("Done!\n");
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    saveTimeToFile("cuda.time", std::to_string(duration.count()));
    printf("Shade function time: %d ms\n", int(duration.count()));

    return 0;
}
