#include <iostream>
#include <gdal.h>
#include <gdal_priv.h>
#include <jpeglib.h>
#include <cmath>
#include <chrono>

typedef unsigned int uint;
typedef unsigned char uchar;

// Calculate shade on a vector of vectors of pixels
float* calculateShade(float* &pixelArray, const uint height, const uint width, int azimuth=315, int altitude=45) {
    // Vector for shaded map
    float* shadeArr = new float[(height - 2) * (width - 2)];
    int zenithDegree = 90 - altitude;
    float zenithRadian = float(zenithDegree) * (M_PI / 180);
    int azimuthMath = (360 - azimuth + 90) % 360;
    float azimuthRadian = float(azimuthMath) * (M_PI / 180);

    // Cell size for shading - "real" pixel size -> bigger = less accurate shading
    int cellSize = 5;
    // Z factor - height exaggeration -> bigger = more intense shading
    int zFactor = 1;
    // Iterate over all the pixels
    for(int i=1; i < height - 1; ++i) {
        // Print current row number
        for(int j=1; j < width - 1; ++j) {
            // Change in height in x-axis
            float changeX = (
                (
                    pixelArray[((i-1)*width) + j + 1] + (2 * pixelArray[(i*width) + j + 1]) + pixelArray[((i+1)*width) + j + 1]
                ) - (
                    pixelArray[((i-1)*width) + j - 1] + (2 * pixelArray[(i*width) + j - 1]) + pixelArray[((i+1)*width) + j - 1]
                )
            ) / (8 * cellSize);
            // Change in height in y-axis
            float changeY = (
                (
                    pixelArray[((i+1)*width) + j - 1] + (2 * pixelArray[((i+1)*width) + j]) + pixelArray[((i+1)*width) + j + 1]
                ) - (
                    pixelArray[((i-1)*width) + j - 1] + (2 * pixelArray[((i-1)*width) + j]) + pixelArray[((i-1)*width) + j + 1]
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
            shadeArr[((i-1)*(width-2)) + j - 1] = hillShade;
        }
    }
    return shadeArr;
}

// Load GeoTIFF file
float* loadGeoTIFF(const std::string &filename, uint &height, uint &width) {
    GDALAllRegister();
    // Path for input data folder
    std::string path = "../DATA/IN/" + filename;
    // Pixel data vector
    float* elevationData;

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

        float pixelHeight;
        // Read and save to vector
        for (int y = 0; y < nYSize; y++) {
            for (int x = 0; x < nXSize; x++) {
                // Get pixel value from dataset
                CPLErr err = poDataset->GetRasterBand(1)->RasterIO(GF_Read, x, y, 1, 1, &pixelHeight, 1, 1, GDT_Float32, 0,0);

                // Check for error
                if (err == CE_None) {
                    elevationData[y*width + x] = pixelHeight;
                } else {
                    readFailure = true;
                    std::cerr<<"Pixel value read failure: ("<<x<<", "<<y<<")"<<std::endl;
                    break;
                }
            }
        }

        // Close dataset
        GDALClose(poDataset);
    } else {
        std::cerr<<"Can't open GeoTIFF: "<<path<<std::endl;
    }
    GDALDestroyDriverManager();

    // If there was an error with reading
    if(readFailure) {
        delete[] elevationData;
    }

    return elevationData;
}

void applyShade(float* &pixels, float* &shade, const uint height, const uint width) {
    for (int i=1; i < height - 1; ++i) {
        for (int j=1; j < width - 1; ++j) {
            pixels[(i*width) + j] += shade[((i-1)*(width-2)) + j - 1];
        }
    }
}

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

    std::cout<<pixelArr[25000]<<"\n";

    std::cout<<"Calculating shade...\n";
    auto start = std::chrono::high_resolution_clock::now();
    float* shadeArr = calculateShade(pixelArr, height, width);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout<<"Applying shade...\n";
    applyShade(pixelArr, shadeArr, height, width);

    std::cout<<pixelArr[25000]<<"\n";

    std::cout<<"Saving preview...\n";
    saveToJPEG("post_shade.jpeg", pixelArr, height, width);

    std::cout<<"Done!\n";
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout<<"Shade function time: "<<duration.count()<<" ms\n";

    return 0;
}

