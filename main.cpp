
#include "clutil.h"
#include "fft.h"
#include "kernels.h"
#include<fstream>
#include<iostream>
#include<string.h>
using namespace std;

bool
readConfig(const char* const fName)
{
    ifstream file(fName);
    if (!file.is_open()) {
        cout << "Config file cannot be opened" << endl;
        return false;
    }
    cout << "Reading from config file....." << endl;
    while (!file.eof()) {
        string line;
        getline(file, line);
        if (line[0] == '\0') break;
        if (line[0] == '#') continue; 
        char config[10];
        unsigned val = 0;
        sscanf(line.c_str(), "%s %d", config, &val);
        if (!strcmp(config, "USE_CPU")) {
            useCpu = val;
        } else if (!strcmp(config, "USE_GPU")) {
            if (val > MAX_GPU_COUNT) {
                cout << "Invalid GPU COUNT" << endl;
                return false;  
            }
            if (val > 0) {
                deviceCount = val;
            } 

        } else if (!strcmp(config, "BLOCK_SIZE")) {
            if (val == 0) {
                cout << "Block size cannot be zero" << endl;
            }   
            blockSize = val;
        } else if (!strcmp(config, "FFT_ALGO")) {
            if (val > 3) {
                cout << "FFT_ALGO config should be 1, 2 or 3" << endl;
                return false;
            }
            fftAlgo = val;
        }  else {
            cout << "Invalid config" << endl;
            return false;
        } 
    }
    return true;
}

int 
main(const int argc, const char* argv[])
{
    
    if (argc < 3) {
        cout << "Usage: clfft inputSize, sampleSize {config_file_name}" << endl;
	return 0;
    }

    const unsigned inputSize = atoi(argv[1]);
    const unsigned sampleSize = atoi(argv[2]);
    
    const char* const fName = (argc == 4) ? argv[3] : 0;
    
    if (sampleSize > inputSize) {
        cout << "sample size cannot be greater than input size" << endl;
        return 0; 

    } 

    if (fName) {
         if (!readConfig(fName)) {
              cout << "Error in config file abort" << endl;  
              return 0;
         }
    } else {
         cout << "Config file not specified. Executing with default config" << endl;
    }

    cout << "USE_CPU " << useCpu << endl;
    cout << "USE_GPU " << deviceCount << endl;
    cout << "BLOCK_SIZE " << blockSize << endl;
   

    //TODO:: move all these to cutils.cpp
    // Allocate host memory. 
    allocateHostMemory(inputSize);

    
    printf("Initializing CL Context..\n");
    // create the OpenCL context on available GPU devices
    init_cl_context(CL_DEVICE_TYPE_GPU);

    printf("Getting Device Count..\n");
    const cl_uint ciDeviceCount =  getDeviceCount();


    if (!ciDeviceCount) {
        printf("No opencl specific devices!\n");
        return 0;
    }

    printf("Creating Command Queue...\n");
    // create a command queue on device 1
    for (unsigned i = 0; i < deviceCount; ++i) {
        createCommandQueue(i);
    }

    if (fftAlgo == SLOW_FFT) {
        runSlowFFT(argv, sampleSize, inputSize);
    } else if (fftAlgo == COOLEY_TUKEY) {
        cooleyTukey(argv, sampleSize, inputSize);
    } else if (fftAlgo == STOCKHALM) {
        stockhamFFT(argv, sampleSize, inputSize);
    } else {
        cout << "Wrong FFT_ALGO config" << endl;
    }

    for (unsigned i = 0; i < deviceCount; ++i) {
        printf("Kernel execution time on GPU %d: %.9f s\n", i, executionTime(i));
    }
    cleanup();
    return 1;
}




