
#include "clutil.h"
#include "fft.h"
#include "kernels.h"
#include<fstream>
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
            deviceCount = val;

        } else if (!strcmp(config, "BLOCK_SIZE")) {
            if (val == 0) {
                cout << "Block size cannot be zero" << endl;
            }   
            blockSize = val;
        } else if (!strcmp(config, "FFT_ALGO")) {
            if (val > 5) {
                cout << "FFT_ALGO config should be from 1 to 5." << endl;
                return false;
            }
            fftAlgo = val;
        }  else if (!strcmp(config, "PRINT_RESULT")) {
            print = val;
        }   else {
            cout << "Invalid config " << config << endl;
            return false;
        } 
    }
    if (deviceCount == 0 && useCpu == 0) {
        cout << "CPU and GPU count are both 0" << endl;
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
    cout << "PRINT_RESULT " << print << endl;
    cout << "FFT ALGO "<< fftAlgo << endl;

    bool result = true;
    if (fftAlgo == SLOW_FFT) {
        result = runSlowFFT(argv, sampleSize, inputSize);
    } else if (fftAlgo == COOLEY_TUKEY) {
        result = runCooleyTukey(argv, sampleSize, inputSize);
    } else if (fftAlgo == STOCKHALM) {
        result = runStockhamFFT(argv, sampleSize, inputSize);
    } else if (fftAlgo == SANDE_TOOKEY) {
      result = runSande_tookeyFFT(argv, sampleSize, inputSize);
    }else if (fftAlgo == BLUESTEINS) {
      cout<< "At Bluesteins"<<endl;  
      result = runBluesteinsFFT(argv, sampleSize, inputSize);
    }

    else {
        cout << "Wrong FFT_ALGO config" << endl;
        result = false;
    }
    if (print) {
        printResult(inputSize);
    }
    cleanup();
    return 1;
}




