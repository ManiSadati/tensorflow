#ifndef TFLITE_FAULT_INJECTION_H_
#define TFLITE_FAULT_INJECTION_H_
#include <fstream>
#include <random>
#include <string>
#include <iostream>
#include <random>
#include <set> 
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include <filesystem> 
namespace fs = std::filesystem; 

enum class FIMode { Profiling = 0 , Injection = 1 };


struct FaultInjection {
    FIMode mode;
    std::string layer_name;
    std::set <std::pair<std::pair<int, std::pair<int, int> >, int> > injectLocations; // {{c,(x,y)},bit}
    int layer_num = 0;
    bool is_layer_valid = false;

    void init(std::string layer_name_) {
        layer_name = layer_name_;
        layer_num = read_layer_num();

        // This file reads whether program is in profiling mode or injection mode
        std::ifstream mode_file("./fi/mode.txt");
        std::string mode_string;
        int num;
        mode_file >> mode_string >> num;
        if (num == layer_num)
            is_layer_valid = true;
        
        if (mode_string == "profiling") {
            mode = FIMode::Profiling;
        }
        else {
            mode = FIMode::Injection;
        }
        if(mode == FIMode::Injection){
            /* 
            The first line of this file would be the layer to be injected.
            The rest of the file contains lines of 4 integers, c, x, y, bit.
            c is the output channel, x and y are the location of the element 
            needs to be injected, and bit is the injected bit. 
            */
            std::ifstream count_file("./fi/locations.txt");
            int x, y, c, bit;
            while (count_file >> c >> x >> y >> bit) {
                injectLocations.insert({{c, {x, y}}, bit});
            }
            count_file.close();
        }
    }

    int read_layer_num(){
        // This file reads which layer of the model we are processing
        std::ifstream layer_num_file("./fi/layer_num.txt");
        if (layer_num_file.is_open()) {
            layer_num_file >> layer_num;
            layer_num_file.close();
        } 

        layer_num;

        std::ofstream layer_num_file_out("./fi/layer_num.txt");
        if (layer_num_file_out.is_open()) {
            layer_num_file_out << layer_num + 1;
            layer_num_file_out.close();
        }
        return layer_num;
    }

    void save_profile(int c_dim, int x_dim, int y_dim, int numOps) {
        if (mode == FIMode::Profiling && is_layer_valid){

            TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "prof %d (%d, %d, %d) %d", layer_num, c_dim, x_dim, y_dim, numOps);
            // This file saves the dimensions of the output matrix for the current layer
            std::ofstream count_file("./fi/dimension.txt", std::ios::app);
            count_file << layer_name << " " << layer_num << " " << c_dim << " " << x_dim << " " << y_dim << " " << numOps << "\n";
            count_file.close();
        }
    }

    bool isFaultyLayer(){
        return (is_layer_valid && mode == FIMode::Injection);
    }
    
    int doFaultInjection(const int value, const std::pair<std::pair<int, std::pair<int, int> >, int>& InjectionLoc) {
        // int new_value = (fi_bit == 7)? (-value) : (value ^ (int)(1<<fi_bit));
        int new_value = value ^ (1 << InjectionLoc.second); 
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Injecting to loc <%d, %d, %d> bit %d: (%d -> %d) \n",
                        InjectionLoc.first.first, InjectionLoc.first.second.first, InjectionLoc.first.second.second, InjectionLoc.second, value, new_value);
        return new_value;
    }
};

#endif  // TFLITE_FAULT_INJECTION_H_
