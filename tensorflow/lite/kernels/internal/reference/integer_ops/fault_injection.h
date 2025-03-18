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
    std::set <std::pair<std::pair<int, int>, int> > injectLocations; // {(x,y),bit}
    int layer_num = -1;
    bool is_layer_valid = false;

    void init(std::string layer_name_) {
        layer_name = layer_name_;
        layer_num = read_layer_num();

        // This file reads whether program is in profiling mode or injection mode
        std::ifstream mode_file("./fi_mode.txt");
        std::string mode_string;
        mode_file >> mode_string;
        
        if (mode_string == "profiling") {
            mode = FIMode::Profiling;
        }
        else {
            mode = FIMode::Injection;
        }
        int num;
        while (mode_file >> num) {
            if(layer_num == num){
                is_layer_valid = true;
            }
        }
            
        if(mode == FIMode::Injection){
            /* 
            The first line of this file would be the layer to be injected.
            The rest of the file contains lines of 3 integers, x, y, bit.
            x and y are the location of the element needs to be injected,
            and bit is the injected bit. 
            */
            std::ifstream count_file("./fi_locations.txt");
            int lnum;
            if (count_file >> lnum && layer_num == lnum) {
                int x, y, bit;
                while (count_file >> x >> y >> bit) {
                    injectLocations.insert({{x, y}, bit});
                }
            }
            count_file.close();
        }
    }

    int read_layer_num(){
        // This file reads which layer of the model we are processing
        std::ifstream layer_num_file("./fi_layer_num.txt");
        if (layer_num_file.is_open()) {
            layer_num_file >> layer_num;
            layer_num_file.close();
        } 

        layer_num;

        std::ofstream layer_num_file_out("./fi_layer_num.txt");
        if (layer_num_file_out.is_open()) {
            layer_num_file_out << layer_num + 1;
            layer_num_file_out.close();
        }
        return layer_num;
    }

    void save_profile(int x_dim, int y_dim) {
        if (mode == FIMode::Profiling && is_layer_valid){

            TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "prof %d %d (%d, %d)",is_layer_valid, layer_num, x_dim, y_dim);
            // This file saves the dimensions of the output matrix for the current layer
            std::ofstream count_file("./fi_dimension.txt", std::ios::app);
            count_file << layer_num << " " << x_dim << " " << y_dim << "\n";
            count_file.close();
        }
    }
    
    bool isFaultyLayer(){
        return (is_layer_valid && mode == FIMode::Injection);
    }
    
    int doFaultInjection(const int value, const std::pair<std::pair<int,int>,int>& InjectionLoc) {
        // int new_value = (fi_bit == 7)? (-value) : (value ^ (int)(1<<fi_bit));
        int new_value = value ^ (1 << InjectionLoc.second);  
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Injecting to loc <%d, %d> bit %d: (%d -> %d) \n",
                        InjectionLoc.first.first, InjectionLoc.first.second, InjectionLoc.second, value, new_value);
        return new_value;
    }
};

#endif  // TFLITE_FAULT_INJECTION_H_
