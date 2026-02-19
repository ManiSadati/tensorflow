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
    int current_layer_num = 0;
    int fault_layer, img_index, iteration;
    std::string fault_type;
    bool is_layer_valid = false;
    bool is_layer_logged = false;

    void init(std::string layer_name_) {
        layer_name = layer_name_;
        current_layer_num = read_layer_num();

        // This file reads whether program is in profiling mode or injection mode
        std::ifstream mode_file("./fi/mode.txt");
        std::string mode_string;
        mode_file >> mode_string >> fault_layer >> img_index >> fault_type >> iteration;
        if (fault_layer == current_layer_num)
            is_layer_valid = true;
        mode_file.close();

        // check if the current layer should be logged
        std::ifstream layer_output_file("./fi/layer_output_list.txt");
        int l;
        while(layer_output_file >> l) {
            if(current_layer_num == l) {
                is_layer_logged = true;
                break;
            }
        }
        layer_output_file.close();


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
        // This file is read to determine which layer of the model we are processing
        std::ifstream layer_num_file("./fi/layer_num.txt");
        if (layer_num_file.is_open()) {
            layer_num_file >> current_layer_num;
            layer_num_file.close();
        } 

        // Then, we write the index of the next layer into the file.
        std::ofstream layer_num_file_out("./fi/layer_num.txt");
        if (layer_num_file_out.is_open()) {
            layer_num_file_out << current_layer_num + 1;
            layer_num_file_out.close();
        }
        return current_layer_num;
    }

    void save_profile(int c_dim, int x_dim, int y_dim, int numOps) {
        if (mode == FIMode::Profiling && is_layer_valid){

            TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "prof %d (%d, %d, %d) %d", current_layer_num, c_dim, x_dim, y_dim, numOps);
            // This file saves the dimensions of the output matrix for the current layer
            std::ofstream count_file("./fi/dimension.txt", std::ios::app);
            count_file << layer_name << " " << current_layer_num << " " << c_dim << " " << x_dim << " " << y_dim << " " << numOps << "\n";
            count_file.close();
        }
    }

    bool isFaultyLayer(){
        return (is_layer_valid && mode == FIMode::Injection);
    }

    bool isLoggedLayer() {
        return (is_layer_logged);
    }

    // value = fi_value_to_change, InjectionLoc = {{fi_c,(fi_x,fi_y)},fi_bit}
    int doFaultInjection(const int value, const std::pair<std::pair<int, std::pair<int, int> >, int>& InjectionLoc) {
        // int new_value = (fi_bit == 7)? (-value) : (value ^ (int)(1<<fi_bit));
        int new_value = value ^ (1 << InjectionLoc.second); 
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Injecting to loc <%d, %d, %d> bit %d: (%d -> %d) \n",
                        InjectionLoc.first.first, InjectionLoc.first.second.first, InjectionLoc.first.second.second, InjectionLoc.second, value, new_value);
        return new_value;
    }

    template <typename OutputType>
    void log_layer_output(const OutputType* output_data, int c_dim, int x_dim, int y_dim) {
        std::ofstream output_file("./fi/output_" + std::to_string(current_layer_num) + "-" + std::to_string(fault_layer) + "-" + std::to_string(img_index) + "-" + fault_type + "-" + std::to_string(iteration) + ".txt", std::ios::app);
        for ( int c = 0; c < c_dim; ++c) {
            for (int x = 0; x < x_dim; ++x) {
                for (int y = 0; y < y_dim; ++y) {
                    int index = c * x_dim * y_dim + x * y_dim + y;
                    output_file << output_data[index] << " ";
                }
                output_file << "\n";
            }
        }
        output_file.close();
    }   
};

#endif  // TFLITE_FAULT_INJECTION_H_
