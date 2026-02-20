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

/*
 * FaultInjection is a lightweight helper used inside TFLite kernels.
 *
 * Contract / Mental model:
 *  - Python controls runs by writing files under ./fi/
 *  - layer_num.txt is a global "op counter" incremented once per kernel call.
 *    Each kernel invocation constructs FaultInjection and calls init(), which:
 *      1) reads current layer index from layer_num.txt
 *      2) increments layer_num.txt (so next op sees next index)
 *  - mode.txt determines whether we are profiling or injecting and which layer index is targeted.
 *  - locations.txt lists injection coordinates for the targeted layer.
 */
struct FaultInjection {
    FIMode mode = FIMode::Profiling;
    std::string layer_name;

    // injectLocations stores the tensor elements to inject faults.
    // Each location is: {{c, {x, y}}, bit}
    // Interpretation of (c, x, y) depends on the op (e.g., for FC you may map c=1).
    std::set <std::pair<std::pair<int, std::pair<int, int> >, int> > injectLocations;
      
    // Derived state for this kernel invocation
    int current_layer_num = 0; // The "op index" for this invocation
    int fault_layer, img_index, iteration;
    std::string fault_type;

    bool is_layer_valid = false; // true if current_layer_num == fault_layer
    bool is_layer_logged = false; // true if current_layer_num appears in layer_output_list.txt

    
    void init(std::string layer_name_) {
        layer_name = layer_name_;

        // Determine this kernel's "op index"
        current_layer_num = read_layer_num();

        // --- Read mode.txt ---
        // Expected format:
        //   profiling <fault_layer> <img_index> None <iteration>
        //   injection <fault_layer> <img_index> <fault_type> <iteration>
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
            * locations.txt format: each line "c x y bit"
            * c: channel index (or 0 for 2D tensors)
            * x,y: coordinates within output tensor
            * bit: bit position to flip (0..N-1)
            */
            std::ifstream count_file("./fi/locations.txt");
            int x, y, c, bit;
            while (count_file >> c >> x >> y >> bit) {
                injectLocations.insert({{c, {x, y}}, bit});
            }
            count_file.close();
        }
    }

    // read layer counter and print the new value.
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

    // Save profiling data for the *target* layer only.
    void save_profile(int c_dim, int x_dim, int y_dim, int numOps) {
        if (mode == FIMode::Profiling && is_layer_valid){

            TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "      Profiling Layer %d (%d, %d, %d) %d", current_layer_num, c_dim, x_dim, y_dim, numOps);
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

    // Flip a specific bit of an integer output element.
    // InjectionLoc = {{c,(x,y)},bit}
    int doFaultInjection(const int value, const std::pair<std::pair<int, std::pair<int, int> >, int>& InjectionLoc) {
        // int new_value = (fi_bit == 7)? (-value) : (value ^ (int)(1<<fi_bit));
        int new_value = value ^ (1 << InjectionLoc.second); 
        TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "      Injecting to loc <%d, %d, %d> bit %d: (%d -> %d) \n",
                        InjectionLoc.first.first, InjectionLoc.first.second.first, InjectionLoc.first.second.second, InjectionLoc.second, value, new_value);
        return new_value;
    }

    // Opens canonical FI output file:
    // ./fi/output_<currentLayer>-<faultLayer>-<img>-<type>-<iter>.txt
    std::ofstream open_output_file() const {
        return std::ofstream("./fi/output_" + std::to_string(current_layer_num) + "-" +
                                std::to_string(fault_layer) + "-" +
                                std::to_string(img_index) + "-" + fault_type + "-" +
                                std::to_string(iteration) + ".txt",
                            std::ios::app);
    }

    // Writes header required by Python reader: "<c_dim> <x_dim> <y_dim>\n"
    static void write_header(std::ofstream& out, int c_dim, int x_dim, int y_dim) {
        out << c_dim << " " << x_dim << " " << y_dim << "\n";
    }
};

#endif  // TFLITE_FAULT_INJECTION_H_
