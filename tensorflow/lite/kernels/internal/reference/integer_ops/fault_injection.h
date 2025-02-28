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
    int current_count;
    int trigger_count;
    int fi_bit;
    int layer_num = -1;
    bool is_layer_valid = false;

    void init(std::string layer_name_) {
        current_count = 0;
        layer_name = layer_name_;
        layer_num = read_layer_num();


        std::ifstream mode_file("./fi_mode.txt");
        std::string mode_string;
        mode_file >> mode_string >> fi_bit;
        
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
            std::ifstream count_file("./fi_count.txt");
            int lnum, tmp, total;
            while(count_file >> lnum >> tmp){
                if (lnum == layer_num){
                    total = tmp;
                }
            }
            count_file.close();

            std::random_device rd; 
            std::mt19937 gen(rd()); 
            std::uniform_int_distribution<> distrib(0, total);

            trigger_count = distrib(gen);
        }
    }

    int read_layer_num(){
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

    void save_profile() {
        if (mode == FIMode::Profiling && is_layer_valid){

            TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "prof %d %d %d",is_layer_valid, layer_num, current_count);
            std::ofstream count_file("./fi_count.txt", std::ios::app);
            count_file << layer_num << " " << current_count << "\n";
            count_file.close();
        }
    }
    
    int doFaultInjection(const int value) {
        if (!is_layer_valid)
            return value;
        if (mode == FIMode::Profiling) {
            current_count++;
            return value;
        } else {
            if (current_count == trigger_count) {
                // int new_value = (fi_bit == 7)? (-value) : (value ^ (int)(1<<fi_bit));
                int new_value = value ^ (int)(1<<fi_bit);

                TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Injecting a Fault! (counter %d) (%d -> %d) \n",trigger_count,value, new_value);
                current_count++;
                return new_value; 
            }
            current_count++;
            return value;
        }
    }
};

#endif  // TFLITE_FAULT_INJECTION_H_
