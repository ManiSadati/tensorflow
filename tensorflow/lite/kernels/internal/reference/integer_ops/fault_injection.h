#ifndef TFLITE_FAULT_INJECTION_H_
#define TFLITE_FAULT_INJECTION_H_
#include <fstream>
#include <random>
#include <string>
#include <iostream>
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/minimal_logging.h"

enum class FIMode { Profiling = 0 , Injection = 1 };


struct FaultInjection {
    FIMode mode;
    int current_count;
    int trigger_count;

    void init() {
        std::ifstream mode_file("/tmp/tflite_fi/fi_mode.txt");
        std::string mode_string;
        std::getline(mode_file, mode_string);
        mode_file.close();
        current_count = 0;
        
        if (mode_string == "profiling") {
            mode = FIMode::Profiling;
        }
        else {
            mode = FIMode::Injection;
            std::ifstream count_file("/tmp/tflite_fi/fi_count.txt");
            int total;
            count_file >> total;
            count_file.close();
            trigger_count = 3;
        }
    }  
    void save_profile() {
        if (mode == FIMode::Profiling){
            std::ofstream count_file("/tmp/tflite_fi/fi_count.txt");
            count_file << current_count;
            count_file.close();
        }
    }
    
    template <typename T>
    T doFaultInjection(const T& value) {
        // std::cout<<" HERE? "<<(int) mode<<" "<<trigger_count<<" / "<<current_count<<" "<<std::endl;
        if (mode == FIMode::Profiling) {
            current_count++;
            return value;
        } else {
            if (current_count == trigger_count) {
                TFLITE_LOG_PROD(tflite::TFLITE_LOG_INFO, "Injecting a Fault! (%d / %d) \n",trigger_count,current_count);
                current_count++;
                return !value; 
            }
            current_count++;
            return value;
        }
    }
};

#endif  // TFLITE_FAULT_INJECTION_H_
