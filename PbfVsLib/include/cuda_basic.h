#ifndef cuda_basic_h
#define cuda_basic_h

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

template<typename T>
void check(T err, const char* const func, const char* const file, 
	const int line) {
	if (err != cudaSuccess) {
		std::stringstream ss;
		ss << "CUDA error at: " << file << ":" << line << std::endl
			<< cudaGetErrorString(err) << " " << func;
		std::cerr << ss.str() << std::endl;
		throw std::runtime_error(ss.str().c_str());
		exit(1);
	}
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
#define CUDA_CALLABLE __host__ __device__
#define kNumThreadPerBlock ((int)512)	

#endif // cuda_basic_h