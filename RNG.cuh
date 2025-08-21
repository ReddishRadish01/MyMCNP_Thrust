#pragma once
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Constants.cuh"

#include "Neutron.cuh"
#include "XSParser.cuh"

//#include "FuelKernel.cuh" <- omitted, this causes serious issues Include issues, related to the #pragma once header. this fucks everythig up.
// the only part that i want to use in FuelKernel.cuh was the enum FissonableElementType. This is moved to Constants.cuh for this reason.



struct McnpAMCM {
	unsigned long long m_xi;

	McnpAMCM(unsigned long long xi)
		: m_xi(xi)
	{}
	~McnpAMCM() {}

	__host__ __device__ unsigned long long gen();
};

struct GnuAMCM {
	unsigned long long m_xi;

	__host__ __device__ GnuAMCM(unsigned long long xi)
		: m_xi(xi)
	{
		if (m_xi % 2 == 0) {
			//printf("Seed Value must be a odd number\n");
			m_xi++;
		}
	}
	__host__ __device__ ~GnuAMCM() {}

	__host__ __device__ unsigned long long gen();
	__host__ __device__ double uniform(double lowerLimit, double upperLimit);
	__host__ __device__ double uniform_open(double lowerLimit, double upperLimit); 
	__host__ __device__ int int_dist(int lower, int upper);

	__host__ __device__ double GnuAMCM::MaxwellDistSample(double a);
	__host__ __device__ double GnuAMCM::WattDistSample(double a = 1, double b = 2);
	__host__ __device__ int GnuAMCM::fissionNeutronNumber(FissionableElementType fissionElement);
	__host__ __device__ double GnuAMCM::GaussianPDF(double inputX, double mean, double stdev);
	__host__ __device__ double GnuAMCM::GaussianCDF(double inputX);

};