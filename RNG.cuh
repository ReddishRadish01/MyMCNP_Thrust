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

	__host__ __device__ inline unsigned long long gen() {		// returns value between 0 and 2^48-1
		// You can use pow(2,48) but our main concern is SPEED!!! use bitshift to get 2^48:  shift value 1 of type ULL(unsigned long long) to the left 48 times: 1ULL<<48
		//unsigned long long xi_nplus1 = (m_xi * 25214903917 + 11) % static_cast<unsigned long long>(pow(2, 48));
		//unsigned long long xi_nplus1 = (m_xi * 25214903917ULL + 11ULL) % (1ULL << 48);
		unsigned long long xi_nplus1 = (m_xi * 25214903917ULL + 11ULL) & (0xFFFFFFFFFFFFULL);	// explicit and
		m_xi = xi_nplus1;
		return xi_nplus1;
	}
	__host__ __device__ double uniform(double lowerLimit, double upperLimit);
	__host__ __device__ double uniform_open(double lowerLimit, double upperLimit); 
	__host__ __device__ int int_dist(int lower, int upper);

	__host__ __device__ double MaxwellDistSample(double a);
	__host__ __device__ double WattDistSample(double a = 1, double b = 2);
	__host__ __device__ int fissionNeutronNumber(FissionableElementType fissionElement);
	__host__ __device__ double GaussianPDF(double inputX, double mean, double stdev);
	__host__ __device__ double GaussianCDF(double inputX);

};