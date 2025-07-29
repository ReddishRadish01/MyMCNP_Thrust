#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <random>

//#include "thrust/device_vector.h"
// This Thrust thing is C++ implementation on GPU - Worth looking up later.


//#include "common.cuh"
#include "Neutron.cuh"
#include "RNG.cuh"
#include "XSParser.cuh"
#include "FuelKernel.cuh"
#include "Constants.cuh"
#include "thrustHeader.cuh"



#ifdef _WIN32
#define TimeDivider 1
#elif __linux__
#define TimeDivider 1000
#else
#define TimeDivider 1
#endif


// currently have a base of 500000 neutrons - these numbers should be a divisor of it.(It doesnt really matter tho it will get truncated down anyways)
int ratioDivider = 200;
__device__ int d_ratioDivider = 200;



__global__ void AddCrossSection(RawCrossSection* rawXS_ptr_1, RawCrossSection* rawXS_ptr_2, double* incidentEnergy, double* value, int count) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < count) {
		//printf("this works!\n");
		value[idx] = getCrossSectionTot(rawXS_ptr_1, incidentEnergy[idx]) + getCrossSectionTot(rawXS_ptr_2, incidentEnergy[idx]);
		printf("%f, %f\n", getCrossSectionTot(rawXS_ptr_1, incidentEnergy[idx]), getCrossSectionTot(rawXS_ptr_2, incidentEnergy[idx]));
	}
}


__global__ void Test(RawCrossSection* rawXS_ptr, double* incidentEnergy, double* value, int count) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < count) {
		printf("%f\n", rawXS_ptr->nf_XS[idx]);
		//value[idx] = getCrossSectionF(rawXS_ptr, incidentEnergy[idx]);
		//value[idx] = rawXS_ptr->getCrossSectionByEnergy(incidentEnergy[idx]);
		value[idx] = rawXS_ptr->getTotalMicroXSByEnergy(0.0253);
	}
}

__global__ void NeutronNullifier(NeutronDistribution* Neutrons, int count) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx == 1) printf("Nullifying all neutrons...\n");
	if (idx < count) {
		//printf("(%f, %f, %f)\n", Neutrons->m_initialNeutron[idx].m_pos.x, Neutrons->m_initialNeutron[idx].m_pos.y, Neutrons->m_initialNeutron[idx].m_pos.z);
		Neutrons->m_initialNeutron[idx].Nullify();
	}
}

__global__ void Step(NeutronDistribution* Neutrons, unsigned int numNeutrons, double time) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeutrons) {
	}

}

__global__ void ThrustTest(BareSphere* CP1, NeutronThrustDevice* d_Neutrons, RawCrossSection* U235XS, RawCrossSection* U238XS,
	RawCrossSection* O16XS, RawCrossSection* C12XS, unsigned int numNeutrons, unsigned long long* seedNo, double* distances, int* counter) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeutrons) {
		if (d_Neutrons->m_neutrons[idx].isNullified() == false) {
			double totalCrossSection = CP1->getTotalMacroXS(d_Neutrons->m_neutrons[idx], U235XS, U238XS, O16XS, C12XS);
			GnuAMCM RNG(seedNo[idx]);
			seedNo[idx] = RNG.gen();
			distances[idx] = -log(RNG.uniform(0.0, 1.0)) / (totalCrossSection * 100);
			d_Neutrons->m_neutrons[idx].UpdateWithLength(distances[idx]);
		}
		else {
			seedNo[idx] = (seedNo[idx] * 25214903917ULL + 11ULL) % (1ULL << 48);
			distances[idx] = 0;
		}

		if (d_Neutrons->m_neutrons[idx].OutofBounds(CP1->m_radius) == true) {
			d_Neutrons->m_neutrons[idx].Nullify();
			atomicAdd(counter, 1);
		}

		if (idx < numNeutrons / d_ratioDivider) {
			d_Neutrons->m_addedNeutrons[idx] = d_Neutrons->m_neutrons[idx];
		}
	}
}





/*
 *  __  __     _   _            _     _                    ___            ___                  _ _               _  _         _        ___     _ _ _    _              ___  _    _
 * |  \/  |___| |_| |_  ___  __| |___| |___  __ _ _  _    | __|__ _ _    / __| __ _ _ __  _ __| (_)_ _  __ _    | \| |_____ _| |_     / __|___| | (_)__(_)___ _ _     |   \(_)__| |_ __ _ _ _  __ ___
 * | |\/| / -_)  _| ' \/ _ \/ _` / _ \ / _ \/ _` | || |   | _/ _ \ '_|   \__ \/ _` | '  \| '_ \ | | ' \/ _` |   | .` / -_) \ /  _|   | (__/ _ \ | | (_-< / _ \ ' \    | |) | (_-<  _/ _` | ' \/ _/ -_)
 * |_|  |_\___|\__|_||_\___/\__,_\___/_\___/\__, |\_, |   |_|\___/_|     |___/\__,_|_|_|_| .__/_|_|_||_\__, |   |_|\_\___/_\_\\__|    \___\___/_|_|_/__/_\___/_||_|   |___/|_/__/\__\__,_|_||_\__\___|
 *                                          |___/ |__/                                   |_|           |___/
 */
 /*
 * maths: the PDF for the distance to its next collision \textit{l} is
 *   p(l)dl = \Sigma_{t} e^{-\Sigma_{t} l} dl
 * we convert the PDF to Cumulative Distribution Function(CDF)
 *   \int_{0}^{l} dl' p(l') = \int_{0}^{l} dl' \Sigma_t e^{-Sigma_t l'} = 1 - e^{-\Sigma_t l}
 * let /xi be the result of the CDF, which is between [0, 1):
 *   1 - e^{-\Sigma_t l} = \xi \srightarrow l = -\frac{\ln(1-\xi)}{\Sigma_t}
 * since \xi is equally distributed in [0,1), 1-\xi will also be [0,1). Therefore:
 *   l = -\frac{\ln(\xi)}{\Sigma_t}
 */


int main() {
	std::cout << "Hello world!\n\n";

	/*
	 *  ___ ___ _____ _____ ___ _  _  ___     _   _ ___     _  _ ___ _   _ _____ ___  ___  _  _ ___
	 * / __| __|_   _|_   _|_ _| \| |/ __|   | | | | _ \   | \| | __| | | |_   _| _ \/ _ \| \| / __|
	 * \__ \ _|  | |   | |  | || .` | (_ |   | |_| |  _/   | .` | _|| |_| | | | |   / (_) | .` \__ \
	 * |___/___| |_|   |_| |___|_|\_|\___|    \___/|_|     |_|\_|___|\___/  |_| |_|_\\___/|_|\_|___/
	 *
	 */
	int numNeutrons = 500000;
	unsigned long long seedNo = 2001;

	//std::cout << std::fixed << std::setprecision(10);
	std::cout << std::fixed << std::scientific;

	NeutronDistribution h_Neutrons(numNeutrons, seedNo, SpectrumType::default);
	h_Neutrons.uniformSpherical(3.048, 2e+6);  // 
	NeutronDistribution* d_Neutrons = nullptr;
	cudaMalloc(&d_Neutrons, sizeof(NeutronDistribution));

	// -- device buffer ararys for struct member neutron arrays
	Neutron* d_bufferInitialNeutron = nullptr;
	Neutron* d_bufferAddedNeutron = nullptr;

	// alloc and copy the host's struct member arrays(type of Neutron)
	// to device buffer arrays
	cudaMalloc(&d_bufferInitialNeutron, h_Neutrons.m_initialNeutronNumber * sizeof(Neutron));
	cudaMalloc(&d_bufferAddedNeutron, h_Neutrons.m_initialNeutronNumber * sizeof(Neutron));
	cudaMemcpy(d_bufferInitialNeutron, h_Neutrons.m_initialNeutron, h_Neutrons.m_initialNeutronNumber * sizeof(Neutron), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bufferAddedNeutron, h_Neutrons.m_addedNeutron, h_Neutrons.m_initialNeutronNumber * sizeof(Neutron), cudaMemcpyHostToDevice);

	// build a temporary host-side copy of object, where the device array buffers will reside in.
	NeutronDistribution tmp_Neutrons = h_Neutrons;
	tmp_Neutrons.m_initialNeutron = d_bufferInitialNeutron;
	tmp_Neutrons.m_addedNeutron = d_bufferAddedNeutron;

	// copy the temporary object containing device array buffers to actual device object
	cudaMemcpy(d_Neutrons, &tmp_Neutrons, sizeof(NeutronDistribution), cudaMemcpyHostToDevice);
	// that'll do for the neutrons.






	// THis is Thrust TEST:
	NeutronThrustHost h_NeutronThrust(numNeutrons, seedNo, SpectrumType::default);
	h_NeutronThrust.uniformSpherical(3.048, 2e+6);

	thrust::device_vector<Neutron> d_NeutronVector = h_NeutronThrust.m_neutrons;
	thrust::device_vector<Neutron> d_addedNeutronVector = h_NeutronThrust.m_addedNeutrons;
	NeutronThrustDevice DeviceCopy = h_NeutronThrust.HtoD(d_NeutronVector, d_addedNeutronVector);
	NeutronThrustDevice* d_NeutronThrust = nullptr;
	cudaMalloc(&d_NeutronThrust, sizeof(NeutronThrustDevice));
	cudaMemcpy(d_NeutronThrust, &DeviceCopy, sizeof(NeutronThrustDevice), cudaMemcpyHostToDevice);




/*
 *  ___ ___ _____ _____ ___ _  _  ___     _   _ ___     _____ _  _ ___     ___ _   _ ___ _
 * / __| __|_   _|_   _|_ _| \| |/ __|   | | | | _ \   |_   _| || | __|   | __| | | | __| |
 * \__ \ _|  | |   | |  | || .` | (_ |   | |_| |  _/     | | | __ | _|    | _|| |_| | _|| |__
 * |___/___| |_|   |_| |___|_|\_|\___|    \___/|_|       |_| |_||_|___|   |_|  \___/|___|____|
 *
 */

 // Composition of Chicago Pile - 1 
 // 6 short ton of U metal , 50 short ton of UO2, 400 short ton of Graphite.
 // radius = ~ 10 ft. (3.048 m)

	BareSphere h_ChicagoPile1(3.048, FissionableElementType::U235, 19.5, 50.0, ModeratorType::Graphite, (100.0 - 50.0));
	BareSphere* d_ChicagoPile1;

	cudaMalloc(&d_ChicagoPile1, sizeof(BareSphere));
	cudaMemcpy(d_ChicagoPile1, &h_ChicagoPile1, sizeof(BareSphere), cudaMemcpyHostToDevice);




	//std::cout << h_O16RawXS.getCrossSectionByEnergy(20.0) << "\n\n";

	GnuAMCM RNG(seedNo);
	// ** Will using "Linear" Additive Multiplicative Congrugental Method (AMCM) works in this parallelized task?
	GnuAMCM h_RNG(seedNo);
	GnuAMCM* d_RNG = nullptr;
	cudaMalloc(&d_RNG, sizeof(GnuAMCM));

	cudaMemcpy(d_RNG, &h_RNG, sizeof(GnuAMCM), cudaMemcpyHostToDevice);
	unsigned long long* h_Seed = new unsigned long long[numNeutrons];
	unsigned long long* d_Seed = nullptr;
	cudaMalloc(&d_Seed, numNeutrons * sizeof(unsigned long long));
	for (int i = 0; i < numNeutrons; i++) {
		h_Seed[i] = RNG.gen();
	}
	cudaMemcpy(d_Seed, h_Seed, numNeutrons * sizeof(unsigned long long), cudaMemcpyHostToDevice);



	double* h_randomDist = new double[numNeutrons];
	double* d_randomDist = nullptr;


	for (int i = 0; i < numNeutrons; i++) {
		//h_randomDist[i] = RNG.gen() / (pow(2, 48) - 1) * h_O16RawXS.ntot_energy[h_O16RawXS.ntot_size - 1];
		//forces the value to be between 0 and max energy value
		h_randomDist[i] = 0.0253 + i * 0.00001;
	}

	cudaMalloc(&d_randomDist, numNeutrons * sizeof(double));
	cudaMemcpy(d_randomDist, h_randomDist, numNeutrons * sizeof(double), cudaMemcpyHostToDevice);

	double* h_value = new double[numNeutrons];
	double* d_value = nullptr;
	cudaMalloc(&d_value, numNeutrons * sizeof(double));

	double* h_return = new double[numNeutrons];
	double* d_return = nullptr;
	cudaMalloc(&d_return, numNeutrons * sizeof(double));

	int h_int = 0;
	int* d_int = nullptr;
	cudaMalloc(&d_int, sizeof(int));
	cudaMemcpy(d_int, &h_int, sizeof(int), cudaMemcpyHostToDevice);




	/*
	 *   ___ ___  ___  ___ ___     ___ ___ ___ _____ ___ ___  _  _     ___ _  _ ___ _____ ___   _   _    ___ ____  _   ___ ___  _  _
	 *  / __| _ \/ _ \/ __/ __|   / __| __/ __|_   _|_ _/ _ \| \| |   |_ _| \| |_ _|_   _|_ _| /_\ | |  |_ _|_  / /_\ |_ _/ _ \| \| |
	 * | (__|   / (_) \__ \__ \   \__ \ _| (__  | |  | | (_) | .` |    | || .` || |  | |  | | / _ \| |__ | | / / / _ \ | | (_) | .` |
	 *  \___|_|_\\___/|___/___/   |___/___\___| |_| |___\___/|_|\_|   |___|_|\_|___| |_| |___/_/ \_\____|___/___/_/ \_\___\___/|_|\_|
	 *
	 */
	 // Initializng host XS and declaring device pointers    //
															 //
	RawCrossSection  h_O16RawXS("o16", Constants::M_O16);   //
	RawCrossSection* d_O16RawXS = nullptr;                  //
	//
	RawCrossSection  h_U235RawXS("u235", Constants::M_U235);//
	RawCrossSection* d_U235RawXS = nullptr;                 //
	//
	RawCrossSection  h_U238RawXS("u238", Constants::M_U238);//
	RawCrossSection* d_U238RawXS = nullptr;                 //
	//
	RawCrossSection  h_C12RawXS("c12", Constants::M_C12);   //
	RawCrossSection* d_C12RawXS = nullptr;                  //
	h_C12RawXS.setDensity(2.0);                             // g/cm^3
	//
	RawCrossSection  h_H1RawXS("h1", Constants::M_H1);      //
	RawCrossSection* d_H1RawXS = nullptr;                   //
	//
// ---------------------------------------------------- //

	double allocStartT = clock();

	// -- Allocating Device CrossSection Structs ------ //
														//
	cudaMalloc(&d_O16RawXS,  sizeof(RawCrossSection));  // 
	cudaMalloc(&d_U235RawXS, sizeof(RawCrossSection));  //

	cudaMalloc(&d_U238RawXS, sizeof(RawCrossSection));  //
	cudaMalloc(&d_C12RawXS,  sizeof(RawCrossSection));  //
	cudaMalloc(&d_H1RawXS,   sizeof(RawCrossSection));  //
	//
// ------------------------------------------------ //

// - Declaring device buffers for Struct's dynamic array members
// - This Process is a "Deep Copy" for GPU.
									   //
	double* d_O16_ntot_energy = nullptr;   //
	double* d_O16_ntot_XS = nullptr;       //
	double* d_O16_nf_energy = nullptr;     //
	double* d_O16_nf_XS = nullptr;         //
	double* d_O16_nel_energy = nullptr;    //
	double* d_O16_nel_XS = nullptr;        //
	double* d_O16_ng_energy = nullptr;     //
	double* d_O16_ng_XS = nullptr;         //
	double* d_O16_ninl_energy = nullptr;   //
	double* d_O16_ninl_XS = nullptr;       //
	double* d_O16_n2n_energy = nullptr;    //
	double* d_O16_n2n_XS = nullptr;        //
	double* d_O16_n3n_energy = nullptr;    //
	double* d_O16_n3n_XS = nullptr;        //
	//
	double* d_U235_ntot_energy = nullptr;  //
	double* d_U235_ntot_XS = nullptr;      //
	double* d_U235_nf_energy = nullptr;    //
	double* d_U235_nf_XS = nullptr;        //
	double* d_U235_nel_energy = nullptr;   //
	double* d_U235_nel_XS = nullptr;       //
	double* d_U235_ng_energy = nullptr;    //
	double* d_U235_ng_XS = nullptr;        //
	double* d_U235_ninl_energy = nullptr;  //
	double* d_U235_ninl_XS = nullptr;      //
	double* d_U235_n2n_energy = nullptr;   //
	double* d_U235_n2n_XS = nullptr;       //
	double* d_U235_n3n_energy = nullptr;   //
	double* d_U235_n3n_XS = nullptr;       //
	//
	double* d_U238_ntot_energy = nullptr;  //
	double* d_U238_ntot_XS = nullptr;      //
	double* d_U238_nf_energy = nullptr;    //
	double* d_U238_nf_XS = nullptr;        //
	double* d_U238_nel_energy = nullptr;   //
	double* d_U238_nel_XS = nullptr;       //
	double* d_U238_ng_energy = nullptr;    //
	double* d_U238_ng_XS = nullptr;        //
	double* d_U238_ninl_energy = nullptr;  //
	double* d_U238_ninl_XS = nullptr;      //
	double* d_U238_n2n_energy = nullptr;   //
	double* d_U238_n2n_XS = nullptr;       //
	double* d_U238_n3n_energy = nullptr;   //
	double* d_U238_n3n_XS = nullptr;       //
	//
	double* d_C12_ntot_energy = nullptr;   //
	double* d_C12_ntot_XS = nullptr;       //
	double* d_C12_nf_energy = nullptr;     //
	double* d_C12_nf_XS = nullptr;         //
	double* d_C12_nel_energy = nullptr;    //
	double* d_C12_nel_XS = nullptr;        //
	double* d_C12_ng_energy = nullptr;     //
	double* d_C12_ng_XS = nullptr;         //
	double* d_C12_ninl_energy = nullptr;   //
	double* d_C12_ninl_XS = nullptr;       //
	double* d_C12_n2n_energy = nullptr;    //
	double* d_C12_n2n_XS = nullptr;        //
	double* d_C12_n3n_energy = nullptr;    //
	double* d_C12_n3n_XS = nullptr;        //
	//
	double* d_H1_ntot_energy = nullptr;    //
	double* d_H1_ntot_XS = nullptr;        //
	double* d_H1_nf_energy = nullptr;      //
	double* d_H1_nf_XS = nullptr;          //
	double* d_H1_nel_energy = nullptr;     //
	double* d_H1_nel_XS = nullptr;         //
	double* d_H1_ng_energy = nullptr;      //
	double* d_H1_ng_XS = nullptr;          //
	double* d_H1_ninl_energy = nullptr;    //
	double* d_H1_ninl_XS = nullptr;        //
	double* d_H1_n2n_energy = nullptr;     //
	double* d_H1_n2n_XS = nullptr;         //
	double* d_H1_n3n_energy = nullptr;     //
	double* d_H1_n3n_XS = nullptr;         //
	//
// ----------------------------------- //


// -- Allocating Each Struct Member's Device Buffer Arrays --------------- //
																		   //
	cudaMalloc(&d_O16_ntot_energy,	h_O16RawXS.ntot_size * sizeof(double));    //
	cudaMalloc(&d_O16_ntot_XS,		h_O16RawXS.ntot_size * sizeof(double));    //
	cudaMalloc(&d_O16_nf_energy,	h_O16RawXS.nf_size * sizeof(double));      //
	cudaMalloc(&d_O16_nf_XS,		h_O16RawXS.nf_size * sizeof(double));      //
	cudaMalloc(&d_O16_nel_energy,	h_O16RawXS.nel_size * sizeof(double));     //
	cudaMalloc(&d_O16_nel_XS,		h_O16RawXS.nel_size * sizeof(double));     //
	cudaMalloc(&d_O16_ng_energy,	h_O16RawXS.ng_size * sizeof(double));      //
	cudaMalloc(&d_O16_ng_XS,		h_O16RawXS.ng_size * sizeof(double));      //
	cudaMalloc(&d_O16_ninl_energy,	h_O16RawXS.ninl_size * sizeof(double));    //
	cudaMalloc(&d_O16_ninl_XS,		h_O16RawXS.ninl_size * sizeof(double));    //
	cudaMalloc(&d_O16_n2n_energy,	h_O16RawXS.n2n_size * sizeof(double));     //
	cudaMalloc(&d_O16_n2n_XS,		h_O16RawXS.n2n_size * sizeof(double));     //
	cudaMalloc(&d_O16_n3n_energy,	h_O16RawXS.n3n_size * sizeof(double));     //
	cudaMalloc(&d_O16_n3n_XS,		h_O16RawXS.n3n_size * sizeof(double));     //
	//
	cudaMalloc(&d_U235_ntot_energy, h_U235RawXS.ntot_size * sizeof(double));   //
	cudaMalloc(&d_U235_ntot_XS,		h_U235RawXS.ntot_size * sizeof(double));   //
	cudaMalloc(&d_U235_nf_energy,	h_U235RawXS.nf_size * sizeof(double));     //
	cudaMalloc(&d_U235_nf_XS,		h_U235RawXS.nf_size * sizeof(double));     //
	cudaMalloc(&d_U235_nel_energy,	h_U235RawXS.nel_size * sizeof(double));    //
	cudaMalloc(&d_U235_nel_XS,		h_U235RawXS.nel_size * sizeof(double));    //
	cudaMalloc(&d_U235_ng_energy,	h_U235RawXS.ng_size * sizeof(double));     //
	cudaMalloc(&d_U235_ng_XS,		h_U235RawXS.ng_size * sizeof(double));     //
	cudaMalloc(&d_U235_ninl_energy, h_U235RawXS.ninl_size * sizeof(double));   //
	cudaMalloc(&d_U235_ninl_XS,		h_U235RawXS.ninl_size * sizeof(double));   //
	cudaMalloc(&d_U235_n2n_energy,	h_U235RawXS.n2n_size * sizeof(double));    //
	cudaMalloc(&d_U235_n2n_XS,		h_U235RawXS.n2n_size * sizeof(double));    //
	cudaMalloc(&d_U235_n3n_energy,	h_U235RawXS.n3n_size * sizeof(double));    //
	cudaMalloc(&d_U235_n3n_XS,		h_U235RawXS.n3n_size * sizeof(double));    //
	//
	cudaMalloc(&d_U238_ntot_energy, h_U238RawXS.ntot_size * sizeof(double));   //
	cudaMalloc(&d_U238_ntot_XS,		h_U238RawXS.ntot_size * sizeof(double));   //
	cudaMalloc(&d_U238_nf_energy,	h_U238RawXS.nf_size * sizeof(double));     //
	cudaMalloc(&d_U238_nf_XS,		h_U238RawXS.nf_size * sizeof(double));     //
	cudaMalloc(&d_U238_nel_energy,	h_U238RawXS.nel_size * sizeof(double));    //
	cudaMalloc(&d_U238_nel_XS,		h_U238RawXS.nel_size * sizeof(double));    //
	cudaMalloc(&d_U238_ng_energy,	h_U238RawXS.ng_size * sizeof(double));     //
	cudaMalloc(&d_U238_ng_XS,		h_U238RawXS.ng_size * sizeof(double));     //
	cudaMalloc(&d_U238_ninl_energy, h_U238RawXS.ninl_size * sizeof(double));   //
	cudaMalloc(&d_U238_ninl_XS,		h_U238RawXS.ninl_size * sizeof(double));   //
	cudaMalloc(&d_U238_n2n_energy,	h_U238RawXS.n2n_size * sizeof(double));    //
	cudaMalloc(&d_U238_n2n_XS,		h_U238RawXS.n2n_size * sizeof(double));    //
	cudaMalloc(&d_U238_n3n_energy,	h_U238RawXS.n3n_size * sizeof(double));    //
	cudaMalloc(&d_U238_n3n_XS,		h_U238RawXS.n3n_size * sizeof(double));    //

	//
	cudaMalloc(&d_C12_ntot_energy,	h_C12RawXS.ntot_size * sizeof(double));    //
	cudaMalloc(&d_C12_ntot_XS,		h_C12RawXS.ntot_size * sizeof(double));    //
	cudaMalloc(&d_C12_nf_energy,	h_C12RawXS.nf_size * sizeof(double));      //
	cudaMalloc(&d_C12_nf_XS,		h_C12RawXS.nf_size * sizeof(double));      //
	cudaMalloc(&d_C12_nel_energy,	h_C12RawXS.nel_size * sizeof(double));     //
	cudaMalloc(&d_C12_nel_XS,		h_C12RawXS.nel_size * sizeof(double));     //
	cudaMalloc(&d_C12_ng_energy,	h_C12RawXS.ng_size * sizeof(double));      //
	cudaMalloc(&d_C12_ng_XS,		h_C12RawXS.ng_size * sizeof(double));      //
	cudaMalloc(&d_C12_ninl_energy,	h_C12RawXS.ninl_size * sizeof(double));    //
	cudaMalloc(&d_C12_ninl_XS,		h_C12RawXS.ninl_size * sizeof(double));    //
	cudaMalloc(&d_C12_n2n_energy,	h_C12RawXS.n2n_size * sizeof(double));     //
	cudaMalloc(&d_C12_n2n_XS,		h_C12RawXS.n2n_size * sizeof(double));     //
	cudaMalloc(&d_C12_n3n_energy,	h_C12RawXS.n3n_size * sizeof(double));     //
	cudaMalloc(&d_C12_n3n_XS,		h_C12RawXS.n3n_size * sizeof(double));     //
	//
	cudaMalloc(&d_H1_ntot_energy,	h_H1RawXS.ntot_size * sizeof(double));     //
	cudaMalloc(&d_H1_ntot_XS,		h_H1RawXS.ntot_size * sizeof(double));     //
	cudaMalloc(&d_H1_nf_energy,		h_H1RawXS.nf_size * sizeof(double));       //
	cudaMalloc(&d_H1_nf_XS,			h_H1RawXS.nf_size * sizeof(double));       //
	cudaMalloc(&d_H1_nel_energy,	h_H1RawXS.nel_size * sizeof(double));      //
	cudaMalloc(&d_H1_nel_XS,		h_H1RawXS.nel_size * sizeof(double));      //
	cudaMalloc(&d_H1_ng_energy,		h_H1RawXS.ng_size * sizeof(double));       //
	cudaMalloc(&d_H1_ng_XS,			h_H1RawXS.ng_size * sizeof(double));       //
	cudaMalloc(&d_H1_ninl_energy,	h_H1RawXS.ninl_size * sizeof(double));     //
	cudaMalloc(&d_H1_ninl_XS,		h_H1RawXS.ninl_size * sizeof(double));     //
	cudaMalloc(&d_H1_n2n_energy,	h_H1RawXS.n2n_size * sizeof(double));      //
	cudaMalloc(&d_H1_n2n_XS,		h_H1RawXS.n2n_size * sizeof(double));      //
	cudaMalloc(&d_H1_n3n_energy,	h_H1RawXS.n3n_size * sizeof(double));      //
	cudaMalloc(&d_H1_n3n_XS,		h_H1RawXS.n3n_size * sizeof(double));      //
	//
// ----------------------------------------------------------------------- //


// --------- Assigning the values to the device buffer arrays with host side arrays ( d -> n ) ---------------------------- //
																															//
	cudaMemcpy(d_O16_ntot_energy,	h_O16RawXS.ntot_energy,	h_O16RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_ntot_XS,		h_O16RawXS.ntot_XS,		h_O16RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_nf_energy,		h_O16RawXS.nf_energy,	h_O16RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_nf_XS,			h_O16RawXS.nf_XS,		h_O16RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_nel_energy,	h_O16RawXS.nel_energy,	h_O16RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_nel_XS,		h_O16RawXS.nel_XS,		h_O16RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_ng_energy,		h_O16RawXS.ng_energy,	h_O16RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_ng_XS,			h_O16RawXS.ng_XS,		h_O16RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_ninl_energy,	h_O16RawXS.ninl_energy,	h_O16RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_ninl_XS,		h_O16RawXS.ninl_XS,		h_O16RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_n2n_energy,	h_O16RawXS.n2n_energy,	h_O16RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_n2n_XS,		h_O16RawXS.n2n_XS,		h_O16RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_n3n_energy,	h_O16RawXS.n3n_energy,	h_O16RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_O16_n3n_XS,		h_O16RawXS.n3n_XS,		h_O16RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	//
	cudaMemcpy(d_U235_ntot_energy,	h_U235RawXS.ntot_energy,	h_U235RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_ntot_XS,		h_U235RawXS.ntot_XS,		h_U235RawXS.ntot_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_nf_energy,	h_U235RawXS.nf_energy,		h_U235RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_nf_XS,		h_U235RawXS.nf_XS,			h_U235RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_nel_energy,	h_U235RawXS.nel_energy,		h_U235RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_nel_XS,		h_U235RawXS.nel_XS,			h_U235RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_ng_energy,	h_U235RawXS.ng_energy,		h_U235RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_ng_XS,		h_U235RawXS.ng_XS,			h_U235RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_ninl_energy,	h_U235RawXS.ninl_energy,	h_U235RawXS.ninl_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_ninl_XS,		h_U235RawXS.ninl_XS,		h_U235RawXS.ninl_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_n2n_energy,	h_U235RawXS.n2n_energy,		h_U235RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_n2n_XS,		h_U235RawXS.n2n_XS,			h_U235RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U235_n3n_energy,	h_U235RawXS.n3n_energy,		h_U235RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //   
	cudaMemcpy(d_U235_n3n_XS,		h_U235RawXS.n3n_XS,			h_U235RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //  
	//
	cudaMemcpy(d_U238_ntot_energy,	h_U238RawXS.ntot_energy,	h_U238RawXS.ntot_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_ntot_XS,		h_U238RawXS.ntot_XS,		h_U238RawXS.ntot_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_nf_energy,	h_U238RawXS.nf_energy,		h_U238RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_nf_XS,		h_U238RawXS.nf_XS,			h_U238RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_nel_energy,	h_U238RawXS.nel_energy,		h_U238RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_nel_XS,		h_U238RawXS.nel_XS,			h_U238RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_ng_energy,	h_U238RawXS.ng_energy,		h_U238RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_ng_XS,		h_U238RawXS.ng_XS,			h_U238RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_ninl_energy,	h_U238RawXS.ninl_energy,	h_U238RawXS.ninl_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_ninl_XS,		h_U238RawXS.ninl_XS,		h_U238RawXS.ninl_size * sizeof(double), cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_n2n_energy,	h_U238RawXS.n2n_energy,		h_U238RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_n2n_XS,		h_U238RawXS.n2n_XS,			h_U238RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_n3n_energy,	h_U238RawXS.n3n_energy,		h_U238RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_U238_n3n_XS,		h_U238RawXS.n3n_XS,			h_U238RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //

	//
	cudaMemcpy(d_C12_ntot_energy,	h_C12RawXS.ntot_energy,		h_C12RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_ntot_XS,		h_C12RawXS.ntot_XS,			h_C12RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_nf_energy,		h_C12RawXS.nf_energy,		h_C12RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_nf_XS,			h_C12RawXS.nf_XS,			h_C12RawXS.nf_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_nel_energy,	h_C12RawXS.nel_energy,		h_C12RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_nel_XS,		h_C12RawXS.nel_XS,			h_C12RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_ng_energy,		h_C12RawXS.ng_energy,		h_C12RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_ng_XS,			h_C12RawXS.ng_XS,			h_C12RawXS.ng_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_ninl_energy,	h_C12RawXS.ninl_energy,		h_C12RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_ninl_XS,		h_C12RawXS.ninl_XS,			h_C12RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_n2n_energy,	h_C12RawXS.n2n_energy,		h_C12RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_n2n_XS,		h_C12RawXS.n2n_XS,			h_C12RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_n3n_energy,	h_C12RawXS.n3n_energy,		h_C12RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_C12_n3n_XS,		h_C12RawXS.n3n_XS,			h_C12RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	//
	cudaMemcpy(d_H1_ntot_energy,	h_H1RawXS.ntot_energy,		h_H1RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_ntot_XS,		h_H1RawXS.ntot_XS,			h_H1RawXS.ntot_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_nf_energy,		h_H1RawXS.nf_energy,		h_H1RawXS.nf_size * sizeof(double),		cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_nf_XS,			h_H1RawXS.nf_XS,			h_H1RawXS.nf_size * sizeof(double),		cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_nel_energy,		h_H1RawXS.nel_energy,		h_H1RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_nel_XS,			h_H1RawXS.nel_XS,			h_H1RawXS.nel_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_ng_energy,		h_H1RawXS.ng_energy,		h_H1RawXS.ng_size * sizeof(double),		cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_ng_XS,			h_H1RawXS.ng_XS,			h_H1RawXS.ng_size * sizeof(double),		cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_ninl_energy,	h_H1RawXS.ninl_energy,		h_H1RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_ninl_XS,		h_H1RawXS.ninl_XS,			h_H1RawXS.ninl_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_n2n_energy,		h_H1RawXS.n2n_energy,		h_H1RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_n2n_XS,			h_H1RawXS.n2n_XS,			h_H1RawXS.n2n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_n3n_energy,		h_H1RawXS.n3n_energy,		h_H1RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	cudaMemcpy(d_H1_n3n_XS,			h_H1RawXS.n3n_XS,			h_H1RawXS.n3n_size * sizeof(double),	cudaMemcpyHostToDevice);    //
	//
// ------------------------------------------------------------------------------------------------------------------------ //



// -- * Building a temporary host-side copy of Object, and fill it with device array buffers -- //
																								//
	RawCrossSection tmp_O16 = h_O16RawXS;                                                           //
	tmp_O16.ntot_energy	=	d_O16_ntot_energy;	tmp_O16.ntot_XS =	d_O16_ntot_XS;                    //
	tmp_O16.nf_energy	=	d_O16_nf_energy;	tmp_O16.nf_XS	=	d_O16_nf_XS;                      //
	tmp_O16.nel_energy	=	d_O16_nel_energy;	tmp_O16.nel_XS	=	d_O16_nel_XS;                     //
	tmp_O16.ng_energy	=	d_O16_ng_energy;	tmp_O16.ng_XS   =	d_O16_ng_XS;                      //
	tmp_O16.ninl_energy =	d_O16_ninl_energy;	tmp_O16.ninl_XS =	d_O16_ninl_XS;                    //
	tmp_O16.n2n_energy	=	d_O16_n2n_energy;	tmp_O16.n2n_XS  =	d_O16_n2n_XS;                     //
	tmp_O16.n3n_energy	=	d_O16_n3n_energy;	tmp_O16.n3n_XS  =	d_O16_n3n_XS;                     //
	//
	RawCrossSection tmp_U235 = h_U235RawXS;                                                         //
	tmp_U235.ntot_energy =	d_U235_ntot_energy;  tmp_U235.ntot_XS =	d_U235_ntot_XS;                  //
	tmp_U235.nf_energy	 =	d_U235_nf_energy;    tmp_U235.nf_XS	  =	d_U235_nf_XS;                    //
	tmp_U235.nel_energy  =	d_U235_nel_energy;   tmp_U235.nel_XS  =	d_U235_nel_XS;                   //
	tmp_U235.ng_energy	 =	d_U235_ng_energy;    tmp_U235.ng_XS	  =	d_U235_ng_XS;                    //
	tmp_U235.ninl_energy =	d_U235_ninl_energy;  tmp_U235.ninl_XS =	d_U235_ninl_XS;                  //
	tmp_U235.n2n_energy  =	d_U235_n2n_energy;   tmp_U235.n2n_XS  =	d_U235_n2n_XS;                   //
	tmp_U235.n3n_energy  =	d_U235_n3n_energy;   tmp_U235.n3n_XS  = d_U235_n3n_XS;                   //
	//
	RawCrossSection tmp_U238 = h_U238RawXS;                                                         //
	tmp_U238.ntot_energy =	d_U238_ntot_energy;  tmp_U238.ntot_XS = d_U238_ntot_XS;                  //
	tmp_U238.nf_energy	 =	d_U238_nf_energy;    tmp_U238.nf_XS   = d_U238_nf_XS;                    //
	tmp_U238.nel_energy  =	d_U238_nel_energy;   tmp_U238.nel_XS  = d_U238_nel_XS;                   //
	tmp_U238.ng_energy   =	d_U238_ng_energy;    tmp_U238.ng_XS   = d_U238_ng_XS;                    //
	tmp_U238.ninl_energy =	d_U238_ninl_energy;  tmp_U238.ninl_XS = d_U238_ninl_XS;                  //
	tmp_U238.n2n_energy  =	d_U238_n2n_energy;   tmp_U238.n2n_XS  = d_U238_n2n_XS;                   //
	tmp_U238.n3n_energy  =	d_U238_n3n_energy;   tmp_U238.n3n_XS  = d_U238_n3n_XS;                   //
	//
	RawCrossSection tmp_C12 = h_C12RawXS;                                                           //
	tmp_C12.ntot_energy = d_C12_ntot_energy;    tmp_C12.ntot_XS	= d_C12_ntot_XS;                   //
	tmp_C12.nf_energy	= d_C12_nf_energy;      tmp_C12.nf_XS	= d_C12_nf_XS;                     //
	tmp_C12.nel_energy	= d_C12_nel_energy;     tmp_C12.nel_XS	= d_C12_nel_XS;                    //
	tmp_C12.ng_energy	= d_C12_ng_energy;      tmp_C12.ng_XS	= d_C12_ng_XS;                     //
	tmp_C12.ninl_energy = d_C12_ninl_energy;    tmp_C12.ninl_XS = d_C12_ninl_XS;                   //
	tmp_C12.n2n_energy	= d_C12_n2n_energy;     tmp_C12.n2n_XS	= d_C12_n2n_XS;                    //
	tmp_C12.n3n_energy	= d_C12_n3n_energy;     tmp_C12.n3n_XS	= d_C12_n3n_XS;                    //
	//
	RawCrossSection tmp_H1 = h_H1RawXS;                                                             //
	tmp_H1.ntot_energy	= d_H1_ntot_energy;     tmp_H1.ntot_XS	= d_H1_ntot_XS;                    //
	tmp_H1.nf_energy	= d_H1_nf_energy;       tmp_H1.nf_XS	= d_H1_nf_XS;                      //
	tmp_H1.nel_energy	= d_H1_nel_energy;      tmp_H1.nel_XS	= d_H1_nel_XS;                     //
	tmp_H1.ng_energy	= d_H1_ng_energy;       tmp_H1.ng_XS	= d_H1_ng_XS;                      //
	tmp_H1.ninl_energy	= d_H1_ninl_energy;     tmp_H1.ninl_XS	= d_H1_ninl_XS;                    //
	tmp_H1.n2n_energy	= d_H1_n2n_energy;      tmp_H1.n2n_XS	= d_H1_n2n_XS;                     //
	tmp_H1.n3n_energy	= d_H1_n3n_energy;      tmp_H1.n3n_XS	= d_H1_n3n_XS;                     //

	// -------------------------------------------------------------------------------------------- //


	// -- Copy the device-array filled struct back to the originally assigned device struct --- //
																								//
	cudaMemcpy(d_O16RawXS, &tmp_O16, sizeof(RawCrossSection), cudaMemcpyHostToDevice);        //
	cudaMemcpy(d_U235RawXS, &tmp_U235, sizeof(RawCrossSection), cudaMemcpyHostToDevice);        //

	cudaMemcpy(d_U238RawXS, &tmp_U238, sizeof(RawCrossSection), cudaMemcpyHostToDevice);        //
	cudaMemcpy(d_C12RawXS, &tmp_C12, sizeof(RawCrossSection), cudaMemcpyHostToDevice);        //
	cudaMemcpy(d_H1RawXS, &tmp_H1, sizeof(RawCrossSection), cudaMemcpyHostToDevice);        //
	//
// ---------------------------------------------------------------------------------------- //

// ---- Memory copy from host to device completed! ---- //


/*
 *  __  __   _   ___ _  _     ___ ___ __  __ _   _ _      _ _____ ___ ___  _  _     ___ _____ _   ___ _____ ___     _  _ ___ ___ ___
 * |  \/  | /_\ |_ _| \| |   / __|_ _|  \/  | | | | |    /_\_   _|_ _/ _ \| \| |   / __|_   _/_\ | _ \_   _/ __|   | || | __| _ \ __|
 * | |\/| |/ _ \ | || .` |   \__ \| || |\/| | |_| | |__ / _ \| |  | | (_) | .` |   \__ \ | |/ _ \|   / | | \__ \   | __ | _||   / _|
 * |_|  |_/_/ \_\___|_|\_|   |___/___|_|  |_|\___/|____/_/ \_\_| |___\___/|_|\_|   |___/ |_/_/ \_\_|_\ |_| |___/   |_||_|___|_|_\___|
 *
 */

	double cpyEndT = clock();
	std::cout << "\nTime took for total allocation: " << (cpyEndT - allocStartT) / TimeDivider << " milliseconds\n";

	int threadPerBlock = 32;
	int blockPerDim = (numNeutrons + threadPerBlock - 1) / threadPerBlock;

	//FindCrossSection <<<blockPerDim, threadPerBlock >> > (d_U235RawXS, d_randomDist, d_value, numNeutrons);
	//Test<<<blockPerDim, threadPerBlock>>>(d_U235RawXS, d_randomDist, d_test, numNeutrons);
	//AddCrossSection<<<blockPerDim, threadPerBlock>>>(d_O16RawXS, d_U238RawXS, d_randomDist, d_test, numNeutrons);

	/*
	cudaError_t err = cudaGetLastError();         // launch‐time errors
	cudaDeviceSynchronize();                      // execution errors
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << "\n";
	}
	*/

	//Test<<<2, threadPerBlock>>>(d_U238RawXS, d_randomDist, d_test, numNeutrons);
	//cudaDeviceSynchronize();
	//cudaMemcpy(h_value, d_value, numNeutrons* sizeof(double), cudaMemcpyDeviceToHost);



	std::cout << "\n\nInitial Neutron position:\n";
	for (int i = 0; i < 10; i++) {
		//std::cout << "for energy " << h_randomDist[i] << " eV, the (n,f) XS of U235 was was " << h_test[i] << " barns.\n";
		//std::cout << "Hello there! this loop works\n";
		std::cout << h_Neutrons.m_initialNeutron[i].m_pos.x << ", " << h_Neutrons.m_initialNeutron[i].m_pos.y << ", "
			<< h_Neutrons.m_initialNeutron[i].m_pos.z << "\n";
		//std::cout << h_Neutrons.m_initialNeutron[i].m_dirVec.x << ", " << h_Neutrons.m_initialNeutron[i].m_dirVec.y << ", "
		//    << h_Neutrons.m_initialNeutron[i].m_dirVec.z << "\n";
	}


	std::cout << "\n\n\n";
	//NeutronNullifier<<<blockPerDim, threadPerBlock>>>(d_Neutrons, numNeutrons);
	/*
	double loopStartT = clock();

	for (int i = 0; i < 1000; i++) {
		getNextReactionDistance<<<blockPerDim, threadPerBlock>>>(d_ChicagoPile1, d_Neutrons, d_U235RawXS, d_U238RawXS, d_O16RawXS, d_C12RawXS,
			numNeutrons, d_Seed, d_return, d_int);
		cudaDeviceSynchronize();
		//if (i % 1000 == 0) {
		//	std::cout << "we are currently at loop " << i << "\n";
		//}
		//cudaMemcpy(&h_int, d_int, sizeof(int), cudaMemcpyDeviceToHost);
		//std::cout << "\n" << h_int << "\n";
	}
	double loopEndT = clock();


	cudaMemcpy(&h_int, d_int, sizeof(int), cudaMemcpyDeviceToHost);
	h_int = 0;
	cudaMemcpy(d_int, &h_int, sizeof(int), cudaMemcpyHostToDevice);




	
	std::cout << "\nEach loop takes " << std::setprecision(8) << (loopEndT - loopStartT) / TimeDivider / 1000  << "milliseconds\n\n\n";
	//cudaMemcpy(h_value, d_value, numNeutrons * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_return, d_return, numNeutrons * sizeof(double), cudaMemcpyDeviceToHost);


	// how to copy back the neutron back to the host?
	// same as copying it to device ! 
	// make a "temporary struct that receives the scalar values (int, enums), and copy it to the original object.
	// for the vector values (i.e. arrays), copy it from the already declared device buffer arrays to the host vector.

	NeutronDistribution h_NeutronsReceiver = h_Neutrons;
	cudaMemcpy(&h_NeutronsReceiver, d_Neutrons, sizeof(NeutronDistribution), cudaMemcpyDeviceToHost);
	h_Neutrons.m_initialNeutronNumber = h_NeutronsReceiver.m_initialNeutronNumber;
	h_Neutrons.m_addedNeutronNumber = h_NeutronsReceiver.m_addedNeutronNumber;

	cudaMemcpy(h_Neutrons.m_initialNeutron, d_bufferInitialNeutron, numNeutrons * sizeof(Neutron), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_Neutrons.m_addedNeutron, d_bufferAddedNeutron, numNeutrons * sizeof(Neutron), cudaMemcpyDeviceToHost);

	// if you will never modify the scalar values of the object, you can just use thes 2 lines:
	//cudaMemcpy(h_Neutrons.m_initialNeutron, d_bufferInitialNeutron, numNeutrons * sizeof(Neutron), cudaMemcpyDeviceToHost);
	//cudaMemcpy(h_Neutrons.m_addedNeutron, d_bufferAddedNeutron, numNeutrons * sizeof(Neutron), cudaMemcpyDeviceToHost);
	// this will do the job.

	std::cout << "\n\nPosition of neutron after reaction:\n";
	for (int i = 0; i < 10; i++) {
		//std::cout << "for energy " << h_randomDist[i] << " eV, the (n,f) XS of U235 was was " << h_test[i] << " barns.\n";
		//std::cout << "Hello there! this loop works\n";
		std::cout << h_Neutrons.m_initialNeutron[i].m_pos.x << ", " << h_Neutrons.m_initialNeutron[i].m_pos.y << ", "
			<< h_Neutrons.m_initialNeutron[i].m_pos.z << "\n";
		//std::cout << h_Neutrons.m_initialNeutron[i].m_dirVec.x << ", " << h_Neutrons.m_initialNeutron[i].m_dirVec.y << ", "
		//    << h_Neutrons.m_initialNeutron[i].m_dirVec.z << "\n";
	}


	std::cout << "\n\n\n\nDistances traveled before inteaction:\n";
	for (int i = 0; i < 10; i++) {
		printf("%e\n", h_return[i]);
	}

	double sum = 0;
	for (int i = 0; i < numNeutrons; i++) {
		sum += h_return[i];
	}
	std::cout << "Average distance traveled: " << sum / numNeutrons << "\n";


	Neutron averageNeutron({ 0,0,0 }, { 0,0,0 }, 1E+6);
	double totalXS = h_ChicagoPile1.getTotalMacroXS(averageNeutron, &h_U238RawXS, &h_U235RawXS, &h_O16RawXS, &h_C12RawXS);
	double averageDistance = -log(0.5) / (totalXS * 100);

	std::cout << "Average distance for Energy of 1 Mev: " << averageDistance << "\n";

	// WHY THE VALUE IS DIFFERENT? for gpu-launched Neutrons, the average distance is 2.22456e-01, but for 1 mev its 1.05 ?? 
	// maybe because of the XS difference? 



	*/

	// THRUST PART TEST
	std::cout << "\n\nInitial Thrust Neutron Position:\n";
	for (int i = 0; i < 10; i++) {
		std::cout << h_NeutronThrust.m_neutrons[i].m_pos.x << ", " << h_NeutronThrust.m_neutrons[i].m_pos.y << ", "
			<< h_NeutronThrust.m_neutrons[i].m_pos.z << "\n";
	}
	

	double loopStartT = clock();
	int loopSize = 10;
	for (int i = 0; i < loopSize; i++) {
		ThrustTest<<<blockPerDim, threadPerBlock>>>(d_ChicagoPile1, d_NeutronThrust, d_U235RawXS, d_U238RawXS, d_O16RawXS, d_C12RawXS,
			numNeutrons, d_Seed, d_return, d_int);
		cudaDeviceSynchronize();
		//cudaMemcpy(&h_int, d_int, sizeof(int), cudaMemcpyDeviceToHost);
		//std::cout << "\n" << h_int << "\n";
	}
	double loopEndT = clock();

	std::cout << "\n\ntime for thrust library, each loop takes :" << std::setprecision(8) << (loopEndT - loopStartT) / TimeDivider / loopSize << "milliseconds\n";
	std::cout << "\ntotal time:" << std::setprecision(8) << (loopEndT - loopStartT) / TimeDivider << "milliseconds\n\n\n";
	// for getting the next reaction distance, each loop takes 5.8 milliseconds. 
	// running program for 10 hours, it will have exectured total of 6,206,896 loops. This is for 500,000 Neutrons.
	// this have almost no performance difference to the original one, but the kick is, the neutron array(with thrust, its a vector) is scalable - a huge developemental benefit.

	h_NeutronThrust.DtoH(d_NeutronVector, d_addedNeutronVector); // send it back to Host

	std::cout << "\n\nAfter reaction Thrust Neutron Position:\n";
	for (int i = 0; i < 10; i++) {
		std::cout << h_NeutronThrust.m_neutrons[i].m_pos.x << ", " << h_NeutronThrust.m_neutrons[i].m_pos.y << ", "
			<< h_NeutronThrust.m_neutrons[i].m_pos.z << "\n";
	}
	std::cout << "\n\n";
	for (int i = 0; i < 10; i++) {
		std::cout << h_NeutronThrust.m_addedNeutrons[i].m_pos.x << ", " << h_NeutronThrust.m_addedNeutrons[i].m_pos.y << ", "
			<< h_NeutronThrust.m_addedNeutrons[i].m_pos.z << "\n";
	}

	NeutronThrustManager Manager(d_NeutronVector, d_addedNeutronVector, seedNo, SpectrumType::default);
	std::cout << Manager.d_neutrons.size() << "  " << Manager.d_addedNeutrons.size() << "\n";

	Manager.MergeNeutron();

	std::cout << Manager.d_neutrons.size();

	


/*
 *  ___  ___   _   _    _    ___   ___   _ _____ ___ ___  _  _
 * |   \| __| /_\ | |  | |  / _ \ / __| /_\_   _|_ _/ _ \| \| |
 * | |) | _| / _ \| |__| |_| (_) | (__ / _ \| |  | | (_) | .` |
 * |___/|___/_/ \_\____|____\___/ \___/_/ \_\_| |___\___/|_|\_|
 *
 */

	delete[] h_value;
	delete[] h_return;
	delete[] h_randomDist;

	delete[] h_Seed;
	cudaFree(d_Seed);

	cudaFree(d_O16RawXS);
	cudaFree(d_U235RawXS);
	cudaFree(d_U238RawXS);
	cudaFree(d_C12RawXS);
	cudaFree(d_H1RawXS);

	cudaFree(d_value);
	cudaFree(d_randomDist);
	cudaFree(d_return);

	cudaFree(d_O16_ntot_energy);
	cudaFree(d_O16_ntot_XS);
	cudaFree(d_O16_nf_energy);
	cudaFree(d_O16_nf_XS);
	cudaFree(d_O16_nel_energy);
	cudaFree(d_O16_nel_XS);
	cudaFree(d_O16_ng_energy);
	cudaFree(d_O16_ng_XS);
	cudaFree(d_O16_ninl_energy);
	cudaFree(d_O16_ninl_XS);
	cudaFree(d_O16_n2n_energy);
	cudaFree(d_O16_n2n_XS);
	cudaFree(d_O16_n3n_energy);
	cudaFree(d_O16_n3n_XS);

	// U235 channels: 
	cudaFree(d_U235_ntot_energy);
	cudaFree(d_U235_ntot_XS);
	cudaFree(d_U235_nf_energy);
	cudaFree(d_U235_nf_XS);
	cudaFree(d_U235_nel_energy);
	cudaFree(d_U235_nel_XS);
	cudaFree(d_U235_ng_energy);
	cudaFree(d_U235_ng_XS);
	cudaFree(d_U235_ninl_energy);
	cudaFree(d_U235_ninl_XS);
	cudaFree(d_U235_n2n_energy);
	cudaFree(d_U235_n2n_XS);
	cudaFree(d_U235_n3n_energy);
	cudaFree(d_U235_n3n_XS);

	// U238 channels:
	cudaFree(d_U238_ntot_energy);
	cudaFree(d_U238_ntot_XS);
	cudaFree(d_U238_nf_energy);
	cudaFree(d_U238_nf_XS);
	cudaFree(d_U238_nel_energy);
	cudaFree(d_U238_nel_XS);
	cudaFree(d_U238_ng_energy);
	cudaFree(d_U238_ng_XS);
	cudaFree(d_U238_ninl_energy);
	cudaFree(d_U238_ninl_XS);
	cudaFree(d_U238_n2n_energy);
	cudaFree(d_U238_n2n_XS);
	cudaFree(d_U238_n3n_energy);
	cudaFree(d_U238_n3n_XS);

	// C12 channels:
	cudaFree(d_C12_ntot_energy);
	cudaFree(d_C12_ntot_XS);
	cudaFree(d_C12_nf_energy);
	cudaFree(d_C12_nf_XS);
	cudaFree(d_C12_nel_energy);
	cudaFree(d_C12_nel_XS);
	cudaFree(d_C12_ng_energy);
	cudaFree(d_C12_ng_XS);
	cudaFree(d_C12_ninl_energy);
	cudaFree(d_C12_ninl_XS);
	cudaFree(d_C12_n2n_energy);
	cudaFree(d_C12_n2n_XS);
	cudaFree(d_C12_n3n_energy);
	cudaFree(d_C12_n3n_XS);

	// H1 channels:
	cudaFree(d_H1_ntot_energy);
	cudaFree(d_H1_ntot_XS);
	cudaFree(d_H1_nf_energy);
	cudaFree(d_H1_nf_XS);
	cudaFree(d_H1_nel_energy);
	cudaFree(d_H1_nel_XS);
	cudaFree(d_H1_ng_energy);
	cudaFree(d_H1_ng_XS);
	cudaFree(d_H1_ninl_energy);
	cudaFree(d_H1_ninl_XS);
	cudaFree(d_H1_n2n_energy);
	cudaFree(d_H1_n2n_XS);
	cudaFree(d_H1_n3n_energy);
	cudaFree(d_H1_n3n_XS);

	cudaFree(d_Neutrons);
	cudaFree(d_bufferInitialNeutron);
	cudaFree(d_bufferAddedNeutron);
	cudaFree(d_ChicagoPile1);

	cudaFree(d_NeutronThrust);
}