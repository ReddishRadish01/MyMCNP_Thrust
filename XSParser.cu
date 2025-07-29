#pragma once
#include <iostream>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>

#include "Constants.cuh"
#include "XSParser.cuh"

#include "RNG.cuh"
#include "Neutron.cuh"



std::string EnergyCrossSection::getNameOfTextFile() {
	return m_XSDataFilename;
}

double EnergyCrossSection::getCrossSectionByIndex(int index) {
	return m_EnergyCrossSection[index].first;
}
double EnergyCrossSection::getEnergyByIndex(int index) {
	return m_EnergyCrossSection[index].second;
}

double EnergyCrossSection::getSizeOfVectorArrayinMB() {
	// sizeof: byte size of an element - in this case, 16+16(double + double) = 32
	// .size(): return number of elements 
	// size of the std::vector<std::pair<double, double>> : std::vector "overhead" + ( number of elements * element size of std::pair<double, double> )
	//	
	return (sizeof(m_EnergyCrossSection) + (m_EnergyCrossSection.size() * sizeof(std::pair<double, double>))) / pow(10, 6);
	// sizeof(m_EnergyCrossSection) * m_EnergyCrossSection.size() / pow(10, 6);
}

// finds the value stored in the vector and outputs the cross section.
void EnergyCrossSection::getCrossSectionByEnergy(const double inputEnergy) {
	// measure time needed for looking through the vector. Mark the starting time:
	auto start = std::chrono::high_resolution_clock::now();

	int index = 0;
	for (const auto& pair : m_EnergyCrossSection) {  // using Range-Based For Loop 
		// this function is faster by 0.0014s avg. 
		// But it depends heavily on whether it is called from the dram memory, or the cache memory.
		if (pair.first == inputEnergy) {
			std::cout << "For neutron energy " << inputEnergy << " eV - ";
			std::cout << "Cross Section: " << m_EnergyCrossSection[index].second << " barns [10^{-24} cm^2]" << "\n";
			std::cout << "index at the text file: " << index + 1 << "\n";
			//return m_EnergyCrossSection[index].second;
			break;
		}
		else if (inputEnergy > pair.first && inputEnergy < m_EnergyCrossSection[index + 1].first) {
			double interpolatedCrossSection;
			interpolatedCrossSection = pair.second + (inputEnergy - pair.first) * (m_EnergyCrossSection[index + 1].second - pair.second) / (m_EnergyCrossSection[index + 1].first - pair.first);
			std::cout << "For neutron energy " << inputEnergy << " eV - ";
			std::cout << "Interpolated Cross Section: " << interpolatedCrossSection;
			std::cout << " barn [10^{-24} cm^2]" << "\n";
			std::cout << "Nearest energy:\n " << pair.first << " eV: " << pair.second << " barns [10^{-24} cm^2]\n";
			std::cout << m_EnergyCrossSection[index + 1].first << " eV: " << m_EnergyCrossSection[index + 1].second << " barns [10^{-24} cm^2]\n";
			//return interpolatedCrossSection;
			break;
		}

		if (inputEnergy < m_EnergyCrossSection[0].first || inputEnergy > m_EnergyCrossSection[m_EnergyCrossSection.size() - 1].first) {
			std::cout << "Energy value out of range!\n";
			break;
		}
		index++;
	}


	/* This one is to implement above without using range based for loop.
	for (const auto& pair = m_EnergyCrossSection; pair != m_EnergyCrossSection.end(); pair++) {
		// ...
	} */

	/*
	  //This one is using plain for loop, which makes indexing easier.
	for (int index = 0; index < m_EnergyCrossSection.size(); index++) { <- this one is slower, about 0.0014s avg
		if (m_EnergyCrossSection[index].first == inputEnergy) {
			std::cout << "For neutron energy " << inputEnergy << " eV - ";
			std::cout << "Cross Section: " << m_EnergyCrossSection[index].second << " barns [10^{-24} cm^2]" << "\n";
			break;
		}
		else if (inputEnergy > m_EnergyCrossSection[index].first && inputEnergy < m_EnergyCrossSection[index + 1].first) {
			double interpolatedCrossSection;
			interpolatedCrossSection = m_EnergyCrossSection[index].second + (inputEnergy - m_EnergyCrossSection[index].first)
				* (m_EnergyCrossSection[index + 1].second - m_EnergyCrossSection[index].second)
				/ (m_EnergyCrossSection[index + 1].first - m_EnergyCrossSection[index].first);

			std::cout << "For neutron energy " << inputEnergy << " eV - ";
			std::cout << "interpolated cross section: " << interpolatedCrossSection;
			std::cout << " barn [10^{-24} cm^2]" << "\n";
			std::cout << "Nearest energy:\n" << m_EnergyCrossSection[index].first << " eV: " << m_EnergyCrossSection[index].second << " barns [10^{-24} cm^2]\n";
			std::cout << m_EnergyCrossSection[index + 1].first << " eV: " << m_EnergyCrossSection[index + 1].second << " barns [10^{-24} cm^2]\n";
			break;
		}

		if (inputEnergy < m_EnergyCrossSection[0].first || inputEnergy > m_EnergyCrossSection[m_EnergyCrossSection.size() - 1].first) {
			std::cout << "Energy value out of range!\n";
			break;
		}
	}
	*/

	// Marking the end time:
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsedTime = end - start;
	std::cout << "Time taken: " << elapsedTime.count() << "seconds\n\n";
}

void EnergyCrossSection::exportToVectors(std::vector<double>& energies, std::vector<double>& crossSections) const {
	energies.reserve(m_EnergyCrossSection.size());
	crossSections.reserve(m_EnergyCrossSection.size());
	for (const auto& pair : m_EnergyCrossSection) {
		energies.push_back(pair.first);
		crossSections.push_back(pair.second);
	}
}

// redundant __device__ function
__device__ double getCrossSectionByEnergy(double incidentEnergy, const double* energy, const double* crossSection, int numOfEntries) {
	if (incidentEnergy <= energy[0]) {
		return crossSection[0];
	}
	if (incidentEnergy >= energy[numOfEntries - 1]) {
		return crossSection[numOfEntries - 1];
	}
	// binary search - better performance than linear search
	int low = 0;
	int high = numOfEntries - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}
	return crossSection[low] + (crossSection[high] - crossSection[low]) * (incidentEnergy - energy[low]) / (energy[high] - energy[low]);
}


__device__ double ArrayCrossSection::getCrossSectionByEnergy(double incidentEnergy) {
	if (incidentEnergy <= energy[0]) return 0;
	if (incidentEnergy >= energy[size - 1]) return crossSection[size - 1];

	int low = 0;
	int high = size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}
	
	return crossSection[low] + (crossSection[high] - crossSection[low]) * (incidentEnergy - energy[low]) / (energy[high] - energy[low]);
	//return 0;
}




// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::getTotalMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= ntot_energy[0]) return 0;
	if (incidentEnergy >= ntot_energy[ntot_size - 1]) return ntot_XS[ntot_size - 1];

	int low = 0;
	int high = ntot_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (ntot_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return ntot_XS[low] + (ntot_XS[high] - ntot_XS[low]) * (incidentEnergy - ntot_energy[low]) / (ntot_energy[high] - ntot_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::getFMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= nf_energy[0]) return 0;
	if (incidentEnergy >= nf_energy[ng_size - 1]) return nf_XS[ng_size - 1];

	int low = 0;
	int high = nf_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (nf_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return nf_XS[low] + (nf_XS[high] - nf_XS[low]) * (incidentEnergy - nf_energy[low]) / (nf_energy[high] - nf_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::getGMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= ng_energy[0]) return 0;
	if (incidentEnergy >= ng_energy[ng_size - 1]) return ng_XS[ng_size - 1];

	int low = 0;
	int high = ng_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (ng_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return ng_XS[low] + (ng_XS[high] - ng_XS[low]) * (incidentEnergy - ng_energy[low]) / (ng_energy[high] - ng_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::getElMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= nel_energy[0]) return 0;
	if (incidentEnergy >= nel_energy[nel_size - 1]) return nel_XS[nel_size - 1];

	int low = 0;
	int high = nel_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (nel_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return nel_XS[low] + (nel_XS[high] - nel_XS[low]) * (incidentEnergy - nel_energy[low]) / (nel_energy[high] - nel_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::getInlMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= ninl_energy[0]) return 0;
	if (incidentEnergy >= ninl_energy[ninl_size - 1]) return ninl_XS[ninl_size - 1];

	int low = 0;
	int high = ninl_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (ninl_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return ninl_XS[low] + (ninl_XS[high] - ninl_XS[low]) * (incidentEnergy - ninl_energy[low]) / (ninl_energy[high] - ninl_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::get2nMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= n2n_energy[0]) return 0;
	if (incidentEnergy >= n2n_energy[n2n_size - 1]) return n2n_XS[n2n_size - 1];

	int low = 0;
	int high = n2n_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (n2n_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return n2n_XS[low] + (n2n_XS[high] - n2n_XS[low]) * (incidentEnergy - n2n_energy[low]) / (n2n_energy[high] - n2n_energy[low]);
}

// returns value in barns ( 10^{-24} cm^2 )
__host__ __device__ double RawCrossSection::get3nMicroXSByEnergy(double incidentEnergy) {
	if (incidentEnergy <= n3n_energy[0]) return 0;
	if (incidentEnergy >= n3n_energy[n3n_size - 1]) return n3n_XS[n3n_size - 1];

	int low = 0;
	int high = n3n_size - 1;
	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (n3n_energy[mid] > incidentEnergy)
			high = mid;
		else
			low = mid;
	}

	return n3n_XS[low] + (n3n_XS[high] - n3n_XS[low]) * (incidentEnergy - n3n_energy[low]) / (n3n_energy[high] - n3n_energy[low]);
}


// this is almost non-used - to get the interaction mode, you really need the composition of the fuel - thus moved to FuelKernel.cu and .cuh
__host__ __device__ InteractionType RawCrossSection::getInteractionModeByEnergy(double incidentEnergy, unsigned long long SeedNo) {
	return InteractionType::nf;
}




__host__ __device__ double getCrossSectionTot(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->ntot_size - 1;
	if (incidentEnergy <= rawXS_ptr->ntot_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->ntot_energy[high]) return rawXS_ptr->ntot_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->ntot_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->ntot_XS[low] + (rawXS_ptr->ntot_XS[high] - rawXS_ptr->ntot_XS[low])
		* (incidentEnergy - rawXS_ptr->ntot_energy[low]) / (rawXS_ptr->ntot_energy[high] - rawXS_ptr->ntot_energy[low]);
}

__host__ __device__ double getCrossSectionInl(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->ninl_size - 1;
	if (incidentEnergy <= rawXS_ptr->ninl_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->ninl_energy[high]) return rawXS_ptr->ninl_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->ninl_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->ninl_XS[low] + (rawXS_ptr->ninl_XS[high] - rawXS_ptr->ninl_XS[low])
		* (incidentEnergy - rawXS_ptr->ninl_energy[low]) / (rawXS_ptr->ninl_energy[high] - rawXS_ptr->ninl_energy[low]);
}

__host__ __device__ double getCrossSectionEl(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->nel_size - 1;
	if (incidentEnergy <= rawXS_ptr->nel_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->nel_energy[high]) return rawXS_ptr->nel_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->nel_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->nel_XS[low] + (rawXS_ptr->nel_XS[high] - rawXS_ptr->nel_XS[low])
		* (incidentEnergy - rawXS_ptr->nel_energy[low]) / (rawXS_ptr->nel_energy[high] - rawXS_ptr->nel_energy[low]);
}

__host__ __device__ double getCrossSectionF(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->nf_size - 1;
	if (incidentEnergy <= rawXS_ptr->nf_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->nf_energy[high]) return rawXS_ptr->nf_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->nf_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->nf_XS[low] + (rawXS_ptr->nf_XS[high] - rawXS_ptr->nf_XS[low])
		* (incidentEnergy - rawXS_ptr->nf_energy[low]) / (rawXS_ptr->nf_energy[high] - rawXS_ptr->nf_energy[low]);
}

__host__ __device__ double getCrossSectionG(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->ng_size - 1;
	if (incidentEnergy <= rawXS_ptr->ng_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->ng_energy[high]) return rawXS_ptr->ng_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->ng_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->ng_XS[low] + (rawXS_ptr->ng_XS[high] - rawXS_ptr->ng_XS[low])
		* (incidentEnergy - rawXS_ptr->ng_energy[low]) / (rawXS_ptr->ng_energy[high] - rawXS_ptr->ng_energy[low]);
}

__host__ __device__ double getCrossSection2N(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->n2n_size - 1;
	if (incidentEnergy <= rawXS_ptr->n2n_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->n2n_energy[high]) return rawXS_ptr->n2n_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->n2n_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->n2n_XS[low] + (rawXS_ptr->n2n_XS[high] - rawXS_ptr->n2n_XS[low])
		* (incidentEnergy - rawXS_ptr->n2n_energy[low]) / (rawXS_ptr->n2n_energy[high] - rawXS_ptr->n2n_energy[low]);
}

__host__ __device__ double getCrossSection3N(RawCrossSection* rawXS_ptr, double incidentEnergy) {
	int low = 0;
	int high = rawXS_ptr->n3n_size - 1;
	if (incidentEnergy <= rawXS_ptr->n3n_energy[0]) return 0;
	if (incidentEnergy >= rawXS_ptr->n3n_energy[high]) return rawXS_ptr->n3n_XS[high];

	while (high - low > 1) {
		int mid = (low + high) / 2;
		if (rawXS_ptr->n3n_energy[mid] > incidentEnergy) {
			high = mid;
		}
		else {
			low = mid;
		}
	}
	return rawXS_ptr->n3n_XS[low] + (rawXS_ptr->n3n_XS[high] - rawXS_ptr->n3n_XS[low])
		* (incidentEnergy - rawXS_ptr->n3n_energy[low]) / (rawXS_ptr->n3n_energy[high] - rawXS_ptr->n3n_energy[low]);
}
