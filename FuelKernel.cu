#pragma once
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Constants.cuh"
#include "FuelKernel.cuh"

#include "RNG.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"

//#include "common.cuh"



// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::getTotalMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS,
	RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	// Some basic theories here: \rho_{UO2} = \frac{Z M_{UO2}}{N_A a^3} <- calculating the atom density from lattice perspective, Z = 4(4 UO2 per unit cell)
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3) ); 
	double atomRho_UO2 = 4 / (pow(5.47e-10, 3) * pow(10, 6));		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	// homogeneous - no geometrical dependency.
	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->getTotalMicroXSByEnergy(incidentNeutron.m_energy)* m_enrichment / 100 + U238XS->getTotalMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->getTotalMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->getTotalMicroXSByEnergy(incidentNeutron.m_energy) );
	
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::getFMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->getFMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->getFMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->getFMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->getFMicroXSByEnergy(incidentNeutron.m_energy));
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::getGMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->getGMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->getGMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->getGMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->getGMicroXSByEnergy(incidentNeutron.m_energy));
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::getElMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->getElMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->getElMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->getElMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->getElMicroXSByEnergy(incidentNeutron.m_energy));
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::getInlMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->getInlMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->getInlMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->getInlMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->getInlMicroXSByEnergy(incidentNeutron.m_energy));
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::get2nMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->get2nMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->get2nMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->get2nMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->get2nMicroXSByEnergy(incidentNeutron.m_energy));
}

// returns cm^{-1}, 10^{-24} is compensated.
__host__ __device__ double BareSphere::get3nMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	return pow(10, -24) * (atomRho_UO2 * volumeFracUO2 * (U235XS->get3nMicroXSByEnergy(incidentNeutron.m_energy) * m_enrichment / 100 + U238XS->get3nMicroXSByEnergy(incidentNeutron.m_energy) * (1 - (m_enrichment / 100)) + 2 * O16XS->get3nMicroXSByEnergy(incidentNeutron.m_energy))
		+ atomRho_Mod * volumeFracC * ModXS->get3nMicroXSByEnergy(incidentNeutron.m_energy));
}

__host__ __device__ double BareSphere::getUO2XSbyEnergy(RawCrossSection* U235XS_ptr, RawCrossSection* U238XS_ptr, RawCrossSection* O16XS_ptr, double* enrichment, double incidentEnergy) {
	return 0.0;
}

__host__ __device__ InteractionType BareSphere::getInterationType(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS, unsigned long long seedNo) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;
	GnuAMCM RNG(seedNo);

	double totalMacroXS = getTotalMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double fissionMacroXS = getFMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double elasticMacroXS = getElMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double captureMacroXS = getGMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double inelasticMacroXS = getInlMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double n2nMacroXS = get2nMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);
	double n3nMacroXS = get3nMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS);

	double cumulativeFission = fissionMacroXS;
	double cumulativeElastic = cumulativeFission + elasticMacroXS;
	double cumulativeCapture = cumulativeElastic + captureMacroXS;
	double cumulativeInelastic = cumulativeCapture + inelasticMacroXS;
	double cumulative2n = cumulativeInelastic + n2nMacroXS;
	double cumulative3n = cumulative2n + n3nMacroXS;

	double random = RNG.uniform(0.0, totalMacroXS);
	if (random < cumulativeFission) {
		return InteractionType::nf;
	}
	else if (random < cumulativeElastic) {
		return InteractionType::nel;
	}
	else if (random < cumulativeCapture) {
		return InteractionType::ng;
	}
	else if (random < cumulativeInelastic) {
		return InteractionType::ninl;
	}
	else if (random < cumulative2n) {
		return InteractionType::n2n;
	}
	else if (random < cumulative3n) {
		return InteractionType::n3n;
	}
	else { return InteractionType::nel; }

}

// returns amu (g/mol) value in Constants.cuh
__host__ __device__ double BareSphere::getTargetNucleusMassElastic(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS, unsigned long long seedNo) {
	double mass_UO2 = U235XS->mass * (m_enrichment / 100.0) + U238XS->mass * (1 - m_enrichment / 100.0) + 2 * O16XS->mass;
	double rho_UO2 = (4 * mass_UO2) / (6.02214076e+23 * pow(5.47e-10, 3));
	double atomRho_UO2 = 4 / pow(5.47e-10, 3) * pow(10, 6);		// #/cm^3 
	double atomRho_Mod = ModXS->rho * Constants::N_A / ModXS->mass;

	double volumeFracUO2 = m_fissionableComposition / 100 / rho_UO2;
	double volumeFracC = m_moderatorComposition / 100 / ModXS->rho;

	GnuAMCM RNG(seedNo);
	double totalElMacroXS = getTotalMacroXS(incidentNeutron, U235XS, U238XS, O16XS, ModXS) / pow(10, 24);	// for performance: this function does returns non-barn value
	double o16ElMacroXS = O16XS->getElMicroXSByEnergy(incidentNeutron.m_energy) * atomRho_UO2 * volumeFracUO2 * 2;
	double u235ElMacroXS = U235XS->getElMicroXSByEnergy(incidentNeutron.m_energy) * atomRho_UO2 * volumeFracUO2 * m_enrichment / 100.0 ;
	double u238ElMacroXS = U238XS->getElMicroXSByEnergy(incidentNeutron.m_energy) * atomRho_UO2 * volumeFracUO2 * (1-m_enrichment) / 100.0;
	double c12ElMacroXS = ModXS->getElMicroXSByEnergy(incidentNeutron.m_energy) * atomRho_Mod * volumeFracC;

	double o16CumulativeXS = o16ElMacroXS;
	double u235CumulativeXS = o16CumulativeXS + u235ElMacroXS;
	double u238CumulativeXS = u235CumulativeXS + u238ElMacroXS;
	double c12CumulativeXS = u238CumulativeXS + c12ElMacroXS;

	double random = RNG.uniform(0.0, totalElMacroXS);

	if (random < o16CumulativeXS) {
		return Constants::M_O16;
	}
	else if (random < u235CumulativeXS) {
		return Constants::M_U235;
	}
	else if (random < u238CumulativeXS) {
		return Constants::M_U238;
	}
	else if (random < c12CumulativeXS) {
		return Constants::M_C12;
	}
	else { return Constants::M_C12; }
	
}






__global__ void getNextReactionDistance(BareSphere* CP1, NeutronDistribution* Neutrons, RawCrossSection* U235XS, RawCrossSection* U238XS,
	RawCrossSection* O16XS, RawCrossSection* C12XS, unsigned int numNeutrons, unsigned long long* seedNo, double* distances, int* counter) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numNeutrons) {
		if (Neutrons->m_initialNeutron[idx].isNullified() == false) {
			double totalCrossSection = CP1->getTotalMacroXS(Neutrons->m_initialNeutron[idx], U235XS, U238XS, O16XS, C12XS); // received in cm^{-1} 
			//unsigned long long threadSeed = seedNo[idx] + 2 * (unsigned long long)(idx) + 1;
			seedNo[idx] = (seedNo[idx] * 25214903917ULL + 11ULL) % (1ULL << 48);    // update the seed - GNU AMCM , 2^48 = 1ULL << 48 (bitshift)
			GnuAMCM RNG(seedNo[idx]);
			//distances[idx] = U235XS->getCrossSectionByEnergy(Neutrons->m_initialNeutron[idx].m_energy);
			distances[idx] = -log(RNG.uniform(0.0, 1.0)) / (totalCrossSection * 100); // converted to meters
			Neutrons->m_initialNeutron[idx].UpdateWithLength(distances[idx]);
		}
		else {
			seedNo[idx] = (seedNo[idx] * 25214903917ULL + 11ULL) % (1ULL << 48);
			distances[idx] = 0;
		}

		if (Neutrons->m_initialNeutron[idx].OutofBounds(CP1->m_radius) == true) {
			//printf("Neutron Index %d is Out of Bounds, Thus terminated", idx);
			Neutrons->m_initialNeutron[idx].Nullify();
			//printf("neutron %d nullified\n", idx);
			atomicAdd(counter, 1);
		}
	}

}

__global__ void GlobalStep(NeutronDistribution* Neutrons, int numNeutrons, double timeStep) {
	int idx = threadIdx.x + blockIdx.x + blockDim.x;
	if (idx < numNeutrons) {
		if (Neutrons->m_initialNeutron[idx].isNullified() == false) {
			Neutrons->m_initialNeutron[idx].m_pos = Neutrons->m_initialNeutron[idx].m_dirVec * Neutrons->m_initialNeutron[idx].Velocity() * timeStep;
		}

		if (Neutrons->m_addedNeutron[idx].isNullified() == false) {
			Neutrons->m_addedNeutron[idx].m_pos = Neutrons->m_addedNeutron[idx].m_dirVec * Neutrons->m_addedNeutron[idx].Velocity() * timeStep;
		}
	}
}