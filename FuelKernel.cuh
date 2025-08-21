#pragma once

#include "Constants.cuh"

#include "RNG.cuh"
#include "XSParser.cuh"
#include "Neutron.cuh"



/*
enum FissionableElementType {
	U235,
	U238,
	Pu239,
	Pu241,
	Th232,
	U235nU238
};

enum ModeratorType {
	Graphite,
	Boron,
	LightWater,
	HeavyWater,
	DilutedBoron
};
*/


struct BareSphere {
	double m_radius;
	FissionableElementType m_FissionableElementType;
	double m_enrichment;	// percent
	double m_fissionableComposition;	// mass percentage	
	ModeratorType m_ModeratorType;	
	double m_moderatorComposition;		// mass percentage



	BareSphere(double radius, FissionableElementType fissionableElementType, double enrichment, double fissionableComposition, 
				ModeratorType moderatorType)
		: m_radius(radius), m_FissionableElementType(fissionableElementType), m_enrichment(enrichment), m_fissionableComposition(fissionableComposition),
		  m_ModeratorType(moderatorType)
	{
		m_moderatorComposition = 100.0 - fissionableComposition;
	}

	__host__ __device__ double getTotalMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double getFMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double getGMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double getElMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double getInlMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double get2nMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double get3nMacroXS(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS);

	__host__ __device__ double getUO2XSbyEnergy(RawCrossSection* U235XS_ptr, RawCrossSection* U238XS_ptr, RawCrossSection* O16XS_ptr, double* enrichment, double IncidentEnergy);

	__host__ __device__ InteractionType getInterationType(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS, unsigned long long seedNo);


	__host__ __device__ double getTargetNucleusMassElastic(Neutron incidentNeutron, RawCrossSection* U235XS, RawCrossSection* U238XS, RawCrossSection* O16XS, RawCrossSection* ModXS, unsigned long long seedNo);
	
};

__global__ void getNextReactionDistance(BareSphere* CP1, NeutronDistribution* Neutrons, RawCrossSection* U235XS, RawCrossSection* U238XS,
	RawCrossSection* O16XS, RawCrossSection* C12XS, unsigned int numNeutrons, unsigned long long* seedNo, double* distances, int* counter);


__global__ void GlobalStep(NeutronDistribution* Neutrons, int numNeutrons, double timeStep = 1.0e-8);