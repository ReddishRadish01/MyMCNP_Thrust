#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <chrono>

#include "Constants.cuh"

#include "RNG.cuh"
#include "XSParser.cuh"

#include "thrustHeader.cuh"



extern int ratioDivider;


struct NeutronDistribution;	// Forward-declare of the struct - one of the Neutron's member method (Fission) has to use NeutronDistribution, hence here.
struct NeutronThrustDevice;

struct vec3 {
	double x, y, z;

	__host__ __device__ vec3()
		: x(), y(), z() {}

	__host__ __device__ vec3(double xx, double yy, double zz)
		: x(xx), y(yy), z(zz)
	{}


	__host__ __device__ vec3 operator-(const vec3 vec) const;
	__host__ __device__ vec3 operator+(const vec3 vec) const;
	__host__ __device__ vec3 operator*(const double coeff) const;
	__host__ __device__ vec3 operator/(const double coeff) const;


	__host__ __device__ vec3 cross(const vec3 vec) const;
	__host__ __device__ double dot(const vec3 vec) const;

	__host__ __device__ double magnitude() const;

	__host__ __device__ vec3 normalize() const;

	__host__ __device__ static vec3 randomUnit(unsigned long long xi);
};

struct Neutron {
	vec3 m_pos;			// METERS !!!
	vec3 m_dirVec;		// UNIT VECTOR
	double m_energy;	// eV !!!

	__host__ __device__ Neutron()
		: m_pos({ 0.0, 0.0, 0.0 }), m_dirVec({ 0.0, 0.0, 0.0 }), m_energy(0.0) {}

	__host__ __device__ Neutron(vec3 pos, vec3 dirVec, double energy)
		: m_pos(pos), m_dirVec(dirVec), m_energy(energy)
	{}

	__host__ __device__ double Velocity() const;
	__host__ __device__ vec3 Neutron::VelocityVec() const;

	__host__ __device__ void Nullify();
	__host__ __device__ bool isNullified() const;
	

	__device__ void ElasticScattering(double targetMass, double* d_depositedEnergy, unsigned long long seedNo, double targetVelocity = 0.0);

	// you have to declare
	// __device__ int g_promptIndex = 0; in the global scope, outside of all functions.
	__device__ void Fission(NeutronDistribution* Neutrons, FissionableElementType fissionElement, double* d_depositedEnergy, unsigned long long seedNo);
	__device__ void Fission(NeutronThrustDevice* Neutrons, FissionableElementType fissionElement, double* d_depositedEnergy, unsigned long long seedNo);

	__device__ void InelasticScattering(double* d_depositedEnergy);

	__device__ void RadiativeCapture(NeutronDistribution* Neutrons, FissionableElementType captureElement, double* d_depositedEnergy);

	__device__ void UpdateWithLength(double length);

	__device__ bool OutofBounds(double distanceLimit) const;

	__device__ void Step(double timeStep = 1.0E-9);
};


struct NeutronDistribution {
	Neutron* m_initialNeutron;
	unsigned int m_initialNeutronNumber;
	unsigned long long m_seedNumber;
	SpectrumType m_spectrumType;
	Neutron* m_addedNeutron;
	unsigned int m_addedNeutronNumber;


	__host__ NeutronDistribution(unsigned int neutronNumber, unsigned long long seedNumber, enum SpectrumType spectrumType = SpectrumType::default)
		: m_initialNeutron(new Neutron[neutronNumber]), m_initialNeutronNumber(neutronNumber), m_seedNumber(seedNumber), m_spectrumType(spectrumType),
		m_addedNeutron(new Neutron[neutronNumber]), m_addedNeutronNumber(neutronNumber)
	{
		//if (spectrumType == SpectrumType::default) {
		//	  uniform(1, 2e+6);
		//}
	}

	__host__ void uniformSpherical(double radius, double maxEnergy = 2e+6);

	__host__ void singleEnergySpherical(double radius, double energy = 0.0253);

	__host__ void finiteCylinder(double radius, double height, double maxEnergy = 2e+6);

	__host__ void thermalPWR(double radius, double height);

	__host__ void FBR(double radius, double height);

	__host__ void uniform(double radius, double maxEnergy = 2e+6);

	//__host__ __device__ void MergeNeutron(Neutron* initialNeutrons, unsigned int initialNeutronSize, Neutron* addedNeutrons, unsigned int addedNeutronSize);
	__host__ __device__ void MergeAndRemoveNeutrons(unsigned int numNeutrons);

	
};


struct NeutronThrustDevice {
	Neutron* m_neutrons;
	Neutron* m_addedNeutrons;
	unsigned int m_neutronNumber;
	unsigned int m_addedNeutronNumber;
	unsigned long long m_seedNumber;
	SpectrumType m_spectrumType;
	

	__host__ __device__ NeutronThrustDevice(Neutron* neutrons, Neutron* addedNeutrons, unsigned int neutronNumber, unsigned int addedNeutronNumber, unsigned long long seedNumber, SpectrumType spectrumType)
		: m_neutrons(neutrons), m_addedNeutrons(addedNeutrons), m_neutronNumber(neutronNumber), m_addedNeutronNumber(addedNeutronNumber),
		m_seedNumber(seedNumber), m_spectrumType(spectrumType)
	{}

	

};


struct NeutronThrustHost {
	thrust::host_vector<Neutron> m_neutrons;
	thrust::host_vector<Neutron> m_addedNeutrons;
	unsigned long long m_seedNumber;
	SpectrumType m_spectrumType;

	__host__ NeutronThrustHost(int initialSize, unsigned long long seedNumber, SpectrumType spectrumType)
		: m_neutrons(initialSize), m_addedNeutrons(initialSize / ratioDivider), m_seedNumber(seedNumber), m_spectrumType(spectrumType)
	{
		// m_neutrons(initialSize);	<- this is wrong. If you really want to omit it in the initializer list, you do:
		// m_neutrons.resize(initialSize);
	}


	// Use this after making device vectors. e.g. @ main:
	// thrust::device_vector<Neutron> d_Neutrons = <NeutronThrustHostName>.m_neutrons;
	__host__ NeutronThrustDevice HtoD(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons);

	__host__ void DtoH(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons);
	// should i DtoH with the nullified neutron removed? or what
	__host__ void uniformSpherical(double radius, double maxEnergy = 2.0e+6);

	__host__ void singleEnergySpherical(double raidus, double energy = 0.0253);

	__host__ void MergeNeutrons();

	
};


// utility class for merging, dropping nullified neutrons.
struct NeutronThrustManager {
	thrust::device_vector<Neutron> d_neutrons;
	thrust::device_vector<Neutron> d_addedNeutrons;
	unsigned long long m_seedNumber;
	SpectrumType m_spectrumType;

	// guess its kinda redundnat?
	NeutronThrustManager(thrust::device_vector<Neutron>& d_neutrons, thrust::device_vector<Neutron>& d_addedNeutrons, unsigned long long seedNo, SpectrumType spectrumType) 
		: d_neutrons(d_neutrons), d_addedNeutrons(d_addedNeutrons), m_seedNumber(seedNo), m_spectrumType(spectrumType)
	{}

	__host__ static void MergeNeutron(thrust::device_vector<Neutron>& d_neutrons, thrust::device_vector<Neutron>& d_addedNeutrons);
	__host__ void MergeNeutron();
};