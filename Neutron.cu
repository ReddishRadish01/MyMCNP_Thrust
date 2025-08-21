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
#include "Neutron.cuh"

#include "thrustHeader.cuh"


#include "RNG.cuh"

/* 
enum SpectrumType {
	default,	//0
	spherical,			// 1
	finiteCylinder,		// 2
	thermalPWR,			// 3
	FBR					// 4
};
*/
// moved to Constants.cuh - all enums are moved there.

// NEUTRONS ALWAYS HAVE SI UNITS: Meters and Seconds. For Energy, it's always in MEV!!!!!!!!!!

struct NeutonDistribution;

__host__ __device__ vec3 vec3::operator-(const vec3 vec) const {
	return { x - vec.x, y - vec.y, z - vec.z };
}
__host__ __device__ vec3 vec3::operator+(const vec3 vec) const {
	return { x + vec.x, y + vec.y, z + vec.z };
}
__host__ __device__ vec3 vec3::operator*(const double coeff) const {
	return { x * coeff, y * coeff, z * coeff };
}

__host__ __device__ vec3 vec3::operator/(const double coeff) const {
	return { x / coeff, y / coeff, z / coeff };
} 
__host__ __device__ vec3 vec3::cross(const vec3 vec) const {
	return {
		y * vec.z - z * vec.y,
		z * vec.x - x * vec.z,
		x * vec.y - y * vec.x
	};
}
__host__ __device__ double vec3::dot(const vec3 vec) const {
	return x * vec.x + y * vec.y + z * vec.z;
}

__host__ __device__ double vec3::magnitude() const {
	return sqrt(x * x + y * y + z * z);
}

__host__ __device__ vec3 vec3::normalize() const {
	return {
		x / magnitude(),
		y / magnitude(),
		z / magnitude()
	};
}

__host__ __device__ vec3 vec3::randomUnit(unsigned long long xi) {	// static
	GnuAMCM localRNG(xi);
	//vec3 randUnitVec = { static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()), static_cast<double>(localRNG.gen()) };
	vec3 randUnitVec = { localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0), localRNG.uniform(-1.0, 1.0) };
	return randUnitVec.normalize();
}


__host__ __device__ double Neutron::Velocity() const {
	return sqrt( 2 * m_energy * Constants::ElectronC / (Constants::M_Neutron * Constants::amuToKilogram) );
	// Constants namespace's atom mass always have gram/mol (i.e. amu) - convert it to kg
}

__host__ __device__ vec3 Neutron::VelocityVec() const {
	return m_dirVec * this->Velocity();
} 


__host__ __device__ void Neutron::Nullify() {
	m_pos = { 0.0, 0.0, 0.0 };
	m_dirVec = { 0.0, 0.0, 0.0 };
	m_energy = 0.0;
}

__host__ __device__ bool Neutron::isNullified() const {
	if (m_dirVec.x == 0 && m_dirVec.y == 0 && m_dirVec.z == 0 && m_pos.x == 0 && m_pos.y == 0 && m_pos.z == 0) { return true; }
	else { return false; }
}


// ElASTIC SCATTERING!!!!!!!!!
__device__ void Neutron::ElasticScattering(double targetMass, double* depositedEnergy, unsigned long long seedNo, double targetVelocity) {
	
	GnuAMCM RNG(seedNo);
	vec3 targetVelocityVec( 0, 0, 0 );
	double comVelocity = (this->Velocity() * targetMass * targetVelocity) / (targetMass + Constants::M_Neutron);
	double comNeutronVelocity = this->Velocity() - comVelocity;
	vec3 comNeutronVelocityVec = (this->VelocityVec() + targetVelocityVec * targetMass) / (targetMass + Constants::M_Neutron);

	vec3 comNeutronDirVec = comNeutronVelocityVec.normalize();
	vec3 scatteringVec = vec3::randomUnit(RNG.gen());
	double mu = m_dirVec.dot(scatteringVec) / (m_dirVec.magnitude() * scatteringVec.magnitude());

	double neutronEnergy = this->m_energy * (pow(targetMass, 2) + pow(Constants::M_Neutron, 2) + 2 * targetMass * Constants::M_Neutron * mu) / pow(Constants::M_Neutron + targetMass, 2);
	double recoilEnergy = this->m_energy - neutronEnergy;
	this->m_energy = neutronEnergy;

	atomicAdd(depositedEnergy, recoilEnergy);
}



__device__ void Neutron::Fission(NeutronDistribution* Neutrons, FissionableElementType fissionElement, double* d_depositedEnergy, unsigned long long seedNo) {
	GnuAMCM RNG(seedNo);
	this->Nullify();
	unsigned int promptNeutronNumber = RNG.fissionNeutronNumber(fissionElement);
	for (int i = 0; i < promptNeutronNumber; i++) {
		Neutrons->m_addedNeutron[Neutrons->m_addedNeutronNumber - 1 + i] = Neutron(this->m_pos, vec3::randomUnit(RNG.gen()), RNG.WattDistSample());
	}
	atomicAdd(&Neutrons->m_addedNeutronNumber, promptNeutronNumber);	
	atomicAdd(&Neutrons->m_initialNeutronNumber, -1);
	//atomicAdd(&Neutrons->m_initialNeutronNumber, UINT_MAX);
	//atomicAdd(&Neutrons->m_initialNeutronNumber, (unsigned int)-1);
	// this -1 to the unsigned int variable gives warning: this can be resolved by passing static_cast<unsigned int>(-1) or (unsigned int)-1. 
	// you ask why unsigned int of -1 works? Unsigned int -1 is equivalent to UINT_MAX, and its same as substracting one. 
	atomicAdd(d_depositedEnergy, 200.0E+6);	// 200 MeV deposited per fission
	// I really need some mesh to get the heat distribution. 
}

__device__ void Neutron::Fission(NeutronThrustDevice* Neutrons, FissionableElementType fissionElement, double* d_depositedEnergy, unsigned long long seedNo) {
	GnuAMCM RNG(seedNo);
	this->Nullify();
	unsigned int promptNeutronNumber = RNG.fissionNeutronNumber(fissionElement);
	for (int i = 0; i < promptNeutronNumber; i++) {
		Neutrons->m_addedNeutrons[Neutrons->m_addedNeutronNumber] = Neutron(this->m_pos, vec3::randomUnit(RNG.gen()), RNG.WattDistSample());
		atomicAdd(&(Neutrons->m_addedNeutronNumber), 1u);
	}
}

__device__ void Neutron::InelasticScattering(double* d_depositedEnergy) {
	// i have no idea how to take care of this inelastic scattering
	atomicAdd(d_depositedEnergy, 1.0E+5);
}

__device__ void Neutron::RadiativeCapture(NeutronDistribution* Neutrons, FissionableElementType captureElement, double* d_depositedEnergy) {
	this->Nullify();
	atomicAdd(&Neutrons->m_initialNeutronNumber, -1);
	atomicAdd(d_depositedEnergy, 2.0E+6);
	// again, i dont have any idea of how to calculate the deposited energy. 
}

__device__ void RadiativeCapture(NeutronThrustDevice* Neutrons, FissionableElementType captureElement, double* d_depositedEnergy) {
	
}



__device__ void Neutron::UpdateWithLength(double length) {
	m_pos.x += length * m_dirVec.x;
	m_pos.y += length * m_dirVec.y;
	m_pos.z += length * m_dirVec.z;
}

__device__ bool Neutron::OutofBounds(double distanceLimit) const {	// maybe this need to be in the FuelKernel.cuh
	if (pow(m_pos.x, 2) + pow(m_pos.y, 2) + pow(m_pos.z, 2) > pow(distanceLimit, 2))
		return true;
	else
		return false;
}

__device__ void Neutron::Step(double timeStep) {
	if (this->isNullified() == false) {
		this->m_pos = this->m_dirVec * this->Velocity() * timeStep;
	}
}


__host__ void NeutronDistribution::uniform(double radius, double maxEnergy) { // default argument should be omitted in definition
	m_spectrumType = SpectrumType::spherical;
	GnuAMCM RNG(m_seedNumber);
	for (int i = 0; i < static_cast<int>(m_initialNeutronNumber); i++) {
		//m_initialNeutron[i] = Neutron()
	}
}

__host__ void NeutronDistribution::finiteCylinder(double radius, double height, double maxEnergy) {
	m_spectrumType = SpectrumType::finiteCylinder;
	GnuAMCM RNG(m_seedNumber);
	for (int i = 0; i < static_cast<int>(m_initialNeutronNumber); i++) {

	}
}

__host__ void NeutronDistribution::thermalPWR(double radius, double height) {
	m_spectrumType = SpectrumType::thermalPWR;
	GnuAMCM RNG(m_seedNumber);
	for (int i = 0; i < static_cast<int>(m_initialNeutronNumber); i++) {

	}
}

__host__ void NeutronDistribution::FBR(double radius, double height) {
	m_spectrumType = SpectrumType::FBR;
	GnuAMCM RNG(m_seedNumber);
	for (int i = 0; i < static_cast<int>(m_initialNeutronNumber); i++) {

	}
}

__host__ void NeutronDistribution::uniformSpherical(double radius, double maxEnergy) {
	GnuAMCM RNG(m_seedNumber);
	for (int i = 0; i < static_cast<int>(m_initialNeutronNumber); i++) {
		vec3 posVec(0.0, 0.0, 0.0);
		posVec = posVec.randomUnit(RNG.gen()) * RNG.uniform(0.0, radius);	// max value for GnuAMCM: 2^48. Randomize it by multiplying it to unit vec
		vec3 dirVec(0.0, 0.0, 0.0);
		dirVec = dirVec.randomUnit(RNG.gen());
		double energy = RNG.uniform(0.0, 1.0) * maxEnergy;
		//m_initialNeutron[i] = Neutron(posVec, dirVec, energy);
		m_initialNeutron[i] = Neutron(posVec, dirVec, maxEnergy);

	}
}

__host__ void NeutronDistribution::singleEnergySpherical(double radius, double energy) {
	GnuAMCM RNG(this->m_seedNumber);
	for (int i = 0; i < static_cast<int>(this->m_initialNeutronNumber); i++) {
		vec3 posVec(0.0, 0.0, 0.0);
		posVec = posVec.randomUnit(RNG.gen()) * RNG.uniform(0.0, radius);
		vec3 dirVec(0.0, 0.0, 0.0);
		dirVec = dirVec.randomUnit(RNG.gen());
		this->m_initialNeutron[i] = Neutron(posVec, dirVec, energy);
	}
}

// 3 methods for rearranging the neutron arrays:
// 1. Purely on the original NeutronDistribution struct: but this cannot be parallelized, meaning this should be done on host. This requires cudaMemcpy to DtoH.
// 2. cudaMemcpy DtoH, change the NeutronDistribution struct to Another Host-dedicated struct composed with std::vector (or other modern containers)
//		 do .push_back() or .erase() stuffs, assign it back to original struct, and cudaMemcpy HtoD.
// 3. Use CUDA Thrust library. <- fuck my life im gonna spend next 1 month of my vacation on this fucking library...


__host__ void NeutronDistribution::MergeAndRemoveNeutrons(unsigned int numNeutrons) {
	// first remove all vacant atoms in both initialNeutron and initialNeutron
	int initialNeutronOffset = m_initialNeutronNumber;
	for (int i = 0; i < numNeutrons; i++) {
		if (m_initialNeutron[i].isNullified()) {	// locate the nullified neutron
			for (int j = 0; j < numNeutrons; j++) {
				m_initialNeutron[i] = m_initialNeutron[initialNeutronOffset - 1];
			}
		}
	}

	if (m_initialNeutronNumber > 0.5 * numNeutrons) {
		int offset = static_cast<int>(numNeutrons - m_initialNeutronNumber);
		for (int i = 0; i < offset; i++) {
			m_initialNeutron[m_initialNeutronNumber - 1 + i] = m_addedNeutron[m_addedNeutronNumber - 1 - offset + i];
			m_addedNeutron[m_addedNeutronNumber - 1 - offset + i].Nullify();
		}
	}
}
// this need something to remove the nullified neutrons out of the NeutronDistirbution; 



__host__ __device__ bool isAddedNeutronAlmostFull() {
	return true;
}

__host__ void NeutronThrustHost::uniformSpherical(double radius, double maxEnergy) {
	GnuAMCM RNG(m_seedNumber);
	for (auto& elemNeutron : this->m_neutrons) {
		vec3 posVec(0.0, 0.0, 0.0);
		posVec = vec3::randomUnit(RNG.gen()) * RNG.uniform(0.0, radius);
		vec3 dirVec(0.0, 0.0, 0.0);
		dirVec = vec3::randomUnit(RNG.gen());
		double energy = RNG.uniform(0.0, 1.0) * maxEnergy;
		elemNeutron = Neutron(posVec, dirVec, energy);
	}
}

__host__ void NeutronThrustHost::singleEnergySpherical(double radius, double energy) {
	GnuAMCM RNG(this->m_seedNumber);
	for (auto& elemNeutron : this->m_neutrons) {
		vec3 posVec(0.0, 0.0, 0.0);
		posVec = vec3::randomUnit(RNG.gen()) * RNG.uniform(0.0, radius);
		vec3 dirVec(0.0, 0.0, 0.0);
		dirVec = vec3::randomUnit(RNG.gen());
		elemNeutron = Neutron(posVec, dirVec, energy);
	}
}



__host__ NeutronThrustDevice NeutronThrustHost::HtoD(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons) {
	return NeutronThrustDevice{ thrust::raw_pointer_cast(d_Neutrons.data()), thrust::raw_pointer_cast(d_addedNeutrons.data()), 
		(unsigned int)(d_Neutrons.size()), (unsigned int)(d_addedNeutrons.size()), m_seedNumber, m_spectrumType };
}

__host__ void NeutronThrustHost::DtoH(thrust::device_vector<Neutron>& d_Neutrons, thrust::device_vector<Neutron>& d_addedNeutrons) {
	cudaDeviceSynchronize();
	// this resizing part takes up so much time.
	this->m_neutrons.resize(d_Neutrons.size());
	thrust::copy(d_Neutrons.begin(), d_Neutrons.end(), this->m_neutrons.begin());
	thrust::copy(d_addedNeutrons.begin(), d_addedNeutrons.end(), this->m_addedNeutrons.begin());
}

__host__ void NeutronThrustHost::MergeNeutrons() {
	
}


__host__ void NeutronThrustManager::MergeNeutron(thrust::device_vector<Neutron>& d_neutrons, thrust::device_vector<Neutron>& d_addedNeutrons) {
	
	auto e1 = thrust::remove_if(d_neutrons.begin(), d_neutrons.end(), [] __device__ (Neutron const& n) { return n.isNullified(); });
	d_neutrons.erase(e1, d_neutrons.end());

	auto e2 = thrust::remove_if(d_addedNeutrons.begin(), d_addedNeutrons.end(), [] __device__ (Neutron const& n) { return n.isNullified(); });
	d_addedNeutrons.erase(e2, d_addedNeutrons.end());
	
	size_t neutronN = d_neutrons.size();
	size_t addedNeutronN = d_addedNeutrons.size();

	d_neutrons.reserve(neutronN + addedNeutronN);
	thrust::copy(d_addedNeutrons.begin(), d_addedNeutrons.end(), std::back_inserter(d_neutrons));
	d_addedNeutrons.clear();

}
	
__host__ void NeutronThrustManager::MergeNeutron() {
	auto e1 = thrust::remove_if(d_neutrons.begin(), d_neutrons.end(), [] __device__ (Neutron const& n) { return n.isNullified(); });
	d_neutrons.erase(e1, d_neutrons.end());

	auto e2 = thrust::remove_if(d_addedNeutrons.begin(), d_addedNeutrons.end(), [] __device__ (Neutron const& n) { return n.isNullified(); });
	d_addedNeutrons.erase(e2, d_addedNeutrons.end());

	size_t neutronN = d_neutrons.size();
	size_t addedNeutronN = d_addedNeutrons.size();

	d_neutrons.reserve(neutronN + addedNeutronN);
	thrust::copy(d_addedNeutrons.begin(), d_addedNeutrons.end(), std::back_inserter(d_neutrons));
	d_addedNeutrons.clear();
}
