#pragma once


#include "Constants.cuh"
#include "RNG.cuh"

#include "Neutron.cuh"
#include "XSParser.cuh"




__host__ __device__ unsigned long long McnpAMCM::gen() {
	//unsigned long long xi_nplus1 = (m_xi * 19073486328125) % static_cast<unsigned long long>(std::pow(2, 48));
	unsigned long long xi_nplus1 = (m_xi * 19073486328125ULL) % (1ULL << 48);
	m_xi = xi_nplus1;
	return xi_nplus1;
}


__host__ __device__ unsigned long long GnuAMCM::gen() {		// returns value between 0 and 2^48-1
	// You can use pow(2,48) but our main concern is SPEED!!! use bitshift to get 2^48:  shift value 1 of type ULL(unsigned long long) to the left 48 times: 1ULL<<48
	//unsigned long long xi_nplus1 = (m_xi * 25214903917 + 11) % static_cast<unsigned long long>(pow(2, 48));
	unsigned long long xi_nplus1 = (m_xi * 25214903917ULL + 11ULL) % (1ULL << 48);
	m_xi = xi_nplus1;
	return xi_nplus1;
}

__host__ __device__ double GnuAMCM::uniform(double lowerLimit, double upperLimit) {
	unsigned long long xi_nplus1 = (m_xi * 25214903917ULL + 11ULL) % (1ULL << 48);
	m_xi = xi_nplus1;
	// linear remapping: random value A between value a,b -> remap to (X,Y): value B = X + (A - a) * (Y - X) / (b - a)
	return lowerLimit + (xi_nplus1) * (upperLimit - lowerLimit) / ((1ULL << 48) - 1);
	// I'm not sure about returning the upperLimit - most RNG uniform distribution returns [Lo, Up): 
	// if the denominator is 2^48, it will return [lowerLimit, upperLimit). but if you do 2^48-1, you will get [lowerLimit, upperLimit].
	// however, utilizing integer division methods will be much faster - it's a bitshift operation.
	// if you want to maximize the speed, use:
	//return lowerlimit + ( xi_nplus1 / (1ULL<<48) ) * (upperLimit - lowerLimit);
}

__host__ __device__ int GnuAMCM::int_dist(int lower, int upper) {
	unsigned long long xi_nplus1 = (m_xi * 2521490317ULL + 11ULL) % (1ULL << 48);
	m_xi = xi_nplus1;
	const unsigned long long range = static_cast<unsigned long long>(upper) - static_cast<unsigned long long>(lower) + 1ULL;
	return static_cast<int>( lower + (xi_nplus1 * range) / (1ULL << 48) );
	// note we devided the xi_nplus1 with 2^48: we dont want to include the upper limit offseted by +1.
	// by rare chance (1/2^48), it will return upper+1: which can be detrimental, since this value will be passed to InteractionType enum - we want 1 to 7, not 8.
}


__host__ __device__ double GnuAMCM::MaxwellDistSample(double a) {
	double U1 = this->uniform(0, 1);
	double U2 = this->uniform(0, 1);
	double U3 = this->uniform(0, 1);
	double c = cos(0.5 * Constants::PI * U3);
	return -a * (log(U1) + (c * c) * log(U2));
}

// Lecture for Fission Spectrum. Visit:
// https://indico.cern.ch/event/145296/contributions/1381141/attachments/136909/194258/lecture24.pdf
__host__ __device__ double GnuAMCM::WattDistSample(double a, double b) {
	//	Chi Spectrum - a.k.a the Watt Distribution:
	//	general Watt Distribution form: Ce^{-E/a} sinh(\sqrt{bE})
	//	P(E) = 0.4865\sinh(\sqrt{2E}) e^{-E} MeV^{-1} <- general form, where C = 0.4865, a = 1, b = 2, hence takes 2 variables, though it will be used with default argument.
	//	This is a Watt Distribution - where the pure thermal emission of Maxwell-Boltzmann Distribution (f(w) = \sqrt{w}e^{-w/a} gets an added prefactor (sinh(\sqrt{bE})).
	//	It's hard to get a Inverse CDF of the P(E), thus we will use the trick that is on the OpenMC Watt Distribution
	//	https://docs.openmc.org/en/stable/_modules/openmc/stats/univariate.html <- at class Watt(Univariate)
	//	E = w + aab/4 + uniform(-1,1) * \sqrt{aab w}, where w = maxwell distribution sample.

	double w = this->MaxwellDistSample(a);
	double u = this->uniform(-1, 1);
	double c = a * a * b / 4.0;

	return w + c + u * sqrt(a * a * b * w) * pow(10, 6);
}

__host__ __device__ int GnuAMCM::fissionNeutronNumber(FissionableElementType fissionElement) {
	if (fissionElement == FissionableElementType::U235) {
		return this->int_dist(2, 3); }	// \nu = 2.4355
	else if (fissionElement == FissionableElementType::U238) { 
		return this->int_dist(2, 4); }	// \nu = 2.819
	else { 
		return 0; 
	}
}

__host__ __device__ double GnuAMCM::GaussianPDF(double inputX, double mean, double stdev) {
	return 1 / (2 * Constants::PI);
}

__host__ __device__ double GnuAMCM::GaussianCDF(double inputX) {
	return 0;
}

//__host__ __device__ double GnuAMCM::GaussainCDFInverse(double random) {
//	return normcdf(1.0);
//}


