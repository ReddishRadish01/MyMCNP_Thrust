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


#include "constants.cuh"

#include "RNG.cuh"
#include "Neutron.cuh"

/*
enum InteractionType {
	nel,
	ninl,
	ng,
	nf,
	n2n,
	n3n
};
*/

struct EnergyCrossSection {
	double m_energy;
	double m_crossSection;
	std::string m_crossSectionStr;
	std::string m_XSDataFilename;
	std::vector< std::pair<double, double> > m_EnergyCrossSection;
	std::vector<double> v_energy, v_crossSection;

	EnergyCrossSection(const std::string XSDataFilename) // CONSTRUCTOR LETS GOOOOOOO
		: m_energy{ 0 }, m_crossSection{ 0 }, m_XSDataFilename{ XSDataFilename }	// Member initializer list
	{
		//m_XSDataFilename = XSDataFilename;
		std::cout << "Loading Cross Section from " << m_XSDataFilename;

		// Parses through the given data file, and stores energy / cross section in the vector.
		std::ifstream XSDataFile(XSDataFilename);
		if (XSDataFile.fail()) {
			std::cout << "\n**ERROR!: File " << XSDataFilename << " doesn't exist! or intentionally omitted \n";
		}
		else {
			std::cout << " - Successful\n";
		}
		while (XSDataFile >> m_energy >> m_crossSectionStr) {
			// The text file always contain <br> on the cross section data. Below code is to erase '<br>' from the string and cast to double.
			size_t pos = m_crossSectionStr.find("<br>");
			if (pos != std::string::npos) {
				m_crossSectionStr.erase(pos, 4); // Erase 4 characters - '<br>'
			}

			// convert the cleaned string to a double - we will use stod function.
			m_crossSection = std::stod(m_crossSectionStr);

			// store each values to energyCrossSection
			m_EnergyCrossSection.emplace_back(m_energy, m_crossSection);

			//std::cout << energy << " " << crossSection << "\n"; // omitted to optimize the program
			v_energy.push_back(m_energy);
			v_crossSection.push_back(m_crossSection);

		}
		XSDataFile.close();
	}

	std::string getNameOfTextFile();

	double getCrossSectionByIndex(int index);
	double getEnergyByIndex(int index);

	double getSizeOfVectorArrayinMB();

	// finds the value stored in the vector and outputs the cross section.
	void getCrossSectionByEnergy(const double inputEnergy);

	void exportToVectors(std::vector<double>& energies, std::vector<double>& crossSections) const;
};

struct ArrayCrossSection {
	double* energy;
	double* crossSection;
	unsigned int size;
	//size_t objectSize;

	ArrayCrossSection(const std::vector<double>& h_energy, const std::vector<double> h_crossSection)
		: size(static_cast<unsigned int>(h_energy.size()))
	{
		energy = new double[size];
		crossSection = new double[size];
		std::memcpy(energy, h_energy.data(), size * sizeof(double));
		std::memcpy(crossSection, h_crossSection.data(), size * sizeof(double));
		//objectSize = 2 * sizeof(double) * size + sizeof(unsigned int) + sizeof(size_t);
	}

	ArrayCrossSection(EnergyCrossSection energyCrossSection)
		: size(static_cast<unsigned int>(energyCrossSection.v_energy.size()))
	{
		energy = new double[size];
		crossSection = new double[size];
		std::memcpy(energy, energyCrossSection.v_energy.data(), size * sizeof(double));
		std::memcpy(crossSection, energyCrossSection.v_crossSection.data(), size * sizeof(double));
		//objectSize = 2 * sizeof(double) * size + sizeof(unsigned int) + sizeof(size_t);
	}

	~ArrayCrossSection() {
		delete[] energy;
		delete[] crossSection;
	}

	__device__ double getCrossSectionByEnergy(double incidentEnergy);
	
	
};

// This is never used - redundant struct.
struct GatheredCrossSection {
	ArrayCrossSection m_XS_ntot;
	ArrayCrossSection m_XS_nf;
	ArrayCrossSection m_XS_nel;
	ArrayCrossSection m_XS_ng;
	ArrayCrossSection m_XS_ninl;
	ArrayCrossSection m_XS_n2n;
	ArrayCrossSection m_XS_n3n;
	//size_t objectSize;

	GatheredCrossSection(const std::string XSFolderName)
		: m_XS_ntot(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_tot_CrossSection.txt")),
		  m_XS_nf(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_f_CrossSection.txt")),
		  m_XS_nel(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_el_CrossSection.txt")),
		  m_XS_ng(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_g_CrossSection.txt")),
		  m_XS_ninl(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_inl_CrossSection.txt")),
		  m_XS_n2n(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_2n_CrossSection.txt")),
		  m_XS_n3n(EnergyCrossSection("XSData/" + XSFolderName + "/" + XSFolderName + "-n_3n_CrossSection.txt"))
	{
		//objectSize = m_XS_ntot.objectSize + m_XS_nf.objectSize + m_XS_nel.objectSize + m_XS_ng.objectSize + m_XS_ninl.objectSize
		//	         + m_XS_n2n.objectSize + m_XS_n3n.objectSize + sizeof(size_t);
	}

	//__device__ double getElementCrossSectionByEnergy(double incidentEnergy);
};

__device__ double getCrossSectionByEnergy(double incidentEnergy, const double* energy, const double* crossSection, int numOfEntries);



struct RawCrossSection {
	double* ntot_energy, * ntot_XS;
	double* nf_energy,   * nf_XS;	  
	double* nel_energy,  * nel_XS;  
	double* ng_energy,   * ng_XS;  
	double* ninl_energy, * ninl_XS;  
	double* n2n_energy,  * n2n_XS;  
	double* n3n_energy,  * n3n_XS;  
	unsigned int ntot_size, nf_size, nel_size, ng_size, ninl_size, n2n_size, n3n_size;
	double mass;
	double rho = 0.0;

	RawCrossSection(const std::string XSFolderName, double mass)
		: mass(mass)
	{
		EnergyCrossSection eXS_ntot("XSData/" + XSFolderName + "/" + XSFolderName + "-n_tot_CrossSection.txt");
		EnergyCrossSection eXS_nf("XSData/" + XSFolderName + "/" + XSFolderName + "-n_f_CrossSection.txt");
		EnergyCrossSection eXS_nel("XSData/" + XSFolderName + "/" + XSFolderName + "-n_el_CrossSection.txt");
		EnergyCrossSection eXS_ng("XSData/" + XSFolderName + "/" + XSFolderName + "-n_g_CrossSection.txt");
		EnergyCrossSection eXS_ninl("XSData/" + XSFolderName + "/" + XSFolderName + "-n_inl_CrossSection.txt");
		EnergyCrossSection eXS_n2n("XSData/" + XSFolderName + "/" + XSFolderName + "-n_2n_CrossSection.txt");
		EnergyCrossSection eXS_n3n("XSData/" + XSFolderName + "/" + XSFolderName + "-n_3n_CrossSection.txt");

		ntot_size = static_cast<unsigned int>(eXS_ntot.v_energy.size());
		nf_size   = static_cast<unsigned int>(eXS_nf.v_energy.size());
		nel_size  = static_cast<unsigned int>(eXS_nel.v_energy.size());
		ng_size   = static_cast<unsigned int>(eXS_ng.v_energy.size());
		ninl_size = static_cast<unsigned int>(eXS_ninl.v_energy.size());
		n2n_size  = static_cast<unsigned int>(eXS_n2n.v_energy.size());
		n3n_size  = static_cast<unsigned int>(eXS_n3n.v_energy.size());

		ntot_energy = new double[ntot_size]; ntot_XS = new double[ntot_size];
		std::memcpy(ntot_energy, eXS_ntot.v_energy.data(), ntot_size * sizeof(double));
		std::memcpy(ntot_XS, eXS_ntot.v_crossSection.data(), ntot_size * sizeof(double));
		
		nf_energy = new double[nf_size]; nf_XS = new double[nf_size];
		std::memcpy(nf_energy, eXS_nf.v_energy.data(), nf_size * sizeof(double));
		std::memcpy(nf_XS, eXS_nf.v_crossSection.data(), nf_size * sizeof(double));

		nel_energy = new double[nel_size]; nel_XS = new double[nel_size];
		std::memcpy(nel_energy, eXS_nel.v_energy.data(), nel_size * sizeof(double));
		std::memcpy(nel_XS, eXS_nel.v_crossSection.data(), nel_size * sizeof(double));

		ng_energy = new double[ng_size]; ng_XS = new double[ng_size];
		std::memcpy(ng_energy, eXS_ng.v_energy.data(), ng_size * sizeof(double));
		std::memcpy(ng_XS, eXS_ng.v_crossSection.data(), ng_size * sizeof(double));

		ninl_energy = new double[ninl_size]; ninl_XS = new double[ninl_size];
		std::memcpy(ninl_energy, eXS_ninl.v_energy.data(), ninl_size * sizeof(double));
		std::memcpy(ninl_XS, eXS_ninl.v_crossSection.data(), ninl_size * sizeof(double));

		n2n_energy = new double[n2n_size]; n2n_XS = new double[n2n_size];
		std::memcpy(n2n_energy, eXS_n2n.v_energy.data(), n2n_size * sizeof(double));
		std::memcpy(n2n_XS, eXS_n2n.v_crossSection.data(), n2n_size * sizeof(double));

		n3n_energy = new double[n3n_size]; n3n_XS = new double[n3n_size];
		std::memcpy(n3n_energy, eXS_n3n.v_energy.data(), n3n_size * sizeof(double));
		std::memcpy(n3n_XS, eXS_n3n.v_crossSection.data(), n3n_size * sizeof(double));
	}

	void setDensity(double density) { rho = density; }


	__host__ __device__ double getTotalMicroXSByEnergy(double incidentEnergy);	// object method works.
	__host__ __device__ double getFMicroXSByEnergy(double incidentEnergy);
	__host__ __device__ double getGMicroXSByEnergy(double incidentEnergy);
	__host__ __device__ double getElMicroXSByEnergy(double incidentEnergy);
	__host__ __device__ double getInlMicroXSByEnergy(double incidentEnergy);
	__host__ __device__ double get2nMicroXSByEnergy(double incidentEnergy);
	__host__ __device__ double get3nMicroXSByEnergy(double incidentEnergy);


	__host__ __device__ InteractionType getInteractionModeByEnergy(double incidentEnergy, unsigned long long SeedNo);
};


__host__ __device__ double getCrossSectionTot(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSectionInl(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSectionEl(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSectionF(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSectionG(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSection2N(RawCrossSection* rawXS_ptr, double incidentEnergy);
__host__ __device__ double getCrossSection3N(RawCrossSection* rawXS_ptr, double incidentEnergy);
