#pragma once

namespace Constants {
	// parameters : use gram -> 
	// for mass, variables are declared in amu - convert to g with the amuToG constnat.

	constexpr double amuToGram = 1.660538921e-24;   // 1 amu = 1.66~~e-24 g
	constexpr double amuToKilogram = 1.660538921e-27;	// 1 amu = 1.66~~e-27 kg
	constexpr double M_U235 = 235.0439231;			// based on amu - more like gram/mol
	constexpr double M_U238 = 238.0507826;			// g/mol
	constexpr double M_O16  = 15.9949146;			// g/mol
	constexpr double M_H1   = 1.0078250;			// g/mol
	constexpr double M_C12  = 12.0;					// g/mol
	constexpr double M_Neutron    = 1.008664916;	// g/mol
	
	constexpr double Rho_UO2 = 10.97;				// g/cm^3

	constexpr double N_A    = 6.02214076e+23;		// Avogadro's Number

	constexpr double PI		= 3.141592653589793238462643383279502884197;	//40 fucking digits man it's accurate AF
	constexpr double ElectronC = 1.60217633e-19;
	


}

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

enum SpectrumType {
	default,	//0
	spherical,			// 1
	finiteCylinder,		// 2
	thermalPWR,			// 3
	FBR					// 4
};	

enum InteractionType {
	nel,
	ninl,
	ng,
	nf,
	n2n,
	n3n
};