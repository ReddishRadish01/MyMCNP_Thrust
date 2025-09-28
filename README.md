# MyMCNP

### Welcome to my side project, making my version of Monte Carlo N-Particle Simulator!

Written entirely on [CUDA C/C++](https://docs.nvidia.com/cuda/), and utilizing it's [Thrust Library](https://developer.nvidia.com/thrust), I'm (hopefully) making a neutron-scalable and time-dependent MCNP.  
I am still working on the code. At the moment, I managed to calculate 500,000 neutrons' average distances in 35 milliseconds.


## To compile and run the code:
` $ nvcc -arch=sm_61(or higher) -rdc=true --extended-lambda XSParser.cu RNG.cu Neutron.cu FuelKernel.cu MyMCNP_Thrust.cu -o <output_file_name> && ./<output_file_name> `  


## Description of each files
### `RNG.cu` and `RNG.cuh`
Utilized GNU's Additive Multiplicative Congrugential Method (AMCM) Pseudo-random number generator - in `GnuAMCM` struct.  

$$ \xi_{n+1} = \left\(5deece66d_{(16)} \times \xi_n + b_{(16)} \right\)\ mod\ 2^{48} $$

- This struct has functions that returns a real and integer RNG values within given range (both closed and open range) - `GnuAMCM::uniform`, `GnuAMCM::uniform_open`, `GnuAMCM::int_dist`.  
- For the fission neutrons, the energy of each neutrons follow $\chi$(chi) spectrum, or the Watt Distribution. Function `GnuAMCM::WattDistSample` returns a energy corresponding to the distribution.  

### `XSParser.cu` and `XSParser.cuh`
- XSData contains text files of **KAERI ENDF/B-VIII.0** cross section for certain nuclides. This is loadable, in GPU, at the main code `MyMCNP_Thrust.cu`.
- The cross section is loaded into host with `EnergyCrossSeciton` class, and struct `RawCrossSection` is specifically used to contain GPU-side arrays of each reaction types.
- This class uses binary search and interpolation to evaluate cross section values from our 'discrete' cross section files.

### `Neutron.cu` and `Neutron.cuh` 
- This code is to represent the state of neutrons (struct `Neutron`), and make a array of neutrons (struct `NeutronDistribution`).
- Struct `NeutronThrustHost` and `NeutronThrustDevice` is the implementation of CUDA's Thrust library, where we can utilize the C++ standard vector containers and it's functionality.
- Has algorithms for elastic scattering, direction and energy calculation, etc.

### `FuelKernel.cu` and `FuelKernel.cuh`
- There is only one type of reactors available, which is Bare Sphere reactor (GODIVA): `BareSphere`
- In this struct, we have code to calculate the Macroscopic Cross Section ($\Sigma$): `BareSphere::getTotalMacroXS`, `BareSphere::getFMacroXS`, $\cdots$
- Calculate interaction type: `BareSphere::getInteractionType`
- Calculate the next reaction distance: `BareSphere::getInteractionType` - based on the following Eq:

$$ l = -\frac{\ln(\xi)}{\Sigma_t} \quad \text{where} \quad \xi \in (0,1] $$


### `MyMCNP_Thrust.cu`
Driver code of this project.  
- Beginning of the main file is a boilerplate for CUDA - declaring GPU-side arrays, allocate the size, and copy the data VIA `cudaMemcpy`. Applied for both Neutrons and XS data files.
- `__global__` functions are so called __'kernels'__. They only execute in the device, and specify warp(thread) size per block, and blocks per dim. They are executed in parallel.
