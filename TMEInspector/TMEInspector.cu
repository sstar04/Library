/*
 * TMEInspector.cu
 *
 *  Created on: 25/mag/2018
 *      Author: Sabrina
 */

#include <math.h> 
#include <stdio.h>
#include <algorithm>    
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "TMEInspector.h"
#include "TMEInspector_kernel.cu"
#include "Filter.h"


TMEInspector::TMEInspector(){
	// TODO Auto-generated constructor stub

}

TMEInspector::TMEInspector(TGSimulator& tgSim): m_tgSim(tgSim){

   // Allocate Unified Memory -- accessible from CPU or GPU
   //printf("numero totale di cellule %d", m_tgSim.Get_numCells() );
      cudaMallocManaged(&m_border, m_tgSim.Get_numCells()*sizeof(bool));
      cudaDeviceSynchronize();

      //initialize Filter instance
      m_filter = Filter(m_tgSim.Get_VCells(),m_tgSim.Get_numCells());
      printf("Sono in TMEInspector \n");
}

TMEInspector::~TMEInspector() {
	
	cudaFree(m_border);
}


bool* TMEInspector::Search_BorderCells(){


	unsigned int numCells = m_tgSim.Get_numCells();

	// Launch kernel on numCells elements on the GPU
	int blockSize = 16;
	int numBlocks = ceil(numCells/blockSize)+1;
	printf("numero di blocchi %d - numeri di threads: %d", numBlocks, blockSize);
	//   init<<<1, numCells>>>(numCells, m_border);
	init<<<numBlocks, blockSize>>>(numCells, m_border);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	return m_border;
}


bool* TMEInspector::Search_BorderCellsTh(){


    unsigned int numCells = m_tgSim.Get_numCells();

    //inizializzo m_border con trust utilizzando un wrapper
    
    // wrap raw pointer with a device_ptr 
    thrust::device_ptr<bool> dev_border = thrust::device_pointer_cast(m_border);

    // use device_ptr in Thrust algorithms
    thrust::fill(dev_border, dev_border + numCells , 1);
  
    //init<<<numBlocks, blockSize>>>(numCells, m_border);   // !non funziona
    
    //Wait for GPU to finish before accessing on host  
    cudaDeviceSynchronize();
      
   return m_border;
}



void TMEInspector::Search_VecBorderCells(std::vector<bool>& IsonAS){

	//FindBorder(). In attesa di implementare questa funzione privata mi limito semplicemente a inizializzare il vettore

	unsigned int numCells = m_tgSim.Get_numCells();
	// Launch kernel on numCells elements on the GPU
	int blockSize = 16;
	int numBlocks = ceil(numCells/blockSize)+1;
	init<<<numBlocks, blockSize>>>(numCells, m_border);

	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	std::copy(m_border, m_border+numCells, IsonAS.begin());
}


void TMEInspector::STARAlgorithm(std::vector<bool>& IsonAS,
									float a )//! algorithm's runtime parameter (USide= a*r_max)
{

	  //Calculate Extended bounding box
	  float U_side = (a*m_tgSim.Get_rmax());


#ifdef DEBUG
	  std::cout << "**** TMEInspector::STARAlgorithm says:" << std::endl;
	  std::cout << "runtime parameter 'a' of STARAlgorithm is set to: " << a << std::endl;
	  std::cout << "the maximum value of the cell's radius is: " << m_tgSim.Get_rmax() << std::endl;
	  std::cout << "the side of the small cubic box is: " <<a*m_tgSim.Get_rmax() << "micron" << std::endl<< std::endl;
#endif

	  m_filter.Calculate_ExtendedBoundingBox(U_side,2);

	  //m_filter.TestNewData();
	  m_filter.ReorderCellsData_by_HashValue();

	return;

}
