/*
 * Filter.cu
 *
 *  Created on: 20/lug/2018
 *      Author: sabry
 */

#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
/*
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/replace.h>
#include <thrust/distance.h>
*/
#include "macro.h"
#include "Filter.h"
#include "Filter_kernel.cu"
#include "Utility.cuh"



Filter::Filter(){
	m_VCells = NULL;
	m_NCells = 0;
}

Filter::Filter(point4d* VCells, uint NCells):m_VCells(VCells), m_NCells(NCells){

	// Allocate Unified Memory -- accessible from CPU or GPU
	cudaMallocManaged(&m_sortedVCells, NCells*sizeof(point4d));
	cudaDeviceSynchronize();
	cudaMallocManaged(&m_cellsHash, NCells*sizeof(unsigned int));
	cudaDeviceSynchronize();
	cudaMallocManaged(&m_cellsIndex, NCells*sizeof(unsigned int));
	cudaDeviceSynchronize();



/*
	// Initialize data on host
	   for(int i = 0; i < NCells; i++){

		   m_sortedVCells[i] = point4d(0,0,0,0);
		   m_cellsHash[i]  = 0;
		   m_cellsIndex[i]  = 0;
	   }

	   std::cout<<"sono in Filter constructor" << std::endl;
	   for(int i=0 ; i<m_NCells; i++){
	   			std::cout << m_cellsHash[i] <<", " ;

	   		}*/
}

Filter::~Filter() {
	// Free memory
	cudaFree(m_sortedVCells);
	cudaFree(m_cellsHash);
	cudaFree(m_cellsIndex);
}

/*loaded =
Filter& Filter::operator=(const Filter& f)
{
	m_VCells = f.m_VCells;
	m_NCells = f.m_NCells;

	return *this;
}*/

void Filter::Print_coordinates(){

	std::cout << "sono in filer, il numero totale di cellule e`:" << m_NCells << std::endl;


	for(int i = 0; i < m_NCells; i++){
	    m_VCells[i].printCoord();
	   }
}


void Filter::CopyOnGPU_SubParameter(){


	SubParams temp;  //non posso definirla come variabile membro altrimenti avrei bisogno di un construttore di default (che e` al momento commentato).
	                  // inserendo un construttore di default l'API cudaMemcpyToSymbol da errore.
	temp.s = m_s;
	temp.NsmallBox = m_NsmallBox;
	temp.bboxSide = m_bboxSide;
	temp.sboxSide = m_sboxSide;
	temp.originTranslation.x = m_originTranslation.x;
	temp.originTranslation.y = m_originTranslation.y;
	temp.originTranslation.z = m_originTranslation.z;
	//inizialize a variable on the GPU's constant memory

	cudaMemcpyToSymbol(GPUparams, &temp, sizeof(SubParams)); // da verifica se passa i valori giusti

#ifdef DEBUG
	std::cout <<"****  Filter::CopyOnGPU_SubParams says:" << std::endl;
	temp.PrintValues();
	std::cout<< "transfer Subdivision parameter on GPU constant memory:" << std::endl;
	//GPUparams.PrintValues();
#endif


}

bbox Filter::Calculate_BoundingBox(){

	//1. Defining the bounding box
	// This step is implemented using thrust library using the code provided in thrust examples at
	// https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu

	cudaEvent_t start, stop;  // create cuda event handles
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<point4d> dev_VCells = thrust::device_pointer_cast(m_VCells);

	//initial bounding box contain first point
	bbox init = bbox(dev_VCells[0],dev_VCells[0]);

	//transform operations
	bbox_transform unary_op;

	//binary reduction
	bbox_reduction binary_op;

	float bbox_time = 0.0f;

	//start record time
	cudaEventRecord(start, 0);
	bbox result = thrust::transform_reduce(dev_VCells, dev_VCells+m_NCells, unary_op, init, binary_op);
	cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	CUDA_CALL(cudaEventElapsedTime(&bbox_time, start, stop));

#ifdef DEBUG
	std::cout <<"****  Filter::Calculate_BoundingBox says:" << std::endl;
	std::cout <<"bounding box execution time: " << bbox_time << std::endl;

	std::cout << "Exact bounding box" << std::endl;
	std::cout << "(" << result.first.x << "," << result.first.y << "," << result.first.z << ")" ;
	std::cout << "(" << result.second.x << "," << result.second.y << "," << result.second.z  <<")" << std::endl<< std::endl;

#endif

	return result;
}

//! Enlarge the bounding box adding two layers of smallBoxes;
void Filter::Calculate_ExtendedBoundingBox(float USide,
		                                          int Nlayers ) //! number of empty layer
{
	bbox a = Calculate_BoundingBox();

	m_sboxSide = USide;

	//calculate the sides of the box considering that bbox=(min, max);
	point4d D = a.second - a.first;

	//calculate the new points for the extended bounding box
	//1.find the longer side of the Bbox
	m_bboxSide =0;
	if(D.x> m_bboxSide) m_bboxSide=D.x;
	if(D.y> m_bboxSide) m_bboxSide=D.y;
	if(D.z> m_bboxSide) m_bboxSide=D.z;

	int s_temp;
	s_temp = ceil(m_bboxSide/m_sboxSide); //number of subdivision

	//2. Enlarge the bboxSide and add Nlayer empty border
	m_bboxSide = m_sboxSide*s_temp + Nlayers*m_sboxSide;

	//set parameters
	m_s = s_temp+Nlayers;
	m_NsmallBox = m_s*m_s*m_s; //total number of small boxes


	//3.Define the new couple of vertex corresponding to the extended bounding box
	point4d max(m_bboxSide, m_bboxSide , m_bboxSide , 0);

	//point4d Increment = D - max; //is the increment that each side needs in order to reach the maximum value.
	point4d Increment = max-D;

	//calculate the new bbox(bmin,bmax)
	point4d bmin(sum(a.first.x,Increment.x/2), sum(a.first.y, Increment.y/2), sum(a.first.z,Increment.z/2), 0);

	point4d bmax(sum(a.second.x,Increment.x/2), sum(a.second.y, Increment.y/2), sum(a.second.z,Increment.z/2), 0);

	//Coordinate that will be used to translate the bbox to the positive semispace
	m_originTranslation.x = bmin.x;
	m_originTranslation.y = bmin.y;
	m_originTranslation.z = bmin.z;

	//Set the member bbox data to the new value
	m_bbox3D = bbox(bmin, bmax);

#ifdef DEBUG
	    std::cout << "**** Filter::Calculate_ExtendedBoundingBox says:" << std::endl;
    	std::cout << "Extended bounding box" << std::endl;
    	std::cout << "(" << m_bbox3D.first.x << "," << m_bbox3D.first.y << "," << m_bbox3D.first.z << ")" ;
    	std::cout << "(" <<  m_bbox3D.second.x << "," << m_bbox3D.second.y << "," << m_bbox3D.second.z  <<")" <<std::endl;

    	std::cout << "the side of the bounding box is large: " << m_bboxSide << std::endl;
    	std::cout << "the vector for the origin translation is: " << m_originTranslation.x << " " << m_originTranslation.y << " " <<  m_originTranslation.z << std::endl << std::endl;


#endif

    //Transfer on GPU constant memory subdivison parameters
    CopyOnGPU_SubParameter();

	return;
}



void Filter::ReorderCellsData_by_HashValue(){

	// wrap raw pointer with a device_ptr
	thrust::device_ptr<unsigned int> th_cellsHash = thrust::device_pointer_cast(m_cellsHash);
	thrust::device_ptr<unsigned int> th_cellsIndex = thrust::device_pointer_cast(m_cellsIndex);

	cudaEvent_t start, stop;  // create cuda event handles
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));

	float calcHashAndSort_time = 0.0f;

	//set the execution configuration
	uint nThreads, nBlocks;
	computeGridSize(m_NCells, 64, nBlocks, nThreads);

	dim3 numThreads(64);
	float PP = sqrt(nBlocks);
	dim3 numBlocks(PP+1 , PP+1);

	cudaEventRecord(start, 0);  //start record time

	//calculate grid hash value for each particle
	calcHashD<<<numBlocks,numThreads>>>(m_cellsHash, m_cellsIndex, m_VCells, m_NCells);
	//Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	/*
	printf("after  kernel ");
	for(int i=0 ; i<m_NCells; i++){
		std::cout << m_cellsHash[i] <<", " ;

	}
*/
	//!!!ATTENZIONE: sort_by_key restituisce l'errore: Errore di bus(core dump creato)
	//thrust::sort_by_key(th_cellsHash, th_cellsHash+m_NCells, th_cellsIndex);
	//Wait for GPU to finish before accessing on host
	//cudaDeviceSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	CUDA_CALL(cudaEventElapsedTime(&calcHashAndSort_time, start, stop));
	cudaDeviceSynchronize();

#ifdef DEBUG
	std::cout <<"Execution time for: hash table and particle index rearrangement" << calcHashAndSort_time << std::endl ;
#endif

}


void Filter::TestNewData(){
   std::cout <<"Sono in test newdata" << std::endl;
	for(int i=0 ; i<m_NCells; i++){
			std::cout << m_cellsHash[i] <<", " ;

		}

}
