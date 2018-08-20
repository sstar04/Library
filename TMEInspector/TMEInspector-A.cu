/*
 * TMEInspector-A.cu
 *
 *  Created on: 17/aug/2018
 *      Author: Sabrina
 */

#include <math.h> 
#include <stdio.h>
#include <algorithm>    
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "TMEInspector.h"
//#include "TMEInspector_kernel.cu"

#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "macro.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/replace.h>
#include <thrust/distance.h>
#include <cmath>
#include "Utility.cuh"
#include "GridConstruction.cuh"
#include "CCL_Algorithm.cuh"
#include "StarAlg_kernel.cuh"
#include <thrust/copy.h>

using namespace std;

//Print execution time flag
bool printExecTime = true;

//debugging flags
bool debugAdCell = false;
bool scaleNumber = false;
bool checkBorder=false;



// external function implemented in "CCL_Algorithm.cuh"
extern void CCL_Algorithm_2D(
		thrust::device_vector<uint>  &node_state,              //input data
		thrust::device_vector<uint>& nodes_label,              //output data for visualize result
		//	vector<uint>& border_cell                          //output data for find the border cell.
		uint size);



void TMEInspector::STARAlgorithm(std::vector<bool>& IsonAS,
									float a )//! algorithm's runtime parameter (USide= a*r_max)
{

	  //Calculate Extended bounding box
      float U_side = (a*m_tgSim.Get_rmax());

      globalParams.cellSide = (U_side); //definita in GridConstruction.cuh

#ifdef DEBUG
	  std::cout << "**** TMEInspector::STARAlgorithm says:" << std::endl;
	  std::cout << "runtime parameter 'a' of STARAlgorithm is set to: " << a << std::endl;
	  std::cout << "the maximum value of the cell's radius is: " << m_tgSim.Get_rmax() << std::endl;
	  std::cout << "the side of the small cubic box is: " <<a*m_tgSim.Get_rmax() << "micron" << std::endl<< std::endl;
#endif

	//////////
    // Space subdivision
    //////////

    //1. Defining the bounding box
    // This step is implemented using thrust library using the code provided in thrust examples at
    // https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu

    cudaEvent_t start, stop;  // create cuda event handles
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    // wrap raw pointer with a device_ptr
	thrust::device_ptr<point4d> dev_VCells = thrust::device_pointer_cast(m_tgSim.Get_VCells());
    
    //initial bounding box contain first point
	bbox init = bbox(dev_VCells[0],dev_VCells[0]);

    //transform operations
    bbox_transform unary_op;

    //binary reduction
    bbox_reduction binary_op;

    float bbox_time = 0.0f;
    
    uint numCells = m_tgSim.Get_numCells();
    
    //start record time
    cudaEventRecord(start, 0);
    bbox result = thrust::transform_reduce(dev_VCells, dev_VCells+numCells, unary_op, init, binary_op);
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

 
    //2. Define the extended bounding box

    bbox box3D;
    box3D = Bbox_EnlargeDivide(result);  //set two layes of extra space between cluser and cubic boundary

     uint numSbox = globalParams.NtotCell;

#ifdef DEBUG
    	cout << "Extended bounding box" << fixed;
    	cout << "(" << box3D.first.x << "," << box3D.first.y << "," << box3D.first.z << ")" ;
    	cout << "(" <<  box3D.second.x << "," << box3D.second.y << "," << box3D.second.z  <<")" <<endl;

    	cout << "the side of the bounding box is large: " << globalParams.bboxSide << endl;
    	cout << "the vector for the origin translation is: " << globalParams.originTranslation.x << " " << globalParams.originTranslation.y << " " <<  globalParams.originTranslation.z << endl;
    
        cout << "Total number of space partitioning cubes: " << numSbox << endl; // print total number of small cubic boxes

#endif 

    //inizialize a variable on the GPU's constant memory
    cudaMemcpyToSymbol(params, &globalParams, sizeof(SimParams));

#ifdef DEBUG
      cout << "Transfer Subdivision parameter on GPU constant memory:" << endl;
#endif

    /////////
    //Setting a hash table and memory rearrangement of the 4D data points.
    ////////
    //////// In this step the kernels implementation are copied by CUDA samples, available at CUDA toolkit package.
    
    thrust::device_vector<unsigned int> gridParticleHash(numCells);
    thrust::device_vector<unsigned int> gridParticleIndex(numCells);
    
    //wrap thrust pointer
    point4d * pos = thrust::raw_pointer_cast(&dev_VCells[0]);
    unsigned int * id_cell = thrust::raw_pointer_cast(&gridParticleHash[0]);
    unsigned int * id_particle = thrust::raw_pointer_cast(&gridParticleIndex[0]);
    
   
    float calcHashAndSort_time = 0.0f;
    
    //set the execution configuration
    uint nThreads, nBlocks;
    computeGridSize(numCells, 64, nBlocks, nThreads);

    dim3 numThreads(64);
    float PP = sqrt(nBlocks);
    dim3 numBlocks(PP+1 , PP+1);

    cudaEventRecord(start, 0);  //start record time

    //calculate grid hash value for each particle
    calcHashD<<<numBlocks,numThreads>>>(id_cell, id_particle, pos, numCells);

    thrust::sort_by_key(gridParticleHash.begin(), gridParticleHash.end(), gridParticleIndex.begin());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&calcHashAndSort_time, start, stop));
    cudaDeviceSynchronize();

#ifdef DEBUG
    cout <<"Execution time for: hash table and particle index rearrangement " << calcHashAndSort_time << endl ;
#endif

    thrust::device_vector<uint> cellStart(numSbox);
    thrust::device_vector<uint> cellStop(numSbox);
    thrust::device_vector<point4d> sortedPoints(numCells);
    thrust::device_vector<float> sortedRadius(numCells);

    uint * cStart = thrust::raw_pointer_cast(&cellStart[0]);
    uint * cStop  = thrust::raw_pointer_cast(&cellStop[0]);
    point4d * sPoints  = thrust::raw_pointer_cast(&sortedPoints[0]);
    float   * sRadius   = thrust::raw_pointer_cast(&sortedRadius[0]);

    float ReordingMemory_time = 0.0f;

    cudaEventRecord(start, 0);    //start record time

    //rearrangement of 4D points vector according to the new particle index
    reorderDataAndFindCellStartD<<<numBlocks,numThreads>>>(cStart, cStop, sPoints, id_cell, id_particle, pos, numCells);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&ReordingMemory_time, start, stop));

#ifdef DEBUG
   std::cout <<"Execution time for: point data memory rearrangement" << ReordingMemory_time << std::endl ;
#endif
   // transferPointCoord<<<numBlocks,numThreads>>>(sPoints, sCoord, sRadius, numCells);

    ///////////
    //calculate cellsize (: the total number of particle for a single small cube)
    ///////////
    thrust::device_vector<uint> cellSize(numSbox);
    uint * cSize  = thrust::raw_pointer_cast(&cellSize[0]);
    thrust::transform(cellStop.begin(), cellStop.end(), cellStart.begin(), cellSize.begin(),
    		thrust::minus<uint>());
#ifdef DEBUG
    	std::cout <<  "  ( Start index, end index ) " << std::endl;
    	for (int i=0 ; i<numSbox ; i++){
    		std::cout << "hash number " << i << " : (" <<  cellStart[i] <<  "," << cellStop[i] << ")" << " " << cellSize[i] << std::endl;
    	}
#endif



    ///////////
    // Boundary search
    ///////////

    thrust::device_vector<bool> borderCell(numCells);
    bool * bCell  = thrust::raw_pointer_cast(&borderCell[0]);

    thrust::device_vector<uint> node_state(numSbox);
    thrust::device_vector<uint> node_label_dev(numSbox);
    uint * n_state  = thrust::raw_pointer_cast(&node_state[0]);
    uint * n_label_dev  = thrust::raw_pointer_cast(&node_label_dev[0]);

    //initialize node property vector
    set_number setProperty;
    thrust::transform_if(thrust::cuda::par, cellSize.begin(), cellSize.end(),node_state.begin(), setProperty , is_positive());
       
    //connected component labeling algorithm
    CCL_Algorithm_2D(node_state, node_label_dev, globalParams.s);



    /////////
    //Detect subgraphs pointing to the environment
    ////////

   /* nodes_label.resize(numSbox); //host vector to store the output results

    for (int i =0 ; i< numSbox ; i++){
    	nodes_label[i] = node_label_dev[i];
    }*/

    //set the execution configuration
    //
    uint nThreads_new, nBlocks_new;
    computeGridSize(numSbox, 64, nBlocks_new, nThreads_new);

    dim3 numThreads_new(8,8);  //dimensione del singolo blocco
    float PP2 = sqrt(nBlocks_new);
    dim3 numBlocks_new(PP2+1 , PP2+1);


    // 1. set the list of ID-subgraphs having nodes at the bounding box
    thrust::device_vector<uint> list_border_label(numSbox);
    uint* list_bl = thrust::raw_pointer_cast(&list_border_label[0]);

    //IntIterator newEnd0 = thrust::remove(thrust::device ,list_border_label.begin() , list_border_label.end(),-1);
    float findEnvSub_time = 0.0;
    cudaEventRecord(start, 0);

    // return the list of label subgraph that belong to the environment
    ListBorderLabelSubgraph<<<numBlocks_new, numThreads_new>>> (n_state, n_label_dev, globalParams.s, list_bl);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findEnvSub_time, start, stop);

#ifdef DEBUG
    	std::cout <<" **************" << std::endl ;
    	std::cout <<"Execution time for the environment subgraph search: " << findEnvSub_time << std::endl ;
    	std::cout <<" **************" << std::endl ;

 //   ofs <<"Execution time for the environment subgraph search: " << findEnvSub_time << std::endl ;

   /*
    	cout << " hash value, list_label , list_border_label, node_state" << endl;
    	for(int i = 0 ; i<numSbox; i++ ){
    		cout << i << " " << nodes_label[i] << list_border_label[i] << " "  << node_state[i] <<  endl;
    	}*/
#endif

    //Eliminate -1 value from the list_border_hash vector.
    int nullvalue=numSbox+1;
    IntIterator newEnd = thrust::remove(thrust::device ,list_border_label.begin() , list_border_label.end(),nullvalue);

#ifdef DEBUG
    	cout << "list of subgraph indexes" << endl;
    	for(IntIterator i = list_border_label.begin(); i < newEnd; i++){
    		cout << *i << ", ";}
    	cout << endl;
#endif

    //delete recurrent value from the list
    IntIterator newEnd_label = thrust::unique(thrust::device, list_border_label.begin(), list_border_label.end());


    //2: set node_state = 2 to all nodes of the previously selected subgraph
    // STEP2: propago lo stato environment a tutti i nodi  dei sottografi di list_border_label  //
        int Nlabels = thrust::distance(list_border_label.begin(),newEnd_label);



    float setStatus2_time = 0.0;
    cudaEventRecord(start, 0);

    SetEnvironmentState<<<numBlocks_new, numThreads_new>>>(list_bl, Nlabels, n_label_dev, globalParams.s ,n_state);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setStatus2_time, start, stop);

    if(printExecTime){
    	cout <<" **************" << endl;
    	cout <<"Execution time for broadcasting the value 2: " << setStatus2_time << std::endl ;
    	cout <<" **************" << endl;
    }

  //  ofs <<" Execution time for broadcasting the value 2 at the environment subgraphs: " << setStatus2_time << std::endl ;


    ///3: Detect border nodes
    // Search for "full" nodes  having at least a single neighbouring "environment" node
    thrust::device_vector<int> list_border_nodes(numSbox);
    int* list_bn = thrust::raw_pointer_cast(&list_border_nodes[0]);

    thrust::device_vector<uint> hash_value(numSbox);
    uint * h_value  = thrust::raw_pointer_cast(&hash_value[0]);

    //fill the vector of hash table.
    thrust::sequence(thrust::cuda::par,hash_value.begin(), hash_value.end());

    float findBorderNodes_time = 0.0;
    cudaEventRecord(start, 0);
    Detect_BorderNodes<<<numBlocks_new, numThreads_new>>> (n_state, h_value,  numSbox, list_bn);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findBorderNodes_time, start, stop);
    if(printExecTime){
    	std::cout <<" **************" << std::endl ;
    	std::cout <<" Execution time for border node detection : " << findBorderNodes_time << std::endl ;
    	std::cout <<" **************" << std::endl ;
    }
    //ofs <<" Execution time for border node detection: " << findBorderNodes_time << std::endl ;




    ////////////
    //LIST of point data at the boundary of the cluster of cells
    ///////////
    //Select those 4D data points included in the border nodes
    ///////////
    thrust::fill(thrust::cuda::par,borderCell.begin(), borderCell.end(), false);
    float findBorderCells_time = 0.0;
    cudaEventRecord(start, 0);

    Find_ListBorderCells<<<numBlocks_new, numThreads_new>>>(sPoints,cStart,cStop,list_bn, bCell, numSbox);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findBorderCells_time, start, stop);

    if(printExecTime){
    std::cout <<" **************" << std::endl ;
    std::cout <<" Execution time to determine the list of point data at the boundary of the cell cluster: " << findBorderCells_time << std::endl ;
    std::cout <<" **************" << std::endl ;
    }

  //  ofs <<" Execution time to determine the list of point data at the boundary of the cell cluster : " << findBorderCells_time << std::endl ;
	if(checkBorder){
    	for(int i = 0 ; i<borderCell.size(); i++ ){
    		cout << i << " = " << borderCell[i] << endl;
    	}
   }

    /////////////////////
    // 3.transfering  data on CPU
    //////////////////

    thrust::host_vector<int> H_borderCell(numCells);
    H_borderCell = borderCell;

    int temp;
    //rimepio il vettore esterno senza variare gli indici delle cellule.
    for(int i = 0; i <numCells; i++)
        {
            temp = gridParticleIndex[i]; //ritorna la posizione della cellula nel vettore non ordinato
            IsonAS[temp] = H_borderCell[i];
        }

     return;

}