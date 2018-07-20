/*
 * TGSimulator.cpp
 *
 *  Created on: 22/mag/2018
 *      Author: Sabrina
 */

#include "TGSimulator.h"
#include <thrust/system/cuda/execution_policy.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

TGSimulator::TGSimulator(){
	// TODO Auto-generated constructor stub

}


TGSimulator::TGSimulator(unsigned int npart): m_numCells(npart) {

   // Allocate Unified Memory -- accessible from CPU or GPU
      cudaMallocManaged(&m_VCells, npart*sizeof(point4d));
      cudaMallocManaged(&m_VType, npart*sizeof(int));
      cudaDeviceSynchronize();
     
}


TGSimulator::TGSimulator(char * fileName, bool scaleNumber){
 

   bool debugAdCell = false;
   bool checkBorder=false;
   bool checkGrid=false;

   //////////////
   //initialize data from vtk file
   //////////////
    std::ifstream fileCoord, fileData;
    fileCoord.open(fileName);
    std::string line;
    std::string a, b, c;
    for(int i=0; i<4 ; i++)getline(fileCoord,line);
    fileCoord >> a >> b >> c;
    
    //get total number of cells
    m_numCells = std::atoi(b.c_str());
   
    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&m_VCells, m_numCells*sizeof(point4d));
    cudaMallocManaged(&m_VType, m_numCells*sizeof(int));
    cudaDeviceSynchronize();

    
    //ponter on radius value
    fileData.open(fileName);
    std::string line2, r;
    std::string word("radius");
    std::size_t found;

    do{
    	getline(fileData,line2);
    	found = line2.find(word);
    }while(found==std::string::npos);
    getline(fileData,line2);
    fileData >> r;
  
    //initialize vector with cells'data
    fileCoord >> a >> b >> c;
    int i = 0;
    while(a !="POINT_DATA"){
    	 if(scaleNumber){  m_VCells[i]=point4d(atof(a.c_str())*pow(10,-6),atof(b.c_str())*pow(10,-6),atof(c.c_str())*pow(10,-6),atof(r.c_str())*pow(10,-6));}
    	 else m_VCells[i]=point4d(atof(a.c_str()),atof(b.c_str()),atof(c.c_str()),atof(r.c_str()));
    	 if (checkGrid) std::cout << "( " << atof(a.c_str()) << ", " << atof(b.c_str()) <<  ", " << atof(c.c_str()) << ", " << atof(r.c_str()) << ")" << std::endl;
    	 fileCoord >> a >> b >> c;
    	 fileData >> r;
    	 i++;
    }
    //initialize type cells vector with a default value
     for(int i = 0; i < m_numCells; i++){

      m_VType[i]  = 0;
    }

}

TGSimulator:: TGSimulator(unsigned int npart, std::vector<double>& x, std::vector<double>& y ,std::vector<double>& z ,std::vector<double>& r ,std::vector<int>& t):
 m_numCells(npart) {

   // Allocate Unified Memory -- accessible from CPU or GPU
      cudaMallocManaged(&m_VCells, npart*sizeof(point4d));
      cudaMallocManaged(&m_VType, npart*sizeof(int));
      cudaDeviceSynchronize();

   // Initialize data      
   for(int i = 0; i < npart; i++){

      m_VCells[i] = point4d(x[i],y[i],z[i],r[i]);
      m_VType[i]  = t[i];
   }
     
}


TGSimulator::~TGSimulator() {
	 // Free memory
      cudaFree(m_VCells);
      cudaFree(m_VType);
}

/*void TGSimulator::Set_Npart(unsigned int npart)
{
	m_numCells = npart;
}*/

/*
thrust::host_vector<point4d>& TGSimulator::Get_dTohCells(){

        m_hCells = m_dCells;  //(il trasferimento dati DtoH richiede un tempo spropositato)
	return m_hCells;

}

void TGSimulator::Get_dCells(std::vector<point4d>& hCells){

       m_hCells = m_dCells;  //! attenzione fare solo se necessario.

         for(int i = 0; i < m_numCells; i++)
        {
            hCells[i] = m_hCells[i];
        }

       //non funziona!!
       //thrust::copy(m_hCells.begin(), m_hCells.end(), hCells.begin());  
        return ;

}
*/

void TGSimulator::PrintCellsPosition(){

      for(int i = 0; i < m_numCells; i++) {
        std::cout << "Cell" << i <<" " ;
        m_VCells[i].printCoord();     
        }

}


