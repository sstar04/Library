/*
 * TGSimulator.h
 *
 *  Created on: 22/mag/2018
 *      Author: Sabrina
 */
/*! < \class TGSimulator: This class knows the spatial information of the cells cluster and the cell phenotype or any other cell's mark. It is resposible for data transfer to GPU
 */


//classe esempio che prende un vettore esterno e lo trasferice nella GPU 

#ifndef TGSIMULATOR_H_
#define TGSIMULATOR_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include "DataStructure.h"

const unsigned int M = 10000;


class TGSimulator {

protected:

	unsigned int m_numCells;    //!< total number of cells in the set

        point4d* m_VCells;  //!< Unified memory vector storing position in the space and radius
        int* m_VType;       //!< Unified memory vector storing cell phenotype

	//Useful CellType parameter
		float m_rmax;      //!< Maximum value of the cell's radius
	
public:
	TGSimulator();    //!< default constructor
        TGSimulator(unsigned int npart);
        
        TGSimulator( char * fileName, bool scaleNumber); //!< vtk test constructor

        //!< vbl constructor
        TGSimulator(unsigned int npart, std::vector<double>& x, std::vector<double>& y ,std::vector<double>& z ,std::vector<double>& r ,std::vector<int>& t);

	virtual ~TGSimulator();


	unsigned int Get_numCells(){return m_numCells;}
        point4d* Get_VCells(){ return m_VCells;};
        float    Get_rmax(){return m_rmax;}

	void Set_rmax(float rmax) {m_rmax = rmax; return;}
        
        void PrintCellsPosition();
       

};

#endif /* TGSIMULATOR_H_ */
