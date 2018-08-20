/*
 * TMEInspector.h
 *
 *  Created on: 25/mag/2018
 *      Author: Sabrina
 */
/*! < \class TMEInspector: This class knows the spatial information of the cells cluster and the cell phenotype or any other cell's mark. It is resposible for data transfer to GPU
 */


//classe esempio che prende un vettore esterno e lo trasferice nella GPU 

#ifndef TMEINSPECTOR_H_
#define TMEINSPECTOR_H_

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <vector>
#include "TGSimulator.h"


class TMEInspector {

protected:

    TGSimulator m_tgSim; 
    bool* m_border;     //vettore lista delle cellule di bordo
	
 
public:
	TMEInspector();
	TMEInspector(TGSimulator& tgSim);
	
        //Costruttore che usa la Unified Memory   
        TMEInspector(unsigned int npart);

	virtual ~TMEInspector();

	//void Set_numCells(unsigned int);
       
    bool* Get_BorderCells();
    bool* Get_BorderCellsTh();
    void Get_VecBorderCells(std::vector<bool>& IsonAS); //!< Copy in a std::vector the result of the STARAlgorithm: search of the cells at the border  

 /*!< Exec the algorithm for detecting the cells at the border of the spheroid print the result, in terms of yes or no, in a std::vector<bool>*/
    void STARAlgorithm(std::vector<bool>& IsonAS, float a);

};

#endif /* TMEINSPECTOR_H_ */
