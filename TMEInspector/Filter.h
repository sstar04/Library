/*
 * Filter.h
 *
 *  Created on: 20/lug/2018
 *      Author: sabry
 */
/*! < \class Filter:
 * This class is a bridge between the 3D point data set (cells position) and the class Graph. It performs mainly the space subdivision
 * algorithm and hash table that map each point (cell) in a subdivision small box.
 * A set of 4D points stored in Unified Memory to get best performance on CUDA code is it's main interface.
 */


#ifndef FILTER_H_
#define FILTER_H_

#include "DataStructure.h"


class Filter {

	friend class TMEInspector;

private:

	point4d* m_VCells;  //!< Unified memory vector storing position in the space and radius
    uint m_NCells;

    bbox m_bbox3D;     //! bounding box for the cell cluster (Thrust::pair object)

    //Subdivision parameters (SubParams)
    unsigned int m_s;          //! number of subdivision of the cubic bounding box's side;
    unsigned int m_NsmallBox;  //! total number of small cubic boxes dividing the cubic bounding box (!!vecchio nome della variabile NtotCell)
    float m_bboxSide;          //! Length of cubic bounding box side
    float m_sboxSide;          //! Length of small cubic boxes side (!! vecchio nome della variabile sizeCell)
    float3 m_originTranslation;

    //parameter used for hash table
    point4d* m_sortedVCells;   //! Unified memory vector: data on vector VCells are sorted by hash value;
    uint*    m_cellsHash;      //! Unified memory vector: each cell inheriths an hash value (unique key) that specifies the small box where the cell is located.
    uint*    m_cellsIndex;     //! Unified memory vector: each cell has an index which correponds to the position in the former vector m_VCells

    bbox Calculate_BoundingBox();
    //SimParams m_globalParams;   //
    void CopyOnGPU_SubParameter();

public:
    Filter();
	Filter(point4d* VCells, uint NCells);
	virtual ~Filter();

	// costruttore copia
	//Filter(const Filter& F);

	void Print_coordinates();

	void Calculate_ExtendedBoundingBox(float USide, int Nlayers); //! determine the space partitioning parameter from the extended bouding box.
    void ReorderCellsData_by_HashValue(); //! calculate Hash table and reorder the contents of the m_VCells vector.

    void TestNewData();
};

#endif /* FILTER_H_ */
