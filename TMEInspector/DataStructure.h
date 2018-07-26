/*
 * DataStructure.h
 *
 *  Created on: Nov 28, 2013
 *      Author: sabry
 */

#ifndef DATASTRUCTURE_H_
#define DATASTRUCTURE_H_

#include <thrust/pair.h>
#include <stdio.h>


const int NLABEL = 100;

typedef unsigned int uint;

// struct that stored parameter related to the space partition operations

struct SubParams   //old name SimParameters
{
	unsigned int s;           //! number of subdivision of the cubic bounding box's side;
	unsigned int NsmallBox;   //! total number of small cubic boxes dividing the cubic bounding box (!!vecchio nome della variabile NtotCell)
	float bboxSide;           //! Length of cubic bounding box side
	float sboxSide;           //! Length of small cubic boxes side   (!! vecchio nome della variabile sizeCell)
	float3 originTranslation;

/*
	//default constructor
	__host__ __device__
	SimParams(): s(0), NsmallBox(0), bboxSide(0), smallBoxSide(0), originTranslation() {}

	__host__ __device__
	SimParams(uint a, uint b, float c, float d, float3 e): s(a), NsmallBox(b), bboxSide(c), smallBoxSide(d), originTranslation(e) {}

	__host__ __device__
	inline SimParams operator=(const SimParams& b)
	{
		return SimParams(b.s, b.NsmallBox, b.bboxSide, b.smallBoxSide, b.originTranslation);
	}
*/
	__host__ __device__
	inline void PrintValues()
	{
		printf("s: %d, NsmallBox: %d, bboxSide: %f , cellSide: %f \n", s, NsmallBox, bboxSide, sboxSide);
		printf("origin translation. x: %f, y: %f, z: %f \n", originTranslation.x, originTranslation.y, originTranslation.z);

		return;
	}

};


struct point3d
{
   float x, y, z;

   //default constructor
   __host__ __device__
   point3d(): x(0),y(0), z(0){}

   __host__ __device__
   point3d(float _x, float _y, float _z): x(_x), y(_y), z(_z){}

   __host__ __device__
   inline point3d operator-(const point3d& b)
   {

     return point3d((x-b.x), (y-b.y), (z-b.z));
   }

    __host__ __device__
   inline point3d operator+(const point3d& b)
   {

     return point3d((x+b.x), (y+b.y), (z+b.z));
   }

};

struct point4d
{
   float x, y, z, r;

   //default constructor
   __host__ __device__
    point4d(): x(0),y(0), z(0), r(0){}

   __host__ __device__
   point4d(float _x, float _y, float _z, float _r): x(_x), y(_y), z(_z), r(_r){}

   //these operator works only for the spatial coordinates.
   __host__ __device__
   inline point4d operator-(const point4d& b)
   {

     return point4d((x-b.x), (y-b.y), (z-b.z), 0);
   }

    __host__ __device__
   inline point4d operator+(const point4d& b)
   {

     return point4d((x+b.x), (y+b.y), (z+b.z),0);
   }

   __host__ __device__
   inline void printCoord()
{
      printf("x: %f, y: %f, z: %f \n", x, y, z); 

}


};

/*
__host__ __device__
 inline int3 operator+(const int3& a,const int3& b)
 {
     int x = a.x+b.x;
     int y = a.y+b.y;
     int z = a.z+b.z;

     int3 c = make_int3(x,y,z);

   return c;
 }
 */
struct is_zero
  {
    __host__ __device__
    bool operator()(int x)
    {
      return x == 0;
    }
  };


//bounding box type
typedef thrust::pair<point4d,point4d> bbox;

/*
SimParams globalParams;

__constant__ SimParams params;

__constant__ uint BorderLabel[NLABEL];



*/

/*  static __inline__ __host__ __device__ unsigned int min(unsigned int a, unsigned int b)
   {
     return umin(a, b);
   }*/

#endif /* DATASTRUCTURE_H_ */
