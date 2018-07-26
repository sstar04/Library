/*
 *
 *  Created on: 23/may/2018
 *      Author: Sabrina Stella
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "TMEInspector/TGSimulator.h"
#include "TMEInspector/TMEInspector.h"


using namespace std;

 bool scaleNumber = false;
////////
/// This program reads - form an external file - the point set NPART and give it to TGSimulator
/// NCELLS = total number of 4D points: (3D coordinates,radius):= (x,y,z,r)
/// NBOXES = total number of partition cubes

int main ()
{

    //////
    //read 4D points from a vtk file
    //////

    char fileName[30];
    int num;
    cout<<"insert file number ";
    if(scaleNumber)cout <<"(position expressed in meter)";
    cin>>num;
    sprintf (fileName, "DataSetExamples/Cells%d.vtk", num);
    cout << fileName << endl;

    
    TGSimulator vbl(fileName, scaleNumber);
    vbl.Set_rmax(8); //maximum vale of the cell's radius simulation parameter
    
    unsigned int NCELLS = vbl.Get_numCells(); 
    cout << "il numero di cellule e':" << NCELLS <<endl;

    //vbl.PrintCellsPosition();
      
     //I need TME information from my dataset provided by TGSimulator object
     TMEInspector current_TMEInspector(vbl);
     

     // get border cells     
     bool* vbl_border = current_TMEInspector.Search_BorderCells();
    
     std::cout << std::endl << " Nuovo Border Cell list :" ;
     for(int i = 0; i < NCELLS; i++) {
        std::cout << vbl_border[i] << " ";     
        }

    /*
    //trasferico il contenuto nel std::vector IsonAS con il contenuto della Unified memory di TMEInspector 
    std::vector<bool> IsonAS(NCELLS);
    current_TMEInspector.Search_VecBorderCells(IsonAS);*/

    std::vector<bool> IsonAS(NCELLS);

    //Runtime parameter for STARAlgorithm
    float a;                //default value 2;
   // float r_max = 8;          //maximum value for the cell radius
    std::cout << "Set the value for the space partitioning parameter a (a*r_max)(defult: 2)";
    std::cin >> a;

     current_TMEInspector.STARAlgorithm(IsonAS, a);
    

    std::cout << std::endl << " Print di IsonAS :" ;
     for(int i = 0; i < NCELLS; i++) {
        std::cout << IsonAS[i] << " ";     
        }
    
 
   
 return 0; 
}
