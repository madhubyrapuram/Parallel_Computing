static char help[] = " Topo Optimization using PETSc\n\n";
#include <stdio.h>
#include <petsc.h>
#include "TopOpt.h"
#include "petscdmda.h" 
#include <petscdm.h>
#include "petscdmda.h" 
#include "mpi.h"
#include "petscviewer.h"

 /*

Parallel 3D topology Optimization using PETSc
ISP Project By Madhusudhan Reddy Byrapuram,CMS Student

	
Note 1 :
This Code by Default solves :Cantilever Beam Topology Optimization with Line Load Along Edge

Note 2: 
Two other cases have also been presented in the code(discussed below),which have been commented out.:
1.Cantilever Beam Topology Optimization with Single Upper Load - (Fig 11 From Results Report) 
2.Cantilever Beam Topology Optimization with Single Load at Lower Edge-(Fig 8 From Results Report)

User can uncomment the relevant case section in the ForceVectorSetUp() method in TopOpt.c.Only one case method can be executed at a time.

*/
int main(int argc, char **argv)
{  

  PetscInitialize(&argc,&argv,(char*)0,help);
  PetscErrorCode ierr;
  GridSetUp();
 
  ierr = PetscFinalize(); 
 
return 0;
}