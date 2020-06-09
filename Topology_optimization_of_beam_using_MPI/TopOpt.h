#ifndef TopOpt
#define TopOpt
#include <petsctime.h>
#include <petsc.h>
#include "petscdmplex.h" 
#include <petscsys.h>
#include <petscdmda.h> 
#include <petscdm.h>


/* Function Declaration */


void GridSetUp();
static PetscErrorCode DMDAGetElementEqnums_u(PetscInt i,PetscInt j,PetscInt k);
static PetscErrorCode ElementStiffnessMatrix(PetscScalar X[],PetscScalar Y[],PetscScalar Z[],PetscScalar nu,PetscScalar *ke);
static PetscScalar Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l);
static PetscScalar  Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]);
static PetscErrorCode ForceVectorSetUp(DM da);
static PetscErrorCode AssembleStiffnessMatrix(DM da);
static PetscErrorCode SolveState(DM da,Mat K);
static PetscErrorCode Create3DVTK(DM da);
static PetscErrorCode ElementStiffnessMatrixCalc(DM da);
static PetscErrorCode SetupH_and_Hs(DM da);
static PetscErrorCode TopoOpt(DM da);
/* typedef struct
	 {
		 PetscScalar GID[3],coord[3];
	 } MeshInfo;  */
#endif