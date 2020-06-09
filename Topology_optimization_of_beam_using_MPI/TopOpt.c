#include "TopOpt.h"

//using namespace std;




/* Variable Defination-
The following variables value can be changed for different optimized results:

 I. Element in x direction -nelx,Element in y direction -nely,Element in z direction -nelz,
	PetscInt  nel,nn,nelx=10,nely=40,nelz=10 
	
 II. Volume Fraction
 PetscScalar volfrac=0.15; 
 
 III. Radius Filter
 PetscScalar rmin=1.5;
 
 IV. Penalization factor p
 PetscScalar penal=3.0;  

*/

Vec U,F,BC; 
PetscErrorCode ierr;
 Vec              local,global;
 Vec  coords;
 DM               da,da_element;
  //MPI_Comm       comm= PETSC_COMM_WORLD;
 DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE,bz=DM_BOUNDARY_NONE;
 DMDAStencilType  stype = DMDA_STENCIL_BOX;
 PetscInt  nel,nn,nelx=100,nely=20,nelz=10,elements_total;
 PetscScalar E0 = 1;          // Young's modulus of solid material
 PetscScalar  Emin = 1e-9;      // Young's modulus of void-like material
 PetscScalar  nu = 0.4;         // Poisson's ratio
 PetscScalar volfrac=0.15; // Vol Fraction
 PetscScalar rmin=1.5;  // Min Radius Ignorance
 PetscScalar penal=3.0;  // Penalty
 Vec XDes; // Design Variable-X
 Mat KGlobal;
 const PetscInt   *elements;// to store local  nodes of Elements
 PetscInt *necon,*ele;  
 const  PetscInt    *gidx;   // Get Global Index Values
 ISLocalToGlobalMapping ltogm; // localtoGlobalMapping Index values
 ISLocalToGlobalMapping ele_ltogm; // localtoGlobalMapping Index values for local indices
 MatStencil             u_eqn[24]; /* 2 degrees of freedom Value of 24 = ndof(3)*nodes(8) */
 Mat GID;   // Storing Global Indices of Nodes
 PetscScalar    ke[24*24]; // element matrix
 PetscScalar X[8],Y[8],Z[8];  // to store element Co-ordinates
 Mat ZGlobal;  // Matrix which stores coordinates corresponding to Node Position
 PetscInt NON;  // Number of Nodes
 Vec ug;  // Vetcor to view coordinate of grids
 PetscInt x_min,x_max,y_min,y_max,z_min,z_max;  // Get Limits of Grid Points
 PetscInt m,i,j,k,x,n;
 int rank ;
//MeshInfo *element; // Create Element Pointer which will access values of Struct MeshInfo
DMDALocalInfo info;
DMDACoor3d  ***Mesh_Coord;
PetscViewer       view,view1,view2;
PetscInt l,dof=3;
const char filename[] ="TopOpt_Visualization.vtr"; // Paraview FileName
//const char file_name[]="ABHI.txt";
const char filename2[]="TopOpt_ProcessInformation.txt"; // Process FileName 
  
PetscScalar KF[24*24];
PetscScalar C;     // Objective Function
Vec dC;  // sensitivity objective function
Vec dV;  //volume sensitivity function
Vec dV_invlmid; // volume sensitivity/invlmid
 //PetscScalar dV; // Volume
KSP ksp1;  // KSP variable
Mat H;  // H matrix
Vec Hs;   // HS Vector
PetscInt count=0;
PetscScalar Density;
// Get pointer to the densities
PetscScalar *xp;
Vec NI;
// Form the Element_DOF-24 entries
PetscInt       el_dof[24]; 
PetscInt daele_nel,daele_nn,rank_size;
const PetscInt   *daele_elements;
Vec XNew;
PC pc;
PetscLogDouble t1,t2,elapsed;
//PetscGetTime(&t1)

/*
GridSetUp()-  Main Process Function which Invokes other functions and also sets up the Mesh Grid
*/



void GridSetUp()
{ 

// Start Time 
t1=MPI_Wtime();


//PetscMalloc1((nelx*nely*nelz),&element);

 // Create 3-D Grid Setup
 // Number of grid points = (number of elements+1) (in each direction) ---- n_gridPts_xDriection=n_elements_xDirection +1
 DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,(nelx+1),(nely+1),(nelz+1),PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,3,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);
 DMSetFromOptions(da);
 DMSetUp(da);

 // Only for elements
 DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,stype,(nelx),(nely),(nelz),PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da_element);
 DMSetFromOptions(da_element);
 DMSetUp(da_element);

// Set Element Design Variable Density
DMCreateGlobalVector(da_element,&XDes); 
VecSet(XDes,volfrac);  // Set to Vol.Fraction Initially

// Define 8-noded Hexahedron
DMDASetElementType(da,DMDA_ELEMENT_Q1);
DMDAGetElements(da,&nel,&nn,&elements);  // nn -number of nodes, nel-number of elements

// Find Processor size and Rank
MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
MPI_Comm_size(PETSC_COMM_WORLD, &rank_size);
 
// Get Global Index values
DMGetLocalToGlobalMapping(da,&ltogm);


 //PetscSynchronizedPrintf(PETSC_COMM_WORLD,"rank : %D number of elements: %D\n",rank,nel);
 elements_total=nel;
/*  for ( j=0; j<nel; j++) 
	{ 	ierr   = PetscSynchronizedPrintf(PETSC_COMM_WORLD," Element :%D [",j);
        for ( i=0; i<nn; i++) 
		{ n = elements[j* nn+i];
         ierr       = PetscSynchronizedPrintf(PETSC_COMM_WORLD," %D ",n);
		 }
		 ierr       = PetscSynchronizedPrintf(PETSC_COMM_WORLD," ] \n");		 
	     } */


// Start Writing Process Information into Text file-"TopOpt_ProcessInformation.txt"

PetscViewerCreate(PETSC_COMM_WORLD,&view2);
PetscViewerSetType(view2,PETSCVIEWERASCII); 
PetscViewerASCIIOpen(PETSC_COMM_WORLD,filename2,&view2);
PetscViewerASCIIPrintf(view2," \n  Elements  in  X-Direction : %D",nelx);
PetscViewerASCIIPrintf(view2," \n  Elements  in Y-Direction   :   %D",nely);
PetscViewerASCIIPrintf(view2," \n  Elements   in Z-Direction : %D",nelz);
PetscViewerASCIIPrintf(view2," \n  Total Elements  : %D ",(nelx*nely*nelz));
PetscViewerASCIIPrintf(view2," \n\n  Total Number of Processors  : %D \n",rank_size);


// ### Inititalize KGlobal -Global Stiffness Matrix and ksp1 - SOLVER ##############################
DMCreateMatrix(da,&KGlobal);
KSPCreate(PETSC_COMM_WORLD,&ksp1);
KSPSetFromOptions(ksp1);

/*
######################################################################################
MAIN PROCESS OF EXEUCTION BELOW:

1. ForceVectorSetUp - Set up the Force Vector and apply Boundary Conditions
2. ElementStiffnessMatrixCalc- Calculate the Element Stiffness Matrix for all elements. The values are almost identical.Just 
	time calculation.Kept in a seprate body so as to avoid repeated calculations during optimization loop.
3.AssembleStiffnessMatrix- Calculate K Global Matrix
4.SolveState - Find Displacement U.
5.SetupH_and_Hs- Set up the minimum distance function but for the DMDA-da_element (DOFs-1)
6.TopoOpt(da)- Run the Topology Optimization process and give the final Design Variable (XDes) values for all Elements
7. Create3DVTK- Create Visualization File
*/

// ################### FORCE VECTOR ###########################
ForceVectorSetUp(da);

// ################### ELEMENT STIFFNESS MATRIX ########################
ElementStiffnessMatrixCalc(da);

// ################### ASSEMBLE STIFFNESS MATRIX ########################
AssembleStiffnessMatrix(da);

// ################### CaLCULATE DISPLACEMENT VECTOR ######################
SolveState(da,KGlobal); 

// ################### CaLCULATE MIINIMUM DISTANCE FUNCTION#####################
SetupH_and_Hs(da_element);  

// ###################  TOPO OPTIMIZATION PROCESS#####################
TopoOpt(da);

// ###################  CREATE VISUALIZATION FILE FOR PARAVIEW #####################
Create3DVTK(da); 


PetscViewerASCIISynchronizedPrintf(view2,"\n FINAL SOLUTION- X Design Variables");
VecView(XNew,view2);
// End Time 
t2=MPI_Wtime();
PetscViewerASCIIPrintf(view2," \n Elapsed time : %g",t2-t1);
 
 
 // Destory All Utilized PETSc Objects
 
VecDestroy(&U);
MatDestroy(&KGlobal);
VecDestroy(&BC);
KSPDestroy(&ksp1);
VecDestroy(&XNew);
VecDestroy(&XDes);
DMDestroy(&da_element);
DMDestroy(&da);
PetscViewerFlush(view2);
PetscViewerDestroy(&view2);
PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT); 
}



/*
static PetscErrorCode TopoOpt(DM da) - This is the topology Optimiuzation function which computes the X Design
Variable for all elements. The value will range between 0 & 1
*/


static PetscErrorCode TopoOpt(DM da)

{



PetscScalar tolx =0.01;
PetscInt maxloop=200,loop=0;
PetscReal maxval;
PetscScalar change;
Vec Disp_Local;
Vec xtmp;
PetscScalar *dCdx;
PetscScalar *Uloc;
PetscScalar *x_physical,*x_pa,*x_new;
PetscScalar uKu=0.0;
Vec dummy_zero;
Vec x_temp;
Vec temp;
Vec X_pa;
Vec X_Phy;
Vec X;
Vec Change;
VecDuplicate(XDes,&dC);
VecDuplicate(XDes,&dV);
VecDuplicate(XDes,&xtmp);
VecDuplicate(XDes,&temp);
VecDuplicate(XDes,&x_temp);
VecDuplicate(XDes,&dummy_zero);
VecDuplicate(XDes,&X_pa);
VecDuplicate(XDes,&XNew);
VecDuplicate(XDes,&X_Phy);
VecDuplicate(XDes,&X);
VecDuplicate(XDes,&Change);
VecSet(X,volfrac);
VecSet(dC,0);
VecGetArray(XDes,&x_physical);
PetscScalar ch = 1.0,l1=0,l2=1e9,move=0.2,lmid,invlmid,x_ma,x_mb,sum;
DMCreateLocalVector(da,&Disp_Local);
PetscInt it=0,vec_size;
PetscScalar tmp;
PetscInt nloc,fscale;


// Start Loop from here ########################
loop = 0; 
change = 1.0;
while(change > tolx && loop < maxloop)

{
loop=loop+1;
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Change Number  :%g\n",change);

// ################ Calculate Stiffness Matrix ################

AssembleStiffnessMatrix(da);

//  ############### Solve System to Get New Values of : U ############
SolveState(da,KGlobal);

// ############### Calculate New values of Obejctive Function and volume Function ############

if(count>0)
{
C=0.0;
VecZeroEntries(Disp_Local);
VecZeroEntries(dC);
VecZeroEntries(xtmp);
VecZeroEntries(Change);
change=0.0;
maxval=0.0;
 tmp=0.0;
 vec_size=0.0;
}

DMGlobalToLocalBegin(da,U,INSERT_VALUES,Disp_Local);
DMGlobalToLocalEnd(da,U,INSERT_VALUES,Disp_Local);
VecGetArray(Disp_Local,&Uloc);
VecGetArray(dC,&dCdx); 
VecGetLocalSize(Disp_Local,&vec_size);

for (PetscInt i=0;i<nel;i++)
{ 
   for (PetscInt j=0;j<nn;j++)
	{ for (PetscInt k=0;k<3;k++)
		{  el_dof[j*3+k] = 3*elements[i* nn+j]+k; }
	}
uKu=0.0;
for (PetscInt k=0;k<24;k++){
			for (PetscInt h=0;h<24;h++){	
				uKu += Uloc[el_dof[k]]*ke[k*24+h]*Uloc[el_dof[h]];				
			}
}

// ###### Objective Function  ######################

C += (Emin + PetscPowScalar(x_physical[i],penal)*(E0 - Emin))*uKu;
dCdx[i]= -1.0 * penal*PetscPowScalar(x_physical[i],penal-1)*(E0- Emin)*uKu; 

}
//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Before C Value is :%g",C);
tmp=C;
C=0.0;
MPI_Allreduce(&tmp,&C,1,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD); 
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," After MPI Reduce C Value is :%g",C);

// ####### Calculate Volume Constraints Sensitivity  ################

VecSet(dV,1);
VecRestoreArray(Disp_Local,&Uloc);
VecRestoreArray(dC,&dCdx); 

VecAssemblyBegin(dC);CHKERRQ(ierr);
VecAssemblyEnd(dC);CHKERRQ(ierr);
VecAssemblyBegin(dV);CHKERRQ(ierr);
VecAssemblyEnd(dV);CHKERRQ(ierr); 

// ########## Calcuate  dv= H*(dv(:)./Hs); &   dc= H*(dc(:)./Hs);  ###################

VecPointwiseDivide(xtmp,dC,Hs);
MatMult(H,xtmp,dC);

// ######## Calculate Filter objective function  => H.(dV/Hs)  ####################

VecPointwiseDivide(xtmp,dV,Hs);
MatMult(H,xtmp,dV);


VecScale(dC,-1);
l1=0;
l2=1e9;
move=0.2;	

// ########### OC update loop Start  ##########################

 while (((l2-l1)/(l1+l2))>1e-3)
	  {  
  
		  fscale=1.0;
		  nloc=0;
		  VecZeroEntries(xtmp);
		  VecZeroEntries(x_temp);
		  VecZeroEntries(temp);
		  VecZeroEntries(X_pa);
		   invlmid=0;
	       lmid=0;
	       lmid =0.5*(l2+l1);
	       invlmid=(1.0/lmid);



		VecPointwiseDivide(x_temp,dC,dV);
		VecWAXPY(temp,invlmid,x_temp,dummy_zero);
		VecPow(temp,0.5);
		VecPointwiseMult(X_pa,X,temp);
		VecGetArray(X_pa,&x_pa);
		VecGetArray(X,&xp);
		VecGetArray(XNew,&x_new);
		VecGetLocalSize(X,&nloc);
		//VecView(X,PETSC_VIEWER_STDOUT_WORLD);

		// OC update scheme 
		
		for (i=0;i<nloc;i++)	
		{  
			x_ma=PetscMin(xp[i]+move,x_pa[i]);
			x_mb=PetscMax(xp[i]-move,PetscMin(1,x_ma));
			x_new[i]=PetscMax(0,x_mb);

		}
		  
		
	   	VecRestoreArray(X_pa,&x_pa);
		VecRestoreArray(X,&xp);
		VecRestoreArray(XNew,&x_new);
		MatMult(H,XNew,xtmp);
		VecPointwiseDivide(X_Phy,xtmp,Hs);
		sum=0.0;
		VecSum(X_Phy,&sum);
		VecGetSize(X_Phy,&vec_size);
		//PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n nele  : %D  vec_size:%D",nel,vec_size);
		
		if (sum > (volfrac*(vec_size)))
		{  l1=0;
			l1 = lmid;}
		else  
		{   l2=0;
			l2 = lmid;}	
			
		   it=it+1;
	  
	  } 
	  
// ########### OC update loop  End   #########################
	  
	  nloc=0.0;
	  VecGetLocalSize(XNew,&nloc);
	  VecGetArray(X,&xp);
	  VecGetArray(XNew,&x_new);
	  change=0.0;
	  
	  for ( i=0;i<nloc;i++)
	  { change=PetscMax(change,PetscAbsReal(x_new[i]-xp[i]));
	  } 
	  
	  VecRestoreArray(X,&xp);
	  VecRestoreArray(XNew,&x_new);
	  tmp=0.0;
	  MPI_Allreduce(&change, &tmp, 1,MPIU_SCALAR, MPI_MAX,PETSC_COMM_WORLD );
	  change=tmp;
	  PetscPrintf(PETSC_COMM_WORLD," \n Iteration number : %D ,C Value: %g , CHANGE IS:  %g",loop,C,change);
	  PetscViewerASCIIPushSynchronized(view2);
	  PetscViewerASCIIPrintf(view2," \n Iteration number : %D ,C Value: %g , CHANGE IS:  %g",loop,C,change);
	  VecZeroEntries(X);
	  VecZeroEntries(XDes);
	  VecCopy(XNew ,X);
	  VecCopy(XNew,XDes);
      //VecView(XNew,PETSC_VIEWER_STDOUT_WORLD); 
}



// ############# Destroy  vectors  #########################

VecDestroy(&Disp_Local);
VecDestroy(&xtmp);
VecDestroy(&temp);
VecDestroy(&x_temp);
VecDestroy(&dummy_zero);
VecDestroy(&X_pa);
VecDestroy(&X_Phy);
VecDestroy(&X);
VecDestroy(&Change);

}

/*
static PetscErrorCode SetupH_and_Hs(DM da) - function to calcuate distance between
neighboring elements based on filter radius Rmin
*/


static PetscErrorCode  SetupH_and_Hs(DM da)
{

// ############################  Create H Matrix  ##########################
DMCreateMatrix(da,&H);
Vec Rmin;
VecDuplicate(XDes,&Rmin);
VecSet(Rmin,rmin);
MatDiagonalSet(H,Rmin,INSERT_VALUES);					
MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);

//MatView(H,PETSC_VIEWER_STDOUT_WORLD);

// ###################################  Create Hs Matrix ###################

DMCreateGlobalVector(da,&Hs);
Vec dummy;
MatGetVecs(H,&Hs,&dummy);CHKERRQ(ierr);
VecDuplicate(Hs,&dummy);
VecSet(dummy,1.0);
MatMult(H,dummy,Hs);
VecAssemblyBegin(Hs);CHKERRQ(ierr);
VecAssemblyEnd(Hs);CHKERRQ(ierr);

//VecView(Hs,PETSC_VIEWER_STDOUT_WORLD);
VecDestroy(&Rmin);

}


/*
Create3DVTK(DM da) - function to create .vtk file for Paraview  visualization and also create the .txt file to write final 
XDes variable values
*/


static PetscErrorCode Create3DVTK(DM da)

{
	
   Vec xg;
   DMDAGetLocalInfo(da,&info);
   DMCreateGlobalVector(da,&ug);
   DMDAVecGetArray(da,ug,&Mesh_Coord);
   for (k=info.zs; k<info.zs+info.zm; k++) {
          for (j=info.ys; j<info.ys+info.ym; j++) {
               for (i=info.xs; i<info.xs+info.xm; i++) 
			   {
		     
			Mesh_Coord[k][j][i].x=i;  // X-Coordinate
		    Mesh_Coord[k][j][i].y=j;  // Y- Coorindate
			Mesh_Coord[k][j][i].z=k; // Z-Coordinate  
			   }
		  }
   }
  
		
	DMDAVecRestoreArray(da,ug,&Mesh_Coord);CHKERRQ(ierr);
	PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&view1);
	PetscObjectSetName((PetscObject) ug, "Co-oordinates_");
	VecView(ug,view1); // Copy Coordinates of DMDA
	PetscObjectSetName((PetscObject) U, "Displacement_");
	VecView(U,view1);  // Copy Displacement Vector to file1
	PetscObjectSetName((PetscObject) F, "Force_");
	VecView(F,view1); // Copy F Vector
	
	
	//PetscObjectSetName((PetscObject) XDes, "DesignVariable_");
	//VecView(XDes,view1); 
	//VecView(XDes,PETSC_VIEWER_STDOUT_WORLD); 
	
	// File 2- Create Similar Size of Design Variable Vector for Paraview to map with da 

	/* 

	//PetscViewerSetFormat(view2, PETSC_VIEWER_ASCII_VTK);
	//DMDASetUniformCoordinates(da_element, 0.0, nelx+1, 0.0, nely+1, 0.0,nelz+1);
	DMView(da_element,view2);
	PetscObjectSetName((PetscObject) XDes, "XDesignVariable_");
	VecView(XDes, view2);
	PetscViewerASCIIPushSynchronized(view2);
	PetscViewerASCIISynchronizedPrintf(view2,"HELLO");
	
	*/
	
	
     Vec T; 
	 PetscScalar *xnew;
	 PetscScalar x[24];
	 DMCreateGlobalVector(da,&T);
	 VecSetLocalToGlobalMapping(T,ltogm);
	 VecGetArray(XNew,&xnew);
	 
	 for (i=0;i<nel;i++)
		 { for (j=0;j<nn;j++)
			 {	
				for (k=0;k<3;k++)
				{
				el_dof[j*3+k] = 3*elements[i*nn+j]+k;
				}
			 }
				 for (k=0;k<24;k++)
				 {
					 x[k]=xnew[i];
				 }		 
				VecSetValuesLocal(T,24,el_dof,x,INSERT_VALUES);
		 }
		 
	VecRestoreArray(XNew,&xnew); 
	VecAssemblyBegin(T);
	VecAssemblyEnd(T);
	//VecView(T,PETSC_VIEWER_STDOUT_WORLD);
		 
	
	
	PetscObjectSetName((PetscObject) T, "DesignVariable_");
	VecView(T,view1);
	//VecView(T,view2);
	
	//   Destroy Vectors
	PetscViewerDestroy(&view1);
	VecDestroy(&ug);
	VecDestroy(&T);
	return 0;                                                             
   
}


/*
PetscErrorCode SolveState(DM da,Mat K)- function to find Displacement U Values
*/


 static PetscErrorCode SolveState(DM da,Mat K)
 {  
   
if (count>0) 
{
   KSPSetInitialGuessNonzero(ksp1,PETSC_TRUE);
   VecZeroEntries(U);
}
 
 //KSPGetPC(ksp1,&pc);
 //PCSetType(pc,PCMG);
 KSPSetType(ksp1,KSPCG);
 //ierr=KSPSetFromOptions(ksp);
//ierr = KSPSetOperators(ksp,KGlobal,KGlobal);
ierr = KSPSetOperators(ksp1,K,K);
ierr = KSPSetTolerances(ksp1,1e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
//ierr = KSPView(ksp1,PETSC_VIEWER_STDOUT_WORLD);
//ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD); 
ierr = KSPSolve(ksp1,F,U);
//ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);
//MatDestroy(&KGlobal);
//VecDestroy(&F);
//KSPDestroy(&ksp1);
count=count+1;
//PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Count :%D \n",count);
return 0;
	 
 }
  
  
 /*
PetscErrorCode ElementStiffnessMatrixCalc(DM da)- function to find ElementStiffnessMatrix
*/
 
  static PetscErrorCode ElementStiffnessMatrixCalc(DM da)
  {


 PetscMemzero(ke,sizeof(PetscScalar)*nn*dof*nn*dof);
 DMDAGetLocalInfo(da,&info);	
   for (k=info.zs; k<info.zs+info.zm; k++) {
    for (j=info.ys; j<info.ys+info.ym; j++) {
       for (i=info.xs; i<info.xs+info.xm; i++) {
	
			DMDAGetElementEqnums_u(i,j,k);		   
            ElementStiffnessMatrix(X,Y,Z,nu,ke);
		
	   }
	}
   }
   
  }

  
   /*
PetscErrorCode AssembleStiffnessMatrix(DM da)- function to find Global Stiffness Matrix
*/
  
 static PetscErrorCode  AssembleStiffnessMatrix(DM da)
 {
	 
	  
  if (count>0)
{
	MatZeroEntries(KGlobal);
    VecZeroEntries(NI);
}

 
VecGetArray(XDes,&xp);
VecDuplicate(BC,&NI);
MatSetLocalToGlobalMapping(KGlobal,ltogm,ltogm); 

for (i=0;i<nel;i++){
		// loop over element nodes
	//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nElement Number:%D ---[",i);	
		for ( j=0;j<nn;j++){
			// Get local dofs
			for (k=0;k<3;k++){
			//	edof[j*3+k] = 3*necon[i*nn+j]+k;
				el_dof[j*3+k] = 3*elements[i* nn+j]+k;
		//	PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d  ",el_dof[j*3+k]);	
			}
		}
 // Use SIMP for stiffness interpolation
Density= Emin + PetscPowScalar(xp[i],penal)*(E0-Emin);
//Density=1;
 for (k=0;k<24*24;k++)
		   {
			//PetscSynchronizedPrintf(PETSC_COMM_WORLD," ke[%d]:%f \n",k,ke[k]);	   
		  KF[k] = ke[k]*Density; 
		 // PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n K index no: (%D) ,K Value: %f , KF Value :%f\n",k,ke[k],KF[k]);	 
		   }

		 MatSetValuesLocal(KGlobal,24,el_dof,24,el_dof,KF,ADD_VALUES);
}
 MatAssemblyBegin(KGlobal,MAT_FINAL_ASSEMBLY);
 MatAssemblyEnd(KGlobal,MAT_FINAL_ASSEMBLY);
 //MatView(KGlobal,PETSC_VIEWER_STDOUT_WORLD);
MatSetOption(KGlobal,MAT_NEW_NONZERO_LOCATIONS,PETSC_TRUE);

 
 // Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
	// 1.: K = N'*K*N
	MatDiagonalScale(KGlobal,BC,BC);
	// 2. Add ones, i.e. K = K + NI, NI = I - N

	VecSet(NI,1.0);
	VecAXPY(NI,-1.0,BC);
	MatDiagonalSet(KGlobal,NI,ADD_VALUES);

	// Zero out possible loads in the RHS that coincide
	// with Dirichlet conditions
	VecPointwiseMult(F,F,BC);
	//VecDestroy(&NI);
	count=count+1;
  }
   
  
 /*
PetscErrorCode  DMDAGetElementEqnums_u- function to find Co-ordinates of nodes of each element
*/
  
   
   static PetscErrorCode DMDAGetElementEqnums_u(PetscInt i,PetscInt j,PetscInt k)
 {
 

 // Node 0
 
 X[0]= i; Y[0] = j; Z[0]= k;            

 
 // Node 1
 X[1]= i+1; Y[1] = j; Z[1]= k;             

 
  
 // Node 2
 X[2]= i+1; Y[2] = j+1; Z[2]= k;     

 
   
 // Node 3
 X[3]= i; Y[3] = j+1; Z[3]= k;     

 
 // Node 4
X[4]= i; Y[4] = j; Z[4]= k+1; 

 
  // Node 5
  X[5]= i+1; Y[5] = j; Z[5]= k+1; 
 
 
 
   // Node 6
  X[6]= i+1; Y[6] = j+1; Z[6]= k+1; 

 
 // Node 7
   X[7]= i; Y[7] = j+1; Z[7]= k+1; 

 } 
 

 
  /*
PetscErrorCode  PetscErrorCode ElementStiffnessMatrix- Find Element Stifnness Matrix K(0)
Some Function block statements created/modified with the help and reference to the following source:
1. ToPy -- Topology optimisation with Python, William Hunter ( https://code.google.com/p/topy/) 
2. C algorithm for 3-dimensional compliant mechanism design by topology optimization-(https://github.com/vbfall/top-opt-3d)

*/
 
 static PetscErrorCode ElementStiffnessMatrix(PetscScalar X[],PetscScalar Y[],PetscScalar Z[],PetscScalar nu,PetscScalar *ke)
 {

 // Lame's parameters (with E=1.0):
PetscScalar lambda = nu/((1.0+nu)*(1.0-2.0*nu));
PetscScalar mu = 1.0/(2.0*(1.0+nu)); 

// Constitutive matrix
	PetscScalar C[6][6] = {{lambda+2.0*mu, lambda, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda+2.0*mu, lambda, 0.0, 0.0, 0.0},
		{lambda, lambda, lambda+2.0*mu, 0.0, 0.0, 0.0},
		{0.0,    0.0,    0.0,           mu,  0.0, 0.0},
		{0.0, 	0.0, 	0.0, 		   0.0, mu,  0.0}, 
        {0.0, 0.0, 0.0, 0.0, 0.0, mu}};
		
// Gauss points (GP) and weigths
// Two Gauss points in all directions (total of eight)
	PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626}; 
// Corresponding weights
    PetscScalar W[2] = {1.0, 1.0};
	
// Matrices that help when we gather the strain-displacement matrix:
	PetscScalar alpha1[6][3]; PetscScalar alpha2[6][3]; PetscScalar alpha3[6][3];
	memset(alpha1, 0, sizeof(alpha1[0][0])*6*3); // zero out
	memset(alpha2, 0, sizeof(alpha2[0][0])*6*3); // zero out
	memset(alpha3, 0, sizeof(alpha3[0][0])*6*3); // zero out
	alpha1[0][0] = 1.0; alpha1[3][1] = 1.0; alpha1[5][2] = 1.0;
	alpha2[1][1] = 1.0; alpha2[3][0] = 1.0; alpha2[4][2] = 1.0;
    alpha3[2][2] = 1.0; alpha3[4][1] = 1.0; alpha3[5][0] = 1.0;
	
PetscScalar dNdxi[8]; PetscScalar dNdeta[8]; PetscScalar dNdzeta[8];
PetscScalar J[3][3];
PetscScalar invJ[3][3];
PetscScalar beta[6][3];
PetscScalar B[6][24]; 

PetscScalar *dN;

// Make sure the stiffness matrix is zeroed out:
memset(ke, 0, sizeof(ke[0])*24*24);

// Perform the numerical integration
	for (PetscInt ii=0; ii<2; ii++){
		        for (PetscInt jj=0; jj<2; jj++){
                         for (PetscInt kk=0; kk<2; kk++){
							 
// Integration point
				PetscScalar xi = GP[ii]; 
				PetscScalar eta = GP[jj]; 
                PetscScalar zeta = GP[kk];
		
//Compute differentiated shape functions
	// At the point given by (xi, eta, zeta).
	// With respect to xi:
	dNdxi[0]  = -0.125*(1.0-eta)*(1.0-zeta);
	dNdxi[1]  =  0.125*(1.0-eta)*(1.0-zeta);
	dNdxi[2]  =  0.125*(1.0+eta)*(1.0-zeta);
	dNdxi[3]  = -0.125*(1.0+eta)*(1.0-zeta);
	dNdxi[4]  = -0.125*(1.0-eta)*(1.0+zeta);
	dNdxi[5]  =  0.125*(1.0-eta)*(1.0+zeta);
	dNdxi[6]  =  0.125*(1.0+eta)*(1.0+zeta);
	dNdxi[7]  = -0.125*(1.0+eta)*(1.0+zeta);
	// With respect to eta:
	dNdeta[0] = -0.125*(1.0-xi)*(1.0-zeta);
	dNdeta[1] = -0.125*(1.0+xi)*(1.0-zeta);
	dNdeta[2] =  0.125*(1.0+xi)*(1.0-zeta);
	dNdeta[3] =  0.125*(1.0-xi)*(1.0-zeta);
	dNdeta[4] = -0.125*(1.0-xi)*(1.0+zeta);
	dNdeta[5] = -0.125*(1.0+xi)*(1.0+zeta);
	dNdeta[6] =  0.125*(1.0+xi)*(1.0+zeta);
	dNdeta[7] =  0.125*(1.0-xi)*(1.0+zeta);
	// With respect to zeta:
	dNdzeta[0]= -0.125*(1.0-xi)*(1.0-eta);
	dNdzeta[1]= -0.125*(1.0+xi)*(1.0-eta);
	dNdzeta[2]= -0.125*(1.0+xi)*(1.0+eta);
	dNdzeta[3]= -0.125*(1.0-xi)*(1.0+eta);
	dNdzeta[4]=  0.125*(1.0-xi)*(1.0-eta);
	dNdzeta[5]=  0.125*(1.0+xi)*(1.0-eta);
	dNdzeta[6]=  0.125*(1.0+xi)*(1.0+eta);
	dNdzeta[7]=  0.125*(1.0-xi)*(1.0+eta);
	
	
	// JacobianMatrix
	J[0][0] = Dot(dNdxi,X,8); J[0][1] = Dot(dNdxi,Y,8); J[0][2] = Dot(dNdxi,Z,8);
	J[1][0] = Dot(dNdeta,X,8); J[1][1] = Dot(dNdeta,Y,8); J[1][2] = Dot(dNdeta,Z,8);
    J[2][0] = Dot(dNdzeta,X,8); J[2][1] = Dot(dNdzeta,Y,8); J[2][2] = Dot(dNdzeta,Z,8);
						// PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n J [1][1] %f,J[2][2] %f  \n",J[2][2] ,J[1][1] );	
	// Inverse and determinant
	 PetscScalar detJ = Inverse3M(J, invJ);
	// Weight factor at this point
     PetscScalar weight = W[ii]*W[jj]*W[kk]*detJ;
	/*  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n Wii %f,Wjj %f,,Wkk %f ,detJ %f  \n",W[ii],W[jj],W[kk],detJ);	 */
	 // Strain-displacement matrix
				memset(B, 0, sizeof(B[0][0])*6*24); // zero out
				for (PetscInt ll=0; ll<3; ll++)
				{
					// Add contributions from the different derivatives
					if (ll==0) {dN = dNdxi;}
					if (ll==1) {dN = dNdeta;}
					if (ll==2) {dN = dNdzeta;}
					// Assemble strain operator
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<3; j++){
							beta[i][j] = invJ[0][ll]*alpha1[i][j]+invJ[1][ll]*alpha2[i][j]+invJ[2][ll]*alpha3[i][j];
               				}
                      }
// Add contributions to strain-displacement matrix
					for (PetscInt i=0; i<6; i++){
						for (PetscInt j=0; j<24; j++){
							B[i][j] = B[i][j] + beta[i][j%3]*dN[j/3];
						}
					}
                 }	
				 
// Finally, add to the element matrix
				for (PetscInt i=0; i<24; i++){
					for (PetscInt j=0; j<24; j++){
						for (PetscInt k=0; k<6; k++){
							for (PetscInt l=0; l<6; l++){
								
								ke[j+24*i] = ke[j+24*i] + weight*(B[k][i] * C[k][l] * B[l][j]);
								//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n Weight %f,B value %f,,C Value %f \n",weight,B[i][j],C[k][l]);	
						//		PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n K index no: (%D,%D) ,K Value: %f \n",i,j,ke[j+24*i]);	 
							}
						}
					}
				}
			}
		}
	}
return 0;				 
				 
	 
 }
 
 
 
 /*
  static PetscScalar Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l) - Function that returns the dot product of v1 and v2, ,which must have the same length l
 */
  static PetscScalar Dot(PetscScalar *v1, PetscScalar *v2, PetscInt l)
  {

	PetscScalar result = 0.0;
	for (PetscInt i=0; i<l; i++)
	{
		result = result + v1[i]*v2[i];
	}
	return result;
}

 /*
 static  PetscScalar Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3])-  Computes the inverse of a 3x3 matrix
 */

 static  PetscScalar Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3])
{
	PetscScalar detJ = J[0][0]*(J[1][1]*J[2][2]-J[2][1]*J[1][2])-J[0][1]*(J[1][0]*J[2][2]-J[2][0]*J[1][2])+J[0][2]*(J[1][0]*J[2][1]-J[2][0]*J[1][1]);
	invJ[0][0] = (J[1][1]*J[2][2]-J[2][1]*J[1][2])/detJ;
	invJ[0][1] = -(J[0][1]*J[2][2]-J[0][2]*J[2][1])/detJ;
	invJ[0][2] = (J[0][1]*J[1][2]-J[0][2]*J[1][1])/detJ;
	invJ[1][0] = -(J[1][0]*J[2][2]-J[1][2]*J[2][0])/detJ;
	invJ[1][1] = (J[0][0]*J[2][2]-J[0][2]*J[2][0])/detJ;
	invJ[1][2] = -(J[0][0]*J[1][2]-J[0][2]*J[1][0])/detJ;
	invJ[2][0] = (J[1][0]*J[2][1]-J[1][1]*J[2][0])/detJ;
	invJ[2][1] = -(J[0][0]*J[2][1]-J[0][1]*J[2][0])/detJ;
	invJ[2][2] = (J[0][0]*J[1][1]-J[1][0]*J[0][1])/detJ;
	return detJ;
}
  
  
    /*
PetscErrorCode  PetscErrorCode  ForceVectorSetUp(DM da)- Set Up Force and Boundary Vector

3 Cases - 

1.Cantilever Beam Topology Optimization with Line Load Along Edge-(Fig 5 from report) 
2.Cantilever Beam Topology Optimization with Single Upper Load - (Fig 11 From Report) 
3.Cantilever Beam Topology Optimization with Single Load at Lower Edge-(Fig 8 From Report)

Default Code Solves  - Cantilever Beam Topology Optimization with Line Load Along Edge-(Fig 5 from report) 

Uncomment specific case to exeucte specific load condition
*/

static PetscErrorCode ForceVectorSetUp(DM da)
{    
 //PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Setting Up FORCE VECTOR \n");
  PetscScalar          *FA,bc_value=0.0;
  //DMDALocalInfo     Info;
  Vec                         Local_Mesh_Coords,FLoc,BCLoc;
  PetscInt                  ix,jy,kz,l,size,m;
    PetscInt                  M=0,N=0,P=0;
  DMDACoor3d        ***Coor_3D;
  PetscViewer       viewer;
  DMCreateGlobalVector(da,&F);
  DMCreateGlobalVector(da,&BC);
  //DMGetLocalToGlobalMapping(da,&ltogm);
  DMCreateLocalVector(da,&FLoc);
  DMCreateLocalVector(da,&BCLoc);
  //VecSetLocalToGlobalMapping(F,ltogm);
  VecZeroEntries(FLoc);
   // Create Duplicate Displacement Vector Like Force Vector
   VecDuplicate(F,&(U));
   // VecDuplicate(F,&(BC));
   VecSet(U,0.0);
     VecZeroEntries(F);
	 VecZeroEntries(BC);
   // Create Boundary Value Vector
   //VecDuplicate(FLoc,&(BCLoc));
   VecSet(BCLoc,1.0);
 
  //VecGetSize(F,&size);
  
  DMDAGetLocalInfo(da,&info);
 //PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in M,N,P  :%D%D%D\n",M,N,P);
  DMCreateLocalVector(da,&Local_Mesh_Coords);
   VecGetSize(Local_Mesh_Coords,&size);
  // PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Size of Local Mesh :%D \n",size);
  DMDAVecGetArray(da,Local_Mesh_Coords,&Coor_3D);
  
  // Get Limits of Grid
  x_min=0.0;
  x_max=(info.mx)-1;
  y_min=0.0;
  y_max=(info.my)-1;
  z_min=0.0;
  z_max=(info.mz)-1;  
  
 for (k=info.zs; k<info.zs+info.zm; ++k) {
              for (j=info.ys; j<info.ys+info.ym; j++) {
                    for (i=info.xs; i<info.xs+info.xm; i++) { 
				 
				      Coor_3D[k][j][i].x=i;  // X-Coordinate
		              Coor_3D[k][j][i].y=j;  // Y- Coorindate
			          Coor_3D[k][j][i].z=k; // Z-Coordinate
					  // PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in X :%D\n",i);
               //     PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in Coord 3DX :%g \n",(double)Coor_3D[k][j][i].x);
				 	//  PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in XYZ  :%D%D%D\n",i,j,k);
			/* 		 PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in Y :%D\n",j);
				     PetscSynchronizedPrintf(PETSC_COMM_WORLD," \n Coordinate Value in Z:%D \n",k); 	  */
		 } 
	}
	}   
DMDAVecRestoreArray(da,Local_Mesh_Coords,&Coor_3D);
VecGetArray(Local_Mesh_Coords,&FA);
VecGetSize(Local_Mesh_Coords,&size);

 for(m=0;m<size;m++)
 {   


 // ### Cantilever Beam Topology Optimization with Line Load Along Edge-(Fig 5 from report) ###############################################
 
   // Left Wall Clamped
   
	if(FA[m] ==x_min && (m%3==0))  // M%3-implies the x-coordinate is always at entry point of multiple of 3
	{     
		 // PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Positive %D\n",m);
		  VecSetValue(BCLoc,m,0.0,INSERT_VALUES); // X-Direction
		  VecSetValue(BCLoc,m+1,0.0,INSERT_VALUES); // Y-Direction
		  VecSetValue(BCLoc,m+2,0.0,INSERT_VALUES); // Z-Direction
		
	}
	 // Line Load on Extreme Bottom of Body
	if ( FA[m] == x_max && FA[m+1] ==y_min && (m%3==0))
	{
		//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Positive %D\n",m);
		VecSetValue(FLoc,m+1,-1,INSERT_VALUES); // -ve -Y-Direction (Downward Force)
	}
	
	if ( FA[m] == x_max && FA[m+1] ==y_min  && FA[m+2]==z_max && (m%3==0))
	{
		//PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Positive %D\n",m);
		VecSetValue(FLoc,m+1,-1,INSERT_VALUES); // -ve -Y-Direction (Downward Force)
	} 
	
	// ###################### Cantilever Beam Topology Optimization with Single Upper Load - (Fig 11 From Report) #############################
	
	  // Left Wall Clamped
/* 	if(FA[m] ==x_min && (m%3==0))  // M%3-implies the x-coordinate is always at entry point of multiple of 3
	{     
		 // PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Positive %D\n",m);
		  VecSetValue(BCLoc,m,0.0,INSERT_VALUES); // X-Direction
		  VecSetValue(BCLoc,m+1,0.0,INSERT_VALUES); // Y-Direction
		  VecSetValue(BCLoc,m+2,0.0,INSERT_VALUES); // Z-Direction
		
	}
	
	
		// Single Load Middle Top Right Of Body
		if ( FA[m] == x_max && FA[m+1] ==y_max  && FA[m+2]==z_max /2&& (m%3==0))
	{
	
		VecSetValue(FLoc,m+1,1,INSERT_VALUES); // +ve Y-Direction (upward Force)
	} */
	

	 	// ###################### Cantilever Beam Topology Optimization with Single Load at Lower Edge-(Fig 8 From Report)#############################
	
	  
/* 	  
	  // Left Wall Clamped
     if(FA[m] ==x_min && (m%3==0))  // M%3-implies the x-coordinate is always at entry point of multiple of 3
	{     
		 // PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Positive %D\n",m);
		  VecSetValue(BCLoc,m,0.0,INSERT_VALUES); // X-Direction
		  VecSetValue(BCLoc,m+1,0.0,INSERT_VALUES); // Y-Direction
		  VecSetValue(BCLoc,m+2,0.0,INSERT_VALUES); // Z-Direction
		
	}
	
		// Single Load Bottom Of Body
		if ( FA[m] == x_max && FA[m+1] ==y_min  && FA[m+2]==z_max /2&& (m%3==0))
	{

		VecSetValue(FLoc,m+1,-1,INSERT_VALUES); // -ve Y-Direction (Downward Force)
	}

	 
	 	
	 */
	
 } 
 
 
 
VecAssemblyBegin(FLoc);
VecAssemblyBegin(BCLoc);
VecAssemblyEnd(FLoc);
VecAssemblyEnd(BCLoc);
 //VecView(BCLoc,PETSC_VIEWER_STDOUT_WORLD);
DMLocalToGlobalBegin(da,FLoc,ADD_VALUES,F);
DMLocalToGlobalEnd(da,FLoc,ADD_VALUES,F);
DMLocalToGlobalBegin(da,BCLoc,ADD_VALUES,BC);
DMLocalToGlobalEnd(da,BCLoc,ADD_VALUES,BC);
/* VecAssemblyBegin(F);
VecAssemblyBegin(BC);
VecAssemblyEnd(BC);
VecAssemblyEnd(F); */
// VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  //VecView(BC,PETSC_VIEWER_STDOUT_WORLD);
  VecDestroy(&FLoc);
   VecDestroy(&BCLoc);

 

	
}




 
 
 

