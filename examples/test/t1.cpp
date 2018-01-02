#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.1 ;
const double R_gas = 287;
const double   Pr  = 0.72;

//Run parameters
//const char *mesh_file        =  "periodic-square.mesh";
const char *mesh_file        =  "periodic-cube.mesh";
//const char *mesh_file        =  "ldc.msh";
//const char *mesh_file        =  "cylinder_visc.msh";
const int    order           =  1;
const double t_final         =  0.00010;
const int    problem         =  1;
const int    ref_levels      =  0;

const bool   time_adapt      =  false;
const double cfl             =  0.20;
const double dt_const        =  0.0001 ;
const int    ode_solver_type =  1; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       =  1000;

const bool   adapt           =  false;
const int    adapt_iter      =  200  ; // Time steps after which adaptation is done
const double tolerance       =  5e-4 ; // Tolerance for adaptation


// Velocity coefficient
void init_function(const Vector &x, Vector &v);


class CNS 
{
private:
    int num_procs, myid;

    ParMesh *pmesh ;

    ParFiniteElementSpace  *fes, *fes_vec, *fes_op;
   
    ParBilinearForm *m, *k_inv_x, *k_inv_y, *k_inv_z;
    ParBilinearForm     *k_vis_x, *k_vis_y, *k_vis_z;

    ParGridFunction *u_sol, *f_inv;   
    HypreParVector  *U;
    ParGridFunction *u_b, *f_I_b;   // Used in calculating b
    ParLinearForm   *b;

    ODESolver *ode_solver; 
    ParGridFunction *u_t, *k_t, *y_t;   

    int dim;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, t, glob_u_max;
    int ti;

    ofstream tke_file;
public:
   CNS();

};





int main(int argc, char *argv[])
{
    // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   int precision = 8;
   cout.precision(precision);

   CNS run;

   return 0;
}

CNS::CNS() 
{
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // Read the mesh from the given mesh file
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
           dim = mesh->Dimension();
   int var_dim = dim + 2;
   int aux_dim = dim + 1; //Auxiliary variables for the viscous terms

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   double kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   DG_FECollection fec(order, dim);
   fes = new ParFiniteElementSpace(pmesh, &fec, var_dim);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   VectorFunctionCoefficient u0(var_dim, init_function);
   
   u_sol = new ParGridFunction(fes);
   u_sol->ProjectCoefficient(u0);

   U = u_sol->GetTrueDofs();

   double tol = std::numeric_limits<double>::epsilon();

   Poly_1D poly1d;
   int p_basis = order , type = 0;
   int Np = p_basis + 1;
   Poly_1D::Basis &basis1d(poly1d.OpenBasis(p_basis, type));

   const double *pts = poly1d.GetPoints(p_basis, type);
   
//   for (int i = 0; i < p_basis + 1; i++)
//   {
//       cout << 2*pts[i] - 1 << "\t"   ; // Get LGL points
//   }
//   cout << endl;

//   Vector P(p_basis + 1);
//   poly1d.CalcLegendreBasis(p_basis, pts[2], P);
//   for (int i = 0; i < p_basis + 1; i++)
//   {
//       cout << P[i] << "\t"   ; // Get LGL points
//   }
//   cout << endl;

   Vector P(Np);
   DenseMatrix van(Np);
   for (int i = 0; i < Np; i++)
   {
       poly1d.CalcLegendreBasis(p_basis, pts[i], P);
       van.SetRow(i, P);
   
       for (int j = 0; j < Np; j++)
       {
           double p_gamma = 2.0/(2.0*j + 1);
           van(i, j)   = van(i, j)/sqrt(p_gamma);
       }
   }

   DenseMatrix inv_van(van);
   inv_van.Transpose();
   inv_van.Invert();

//   van.Print();
//   inv_van.Print();
   Vector mod_lag_R(Np), mod_lag_L(Np);
   poly1d.CalcLegendreBasis(p_basis, 0.0, P);
   for (int j = 0; j < Np; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np - 1] *= 0.8;
   inv_van.Mult(P, mod_lag_L);
   poly1d.CalcLegendreBasis(p_basis, 1.0, P);
   for (int j = 0; j < Np; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np - 1] *= 0.8;
   inv_van.Mult(P, mod_lag_R);
//   cout << mod_lag_L[0] << "\t" << mod_lag_L[1] << "\t" << mod_lag_L[2] << endl;
//   cout << mod_lag_R[0] << "\t" << mod_lag_R[1] << "\t" << mod_lag_R[2] << endl;



   FaceElementTransformations *Trans; 
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   int ndof1, ndof2;
   Vector shape1, shape2;
   Vector mod_shape1, mod_shape2;
   Vector shape_x(p_basis + 1), shape_y(p_basis + 1), shape_z(p_basis + 1);

//   for (int i = 0; i < pmesh->GetNumFaces(); i++)
   for (int i = 0; i < 10; i++)
   {
       Trans = pmesh->GetInteriorFaceTransformations(i);
       if (Trans != NULL)
       {
   
           fe1 = fes->GetFE(Trans->Elem1No);
           fe2 = fes->GetFE(Trans->Elem2No);
       
           fes->GetElementVDofs(Trans->Elem1No, vdofs1);
           fes->GetElementVDofs(Trans->Elem2No, vdofs2);
           
           ndof1 = fe1->GetDof();
           ndof2 = fe2->GetDof();
    
           shape1.SetSize(ndof1);
           shape2.SetSize(ndof2);

           mod_shape1.SetSize(ndof1);
           mod_shape2.SetSize(ndof2);
   
           int f_order;
           f_order = (std::min(Trans->Elem1->OrderW(), Trans->Elem2->OrderW()) +
                   2*std::max(fe1->GetOrder(), fe2->GetOrder()));
           const IntegrationRule *ir  = &IntRules.Get(Trans->FaceGeom, f_order);
    
           for (int p = 0; p < ir->GetNPoints(); p++)
           {
              const IntegrationPoint &ip = ir->IntPoint(p);
              IntegrationPoint eip1, eip2;
              Trans->Loc1.Transform(ip, eip1);
              if (ndof2)
              {
                 Trans->Loc2.Transform(ip, eip2);
              }
    
              Trans->Face->SetIntPoint(&ip);
              Trans->Elem1->SetIntPoint(&eip1);
              Trans->Elem2->SetIntPoint(&eip2);
    
              fe1->CalcShape(eip1, shape1);
              fe2->CalcShape(eip2, shape2);
    
              basis1d.Eval(eip1.x, shape_x);
              basis1d.Eval(eip1.y, shape_y);

              if (abs(eip1.x) < tol)
              {
                  shape_x = mod_lag_L;              
              }
              else if ( 1 - abs(eip1.x) < tol)
              {
                  shape_x = mod_lag_R;              
              }
              if (abs(eip1.y) < tol)
              {
                  shape_y = mod_lag_L;              
              }
              else if ( 1 - abs(eip1.y) < tol)
              {
                  shape_y = mod_lag_R;              
              }

              if (dim == 2)
              {
                  mod_shape1 = 0.0;
                  for (int ot = 0, jt = 0; jt <= p_basis; jt++)
                      for (int it = 0; it <= p_basis; it++)
                      {
                          mod_shape1(ot++) = shape_x(it)*shape_y(jt);
                      }
              }
              else if (dim == 3)
              {
                  basis1d.Eval(eip1.z, shape_z);
                  if (abs(eip1.z) < tol)
                  {
                      shape_z = mod_lag_L;              
                  }
                  else if ( 1 - abs(eip1.z) < tol)
                  {
                      shape_z = mod_lag_R;              
                  }
                  for (int ot = 0, kt = 0; kt <= p_basis; kt++)
                      for (int jt = 0; jt <= p_basis; jt++)
                          for (int it = 0; it <= p_basis; it++)
                          {
                              mod_shape1(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                          }
              }
//              cout << eip1.x << "\t" << eip1.y<< "\t" << eip1.z << endl;
//              cout << eip2.x << "\t" << eip2.y<< "\t" << eip2.z << endl;
//              cout << eip1.x << "\t" << shape_x[0] << "\t" << shape_x[1] << "\t" << shape_x[2] << endl;
              cout << shape1[0] << "\t" << shape1[1] << "\t" << shape1[2] << endl;
              cout << mod_shape1[0] << "\t" << mod_shape1[1] << "\t" << mod_shape1[2] << endl;

              basis1d.Eval(eip2.x, shape_x);
              basis1d.Eval(eip2.y, shape_y);
                  
              if (abs(eip2.x) < tol)
              {
                  shape_x = mod_lag_L;              
              }
              else if ( 1 - abs(eip2.x) < tol)
              {
                  shape_x = mod_lag_R;              
              }
              if (abs(eip2.y) < tol)
              {
                  shape_y = mod_lag_L;              
              }
              else if ( 1 - abs(eip2.y) < tol)
              {
                  shape_y = mod_lag_R;              
              }

              if (dim == 2)
              {
                  mod_shape2 = 0.0;
                  for (int ot = 0, jt = 0; jt <= p_basis; jt++)
                      for (int it = 0; it <= p_basis; it++)
                      {
                          mod_shape2(ot++) = shape_x(it)*shape_y(jt);
                      }
              }
              else if (dim == 3)
              {
                  basis1d.Eval(eip2.z, shape_z);
                  if (abs(eip2.z) < tol)
                  {
                      shape_z = mod_lag_L;              
                  }
                  else if ( 1 - abs(eip2.z) < tol)
                  {
                      shape_z = mod_lag_R;              
                  }
                  for (int ot = 0, kt = 0; kt <= p_basis; kt++)
                      for (int jt = 0; jt <= p_basis; jt++)
                          for (int it = 0; it <= p_basis; it++)
                          {
                              mod_shape2(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                          }
              }


//              cout << shape2[0] << "\t" << shape2[1] << "\t" << shape2[2] << endl;
//              cout << mod_shape2[0] << "\t" << mod_shape2[1] << "\t" << mod_shape2[2] << endl;

//              cout << i << "\t" << p << "\t" << shape1[0] << "\t" << shape1[1] << "\t" << shape1[2] << endl;
      
           }
       }

   }

}




//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p;

       if (problem == 0) // Smooth periodic density. Exact solution to Euler 
       {
           rho =  1.0 + 0.2*sin(M_PI*(x(0) + x(1)));
           p   =  1.0; 
           u1  =  1.0;
           u2  =  1.0;
       }
       else if (problem == 1) // Taylor Green Vortex; exact solution of incompressible NS
       {
           rho =  1.0;
           p   =  100 + (rho/4.0)*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)));
           u1  =      sin(M_PI*x(0))*cos(M_PI*x(1));
           u2  = -1.0*cos(M_PI*x(0))*sin(M_PI*x(1));
       }
       else if (problem == 2) // Isentropic vortex; exact solution for Euler
       {
           double T, M;
           double beta, omega, f, du, dv, dT;
    
           beta = 5;
    
           f     = (pow(x(0), 2) + pow(x(1), 2));
           omega = (beta/(2*M_PI))*exp(0.5*(1 - f));
           du    = -x(1)*omega;
           dv    =  x(0)*omega;
           dT    = - (gamm - 1)*beta*beta/(8*gamm*M_PI*M_PI) * exp(1 - f);
    
           u1    = 1 + du;
           u2    = dv;
           T     = 1 + dT;
           rho   = pow(T, 1/(gamm - 1));
           p     = rho*R_gas*T;

       }
       
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
   else if (dim == 3)
   {
       double rho, u1, u2, u3, p;

       if (problem == 0) // Smooth periodic density. Exact solution to Euler 
       {
           rho =  1.0 + 0.2*sin(M_PI*(x(0) + x(1) + x(3)));
           p   =  1.0; 
           u1  =  1.0;
           u2  =  1.0;
           u3  =  1.0;
       }
       else if (problem == 1) // Taylor Green Vortex; exact solution of incompressible NS
       {
           rho =  1.0;
           p   =  100 + (rho/16.0)*(cos(2.0*x(0)) + cos(2.0*x(1)))*(cos(2.0*x(2) + 2));
           u1  =      sin(x(0))*cos(x(1))*cos(x(2));
           u2  = -1.0*cos(x(0))*sin(x(1))*cos(x(2));
           u3  =  0.0;
       }

       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = rho * u3;                //rho * v
       v(4) = p/(gamm - 1) + 0.5*rho*v_sq;
   }

}


