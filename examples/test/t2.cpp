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

   double m_fac = 1.0;

   Poly_1D poly1, poly2;
   int p1_basis = order, p2_basis = order, type = 0;
   
   int Np1 = p1_basis + 1, Np2 = p2_basis + 1;
   Poly_1D::Basis &basis1(poly1.OpenBasis(p1_basis, type));
   Poly_1D::Basis &basis2(poly2.OpenBasis(p2_basis, type));

   const double *pts1 = poly1.GetPoints(p1_basis, type);
   Vector P; P.SetSize(Np1);
   DenseMatrix van1(Np1);
   for (int i = 0; i < Np1; i++)
   {
       poly1.CalcLegendreBasis(p1_basis, pts1[i], P);
       van1.SetRow(i, P);
   
       for (int j = 0; j < Np1; j++)
       {
           double p_gamma = 2.0/(2.0*j + 1);
           van1(i, j)   = van1(i, j)/sqrt(p_gamma);
       }
   }
   const double *pts2 = poly2.GetPoints(p2_basis, type);
   P.SetSize(Np2);
   DenseMatrix van2(Np2);
   for (int i = 0; i < Np2; i++)
   {
       poly2.CalcLegendreBasis(p2_basis, pts2[i], P);
       van2.SetRow(i, P);
   
       for (int j = 0; j < Np2; j++)
       {
           double p_gamma = 2.0/(2.0*j + 1);
           van2(i, j)   = van2(i, j)/sqrt(p_gamma);
       }
   }

   van1.Transpose(); van1.Invert();
   van2.Transpose(); van2.Invert();

   Vector mod_lag1_R, mod_lag1_L;
   mod_lag1_R.SetSize(Np1); mod_lag1_L.SetSize(Np1);
   P.SetSize(Np1);
   poly1.CalcLegendreBasis(p1_basis, 0.0, P);
   for (int j = 0; j < Np1; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np1 - 1] *= m_fac;
   van1.Mult(P, mod_lag1_L);
   poly1.CalcLegendreBasis(p1_basis, 1.0, P);
   for (int j = 0; j < Np1; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np1 - 1] *= m_fac;
   van1.Mult(P, mod_lag1_R);

   Vector mod_lag2_R, mod_lag2_L;
   mod_lag2_R.SetSize(Np2); mod_lag2_L.SetSize(Np2);
   P.SetSize(Np2);
   poly2.CalcLegendreBasis(p2_basis, 0.0, P);
   for (int j = 0; j < Np2; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np2 - 1] *= m_fac;
   van2.Mult(P, mod_lag2_L);
   poly2.CalcLegendreBasis(p2_basis, 1.0, P);
   for (int j = 0; j < Np2; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np2 - 1] *= m_fac;
   van2.Mult(P, mod_lag2_R);


   FaceElementTransformations *Trans; 
   const FiniteElement *fe1, *fe2;
   Array<int> vdofs1, vdofs2;
   int ndof1, ndof2;
   Vector shape1, shape2;
   Vector mod_shape1, mod_shape2;
   Vector shape_x, shape_y, shape_z;

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
    
              shape_x.SetSize(Np1); shape_y.SetSize(Np1);
              basis1.Eval(eip1.x, shape_x);
              basis1.Eval(eip1.y, shape_y);
        
              if (abs(eip1.x) < tol)
              {
                  shape_x = mod_lag1_L;              
              }
              else if ( 1 - abs(eip1.x) < tol)
              {
                  shape_x = mod_lag1_R;              
              }
              if (abs(eip1.y) < tol)
              {
                  shape_y = mod_lag1_L;              
              }
              else if ( 1 - abs(eip1.y) < tol)
              {
                  shape_y = mod_lag1_R;              
              }
        
              if (dim == 2)
              {
                  mod_shape1 = 0.0;
                  for (int ot = 0, jt = 0; jt <= p1_basis; jt++)
                      for (int it = 0; it <= p1_basis; it++)
                      {
                          mod_shape1(ot++) = shape_x(it)*shape_y(jt);
                      }
              }
              else if (dim == 3)
              {
                  shape_z.SetSize(Np1); 
                  basis1.Eval(eip1.z, shape_z);
                  if (abs(eip1.z) < tol)
                  {
                      shape_z = mod_lag1_L;              
                  }
                  else if ( 1 - abs(eip1.z) < tol)
                  {
                      shape_z = mod_lag1_R;              
                  }
                  for (int ot = 0, kt = 0; kt <= p1_basis; kt++)
                      for (int jt = 0; jt <= p1_basis; jt++)
                          for (int it = 0; it <= p1_basis; it++)
                          {
                              mod_shape1(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                          }
              }
              
              shape_x.SetSize(Np2); shape_y.SetSize(Np2);
              basis2.Eval(eip2.x, shape_x);
              basis2.Eval(eip2.y, shape_y);
        
              if (abs(eip2.x) < tol)
              {
                  shape_x = mod_lag2_L;              
              }
              else if ( 1 - abs(eip2.x) < tol)
              {
                  shape_x = mod_lag2_R;              
              }
              if (abs(eip2.y) < tol)
              {
                  shape_y = mod_lag2_L;              
              }
              else if ( 1 - abs(eip2.y) < tol)
              {
                  shape_y = mod_lag2_R;              
              }
        
              if (dim == 2)
              {
                  mod_shape2 = 0.0;
                  for (int ot = 0, jt = 0; jt <= p2_basis; jt++)
                      for (int it = 0; it <= p2_basis; it++)
                      {
                          mod_shape2(ot++) = shape_x(it)*shape_y(jt);
                      }
              }
              else if (dim == 3)
              {
                  shape_z.SetSize(Np2); 
                  basis2.Eval(eip2.z, shape_z);
                  if (abs(eip2.z) < tol)
                  {
                      shape_z = mod_lag2_L;              
                  }
                  else if ( 1 - abs(eip2.z) < tol)
                  {
                      shape_z = mod_lag2_R;              
                  }
                  for (int ot = 0, kt = 0; kt <= p2_basis; kt++)
                      for (int jt = 0; jt <= p2_basis; jt++)
                          for (int it = 0; it <= p2_basis; it++)
                          {
                              mod_shape2(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                          }
              }

//              cout << shape2[0] << "\t" << shape2[1] << "\t" << shape2[2] << endl;
//              cout << mod_shape2[0] << "\t" << mod_shape2[1] << "\t" << mod_shape2[2] << endl;

              cout << shape1[0] << "\t" << shape1[1] << "\t" << shape1[2] << endl;
              cout << mod_shape1[0] << "\t" << mod_shape1[1] << "\t" << mod_shape1[2] << endl;

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


