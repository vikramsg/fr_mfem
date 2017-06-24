#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Problem to solve
int problem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.00;
const double   Pr  = 0.72;

void df1(const Vector &x, Vector &v);
void df2(const Vector &x, Vector &v);

void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);


int main(int argc, char *argv[])
{
   const char *mesh_file = "char_wall.msh";
   int    order      = 2;
   double t_final    = 0.0075;
   double dt         = 0.0075;
   int    vis_steps  = 100;
   int    ref_levels = 0;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 0, 1);
   int     dim = mesh->Dimension();
   int var_dim = dim + 2;

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   //    Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   FiniteElementSpace fes(mesh, &fec, var_dim);

   VectorFunctionCoefficient u0(var_dim, init_function);
   GridFunction u_sol(&fes);
   u_sol.ProjectCoefficient(u0);

   FiniteElementSpace fes_vec(mesh, &fec, dim*var_dim);
   GridFunction f_inv(&fes_vec);
   getInvFlux(dim, u_sol, f_inv);

   /////////////////////////////////////////////////////////////
   VectorGridFunctionCoefficient u_vec(&u_sol);
   VectorFunctionCoefficient u_char_bnd(var_dim, df1); // Defines characterstic boundary condition

   // Linear form for characteristic boundary
   Array<int> dir_bdr(mesh->bdr_attributes.Max());
   dir_bdr     = 0; // Deactivate all boundaries

   LinearForm b1(&fes);
//   dir_bdr[3]  = 1; // For each boundary activate only that one for sending a function 
   b1.AddBdrFaceIntegrator(
      new DG_CNS_Characteristic_Integrator(
      u_vec, u_char_bnd, var_dim, -1.0), dir_bdr); 
   b1.Assemble();

   /////////////////////////////////////////////////////////////
   VectorFunctionCoefficient u_wall_bnd(dim, df2); // Defines wall boundary condition 
   // Linear form for wall boundary
   dir_bdr     = 0; // Deactivate all boundaries

   LinearForm b2(&fes);
   dir_bdr[1]  = 1; // For each boundary activate only that one for sending a function 
   b2.AddBdrFaceIntegrator(
      new DG_CNS_NoSlipWall_Integrator(
      u_vec, u_wall_bnd, var_dim, -1.0), dir_bdr); 
   b2.Assemble();
   //////////////

   FiniteElementSpace fes_op(mesh, &fec);
   BilinearForm m(&fes_op);
   m.AddDomainIntegrator(new MassIntegrator);

   m.Assemble();
   m.Finalize();
   ////////////
   
   CGSolver M_solver;
   M_solver.SetOperator(m.SpMat());

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);

   int offset = u_sol.Size()/var_dim;
   Array<int> offsets[dim*var_dim];
   for(int i = 0; i < dim*var_dim; i++)
   {
       offsets[i].SetSize(offset);
   }

   for(int j = 0; j < dim*var_dim; j++)
   {
       for(int i = 0; i < offset; i++)
       {
           offsets[j][i] = j*offset + i ;
       }
   }

   Vector y(offset), f_x(offset), f_x_m(offset);
   Vector b_sub(offset), y_temp(var_dim*offset);
   y = 0.0; f_x = 0.0;
   for(int i = 0; i < var_dim; i++)
   {
       b1.GetSubVector(offsets[i], b_sub);
       f_x += b_sub; // Needs to be added only once
       M_solver.Mult(f_x, f_x_m);
       y_temp.SetSubVector(offsets[i], f_x_m);
   }
 

   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, &fec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << b1[sub1] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << y_temp[sub1] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << f_inv(sub4) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << '\t' << u_sol(sub2) << '\t' << u_sol(sub3) << '\t' << u_sol(sub4) << endl;      
   }


   delete mesh;

   return 0;
}



// Inviscid flux 
void getInvFlux(int dim, const Vector &u, Vector &f)
{
    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }
    Vector rho, rho_u1, rho_u2, E;
    u.GetSubVector(offsets[0], rho   );
    u.GetSubVector(offsets[3],      E);

    Vector rho_vel[dim];
    for(int i = 0; i < dim; i++) u.GetSubVector(offsets[1 + i], rho_vel[i]);

    for(int i = 0; i < offset; i++)
    {
        double vel[dim];        
        for(int j = 0; j < dim; j++) vel[j]   = rho_vel[j](i)/rho(i);

        double vel_sq = 0.0;
        for(int j = 0; j < dim; j++) vel_sq += pow(vel[j], 2);

        double pres    = (E(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);

        for(int j = 0; j < dim; j++) 
        {
            f(j*var_dim*offset + i)       = rho_vel[j][i]; //rho*u

            for (int k = 0; k < dim ; k++)
            {
                f(j*var_dim*offset + (k + 1)*offset + i)     = rho_vel[j](i)*vel[k]; //rho*u*u + p    
            }
            f(j*var_dim*offset + (j + 1)*offset + i)        += pres; 

            f(j*var_dim*offset + (var_dim - 1)*offset + i)   = (E(i) + pres)*vel[j] ;//(E+p)*u
        }
    }
}

void df1(const Vector &x, Vector &v)
{
    double rho = 2;
    double u1  = 1;
    double u2  = 0;
    double p   = 1.6;
    
    double v_sq = pow(u1, 2) + pow(u2, 2);

    v(0) = rho;                     //rho
    v(1) = rho * u1;                //rho * u
    v(2) = rho * u2;                //rho * v
    v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
}

void df2(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
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
       if (problem == 0)
       {
//           rho = 1 + 0.2*sin(M_PI*(x(0) + x(1)));
//           u1  = 1.0; u2 =-0.5;
           rho = 1 + 0.2*sin(M_PI*(x(0) ));
           u1  = 1.0; u2 =-0.0;
           p   = 1;
       }
       else if (problem == 1)
       {
           if (x(0) < 0.0)
           {
               rho = 1.0; 
               u1  = 0.0; u2 = 0.0;
               p   = 1;
           }
           else
           {
               rho = 0.125;
               u1  = 0.0; u2 = 0.0;
               p   = 0.1;
           }
       }
       else if (problem == 2) //Taylor Green Vortex
       {
           rho =  1.0;
           p   =  100 + rho/4.0*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)));
           u1  =      sin(M_PI*x(0))*cos(M_PI*x(1))/rho;
           u2  = -1.0*cos(M_PI*x(0))*sin(M_PI*x(1))/rho;
       }

    
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
}


