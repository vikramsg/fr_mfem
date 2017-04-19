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
const double   mu  = 0.05;
const double   Pr  = 0.72;

// Velocity coefficient
void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form is M du/dt = K u + b, where M and K are the mass
    and operator matrices, and b describes the face correction terms. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &K_inv_x, &K_inv_y;//, &K_vis_x, &K_vis_y;
   const Vector &b;
   HypreSmoother M_prec;
   CGSolver M_solver;

   mutable Vector z;


public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, const Vector &_b);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};



int main(int argc, char *argv[])
{
    
   // 1. Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);


   const char *mesh_file = "periodic-square.mesh";
   int    order      = 1;
   double t_final    = 0.0100;
   double dt         = 0.0100;
   int    vis_steps  = 100;
   int    ref_levels = 0;

          problem    = 0;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int     dim = mesh->Dimension();
   int var_dim = dim + 2;


   // 6. Define the parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   for (int lev = 0; lev < ref_levels; lev++)
   {
      pmesh->UniformRefinement();
   }


   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   ParFiniteElementSpace *fes = new ParFiniteElementSpace(pmesh, &fec, var_dim);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);

   ParFiniteElementSpace *fes_op = new ParFiniteElementSpace(pmesh, &fec);
   
   ParBilinearForm *m = new ParBilinearForm(fes_op);
   m->AddDomainIntegrator(new MassIntegrator);
   m->Assemble();
   m->Finalize();

   ParBilinearForm *k_inv_x      = new ParBilinearForm(fes_op);
   k_inv_x->AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));

   int skip_zeros = 1;
   k_inv_x->Assemble(skip_zeros);
   k_inv_x->Finalize(skip_zeros);
   //////////////////////////////////////////////////////////// 
   ParLinearForm *b = new ParLinearForm(fes);
//   b->AddFaceIntegrator(
//      new DGEulerIntegrator(u_vec, f_vec, var_dim, -1.0));
   b->Assemble();
   /////////////////////////////////////////////////////////////
   //Parallel matrices need to be created
   
   HypreParMatrix *K = k_inv_x->ParallelAssemble();
   HypreParMatrix *M = m->ParallelAssemble();
   HypreParVector *B = b->ParallelAssemble();
   /////////////////////////////////////////////////////////////
   
   VectorFunctionCoefficient u0(var_dim, init_function);
   ParGridFunction *u_sol = new ParGridFunction(fes);
   u_sol->ProjectCoefficient(u0);
   HypreParVector *U = u_sol->GetTrueDofs();

   ParFiniteElementSpace *fes_vec = new ParFiniteElementSpace(pmesh, &fec, dim*var_dim);
   ParGridFunction *f_inv = new ParGridFunction(fes_vec);
   getInvFlux(dim, *u_sol, *f_inv);

   FE_Evolution adv(*M, *K, *K, *B);
   ODESolver *ode_solver = new ForwardEulerSolver; 
//   ODESolver *ode_solver = new RK3SSPSolver; 

   double t = 0.0;
   adv.SetTime(t);
   ode_solver->Init(adv);

   bool done = false;
   for (int ti = 0; !done; )
   {
//      b.Assemble();

      double dt_real = min(dt, t_final - t);
      ode_solver->Step(*U, t, dt_real);
      ti++;

      if (myid == 0)
      {
          cout << "time step: " << ti << ", time: " << t << ", max_sol: " << u_sol->Max() << endl;
      }

      getInvFlux(dim, *u_sol, *f_inv); // To update f_vec

      done = (t >= t_final - 1e-8*dt);
//
//      if (done || ti % vis_steps == 0)
//      {
//          getFields(u_sol, rho, u1, u2, E);
//
//          dc->SetCycle(ti);
//          dc->SetTime(t);
//          dc->Save();
//      }
   }
 

   ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, &fec, dim);
   ParGridFunction *x_ref = new ParGridFunction(fespace);
   pmesh->GetNodes(*x_ref);

   int offset = x_ref->Size()/dim;
   for (int i = 0; i < offset; i++)
   {
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
       if (myid == 0)
       {
//           cout << i << '\t' << x_ref[0](sub1) << '\t' << x_ref[0](sub2) << '\t' <<  u_sol[0](sub1) << '\t' << endl;  
//           cout << i << '\t' << x_ref[0](sub1) << '\t' << x_ref[0](sub2) << '\t' <<  u_sol[0](sub1) << '\t' << f_inv[0](sub4) << endl; 
       }
   }


   delete pmesh;
   delete fes;
   delete fes_op;
   delete fes_vec;
   delete M;
   delete K;
   delete B;
   delete k_inv_x;
   delete u_sol;
   delete f_inv;
   delete U;


   MPI_Finalize();
   return 0;
}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, const Vector &_b)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), b(_b), z(_b.Size())
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}



void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    int offset  = K_inv_x.GetNumRows();

    int dim = x.Size()/offset - 2; //FIXME Need better way of defining dim 
    int var_dim = dim + 2;

    Vector y_temp;
    y_temp.SetSize(x.Size()); 

    Vector f(dim*x.Size());
    getInvFlux(dim, x, f);

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

    Vector f_sol(offset), f_x(offset), f_x_m(offset);
    Vector b_sub(offset);
    y = 0.0;
    for(int i = 0; i < var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
//        b.GetSubVector(offsets[i], b_sub);
//        f_x += b_sub; // Needs to be added only once
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i], f_x_m);
    }
    y += y_temp;

//    for(int i = var_dim + 0; i < 2*var_dim; i++)
//    {
//        f.GetSubVector(offsets[i], f_sol);
//        K_inv_y.Mult(f_sol, f_x);
//        M_solver.Mult(f_x, f_x_m);
//        y_temp.SetSubVector(offsets[i - var_dim], f_x_m);
//    }
//    y += y_temp;
   
    for (int j = 0; j < offset; j++) cout << j << '\t' << x(j) << '\t'<< y(0*offset + j) << endl;

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
           rho = 1 + 0.2*sin(M_PI*(x(0) + x(1)));
           u1  = 1.0; u2 =-0.5;
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


void getFields(const GridFunction &u_sol, GridFunction &rho, GridFunction &u1, GridFunction &u2, GridFunction &E)
{

    int vDim  = u_sol.VectorDim();
    int dofs  = u_sol.Size()/vDim;

    for (int i = 0; i < dofs; i++)
    {
        rho[i] = u_sol[         i];        
        u1 [i] = u_sol[  dofs + i]/rho[i];        
        u2 [i] = u_sol[2*dofs + i]/rho[i];        
        E  [i] = u_sol[3*dofs + i];        
    }
}
