#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.010;
const double R_gas = 287;
const double   Pr  = 0.72;

//Additional variables
bool viscous       = true ; // True if mu is not zero

//Run parameters
const char *mesh_file        =  "periodic-cube.mesh";
const int    order           =  3;
const double t_final         =  0.01000;
const int    problem         =  1;
const int    ref_levels      =  3;

const bool   time_adapt      =  false;
const double cfl             =  0.20;
const double dt_const        =  0.0001  ;
const int    ode_solver_type =  2; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       =  100 ;

const bool   adapt           =  false; 
const int    adapt_iter      =  40   ; // Time steps after which adaptation is done
const double tolerance       =  1e-4 ; // Tolerance for adaptation


// Velocity coefficient
void init_function(const Vector &x, Vector &v);
double getUMax(int dim, const Vector &u);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z, 
        const CGSolver &M_solver, const Vector &u, 
        const Vector &b_aux_x, const Vector &b_aux_y, const Vector &b_aux_z, Vector &aux_grad);

void getAuxVar(int dim, const Vector &u, Vector &aux_sol);

void getFields(const GridFunction &u_sol, Vector &rho, Vector &M, Vector &p);
void postProcess(ParMesh &mesh, ParGridFunction &u_sol, 
                int cycle, double time);

void testCompute(ParFiniteElementSpace &fes, const ParGridFunction &uD,
        VectorGridFunctionCoefficient &u_coeff);


/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form is M du/dt = K u + b, where M and K are the mass
    and operator matrices, and b describes the face correction terms. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &K_inv_x, &K_inv_y, &K_inv_z;
   ParLinearForm &b, &b_aux_x, &b_aux_y, &b_aux_z;
   ParGridFunction &u_sol, &f_inv, &aux_sol, &aux_grad, &f_vis;
   HypreSmoother M_prec;
   CGSolver M_solver;

public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            ParLinearForm &_b, ParLinearForm &_b_aux_x, ParLinearForm &_b_aux_y, ParLinearForm &_b_aux_z,
                            ParGridFunction &_u_sol, ParGridFunction &_f_inv, ParGridFunction &_aux_sol, 
                            ParGridFunction &_aux_grad, ParGridFunction &_f_vis);

   void GetSize() ;

   CGSolver &GetMSolver() ;

   void Update();

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

class CNS 
{
private:
    int num_procs, myid;

    ParMesh *pmesh;

    ParFiniteElementSpace *fes, *fes_vec, *fes_op;
    ParFiniteElementSpace *fes_aux_grad;
   
    ParBilinearForm *m, *k_inv_x, *k_inv_y, *k_inv_z;

    ODESolver *ode_solver; 
    FE_Evolution *adv; 

    int dim;

    ParGridFunction u_sol, f_inv;   
    ParGridFunction aux_sol, aux_grad;
    ParGridFunction f_vis;
    ParLinearForm *b;
    ParLinearForm *b_aux_x;
    ParLinearForm *b_aux_y;
    ParLinearForm *b_aux_z;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, t, glob_u_max;
    int ti;
public:
   CNS();

   void Step();
   void Update(Array<int> &newEleOrder);

   void UpdateElementOrder(Vector &error, double tolerance, Array<int> &newEleOrder);

   ~CNS(); 
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

   if (abs(mu) < 1E-12)
       viscous = false;

   CNS run;

   MPI_Finalize();

   return 0;
}

CNS::CNS() 
{
   int num_procs, myid;
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
   fes    = new ParFiniteElementSpace(pmesh, &fec, var_dim);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   fes_op = new ParFiniteElementSpace(pmesh, &fec);

   VectorFunctionCoefficient u0(var_dim, init_function);
   u_sol.SetSpace(fes);
   u_sol.ProjectCoefficient(u0);

   fes_vec = new ParFiniteElementSpace(pmesh, &fec, dim*var_dim);
   f_inv.SetSpace(fes_vec);
   getInvFlux(dim, u_sol, f_inv);

   ParFiniteElementSpace fes_aux(pmesh, &fec, aux_dim);
   fes_aux_grad = new ParFiniteElementSpace(pmesh, &fec, dim*aux_dim);

   aux_grad.SetSpace(fes_aux_grad);
   f_vis.SetSpace(fes_vec);

   aux_sol.SetSpace(&fes_aux);
   getAuxVar(dim, u_sol, aux_sol);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector xdir(dim), ydir(dim), zdir(dim);
   xdir = 0.0; xdir(0) = 1.0;
   VectorConstantCoefficient x_dir(xdir);
   ydir = 0.0; ydir(1) = 1.0;
   VectorConstantCoefficient y_dir(ydir);
       
   VectorConstantCoefficient z_dir(ydir);
   if (dim == 3)
   {
       zdir = 0.0; zdir(2) = 1.0;
       VectorConstantCoefficient z_dir_(zdir);
       z_dir = z_dir_; // FIXME: Check whether this is correct
   }

   m = new ParBilinearForm(fes_op);
   m->AddDomainIntegrator(new MassIntegrator);
   k_inv_x = new ParBilinearForm(fes_op);
   k_inv_x->AddDomainIntegrator(new ConvectionIntegrator(x_dir,  1.0));
   k_inv_y = new ParBilinearForm(fes_op);
   k_inv_y->AddDomainIntegrator(new ConvectionIntegrator(y_dir,  1.0));
   if (dim == 3)
   {
       k_inv_z = new ParBilinearForm(fes_op);
       k_inv_z->AddDomainIntegrator(new ConvectionIntegrator(z_dir,  1.0));
   }

   m->Assemble();
   m->Finalize();
   int skip_zeros = 1;
   k_inv_x->Assemble(skip_zeros);
   k_inv_x->Finalize(skip_zeros);
   k_inv_y->Assemble(skip_zeros);
   k_inv_y->Finalize(skip_zeros);
   if (dim == 3)
   {
       k_inv_z->Assemble(skip_zeros);
       k_inv_z->Finalize(skip_zeros);
   }

   /////////////////////////////////////////////////////////////

   VectorGridFunctionCoefficient u_vec(&u_sol);
   VectorGridFunctionCoefficient f_vec(&f_inv);

   VectorGridFunctionCoefficient aux_vec(&aux_sol);
   VectorGridFunctionCoefficient f_vis_vec(&f_vis);
   VectorGridFunctionCoefficient aux_grad_vec(&aux_grad);

   u_sol.ExchangeFaceNbrData(); //Exchange data across processors
   f_inv.ExchangeFaceNbrData(); 

   aux_sol.ExchangeFaceNbrData(); 

   aux_grad.ExchangeFaceNbrData(); 
   f_vis.ExchangeFaceNbrData(); 

   
   /////////////////////////////////////////////////////////////
   // Linear form
   b = new ParLinearForm(fes);
   b->AddFaceIntegrator(
//      new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, 
      new DGEulerIntegrator(R_gas, gamm, u_sol, f_inv, u_sol.FaceNbrData(),  f_inv.FaceNbrData(),
          - 1.0));
   b->AddFaceIntegrator(
//      new DG_Viscous_Integrator(R_gas, gamm, mu, Pr, u_vec, f_vis_vec, aux_grad_vec,  1.0));
      new DG_Viscous_Integrator(R_gas, gamm, mu, Pr, 
                                u_sol, f_vis, aux_grad,  
                                u_sol.FaceNbrData(), f_vis.FaceNbrData(), aux_grad.FaceNbrData(),  
                                1.0));
   ///////////////////////////////////////////////////////////
   b_aux_x = new ParLinearForm(&fes_aux);
   b_aux_y = new ParLinearForm(&fes_aux);

   if (viscous)
   {
       b_aux_x->AddFaceIntegrator(
//          new DG_Viscous_Aux_Integrator(x_dir, aux_vec, 1.0));
          new DG_Viscous_Aux_Integrator(xdir, aux_sol, aux_sol.FaceNbrData(), 1.0));
       b_aux_x->Assemble();

       b_aux_y->AddFaceIntegrator(
//          new DG_Viscous_Aux_Integrator(y_dir, aux_vec, 1.0));
          new DG_Viscous_Aux_Integrator(ydir, aux_sol, aux_sol.FaceNbrData(), 1.0));
       b_aux_y->Assemble();

       if (dim == 3)
       {
           b_aux_z = new ParLinearForm(&fes_aux);
           b_aux_z->AddFaceIntegrator(
//          new DG_Viscous_Aux_Integrator(z_dir, aux_vec, 1.0));
               new DG_Viscous_Aux_Integrator(zdir, aux_sol, aux_sol.FaceNbrData(), 1.0));
           b_aux_z->Assemble();
       }
   }

   ///////////////////////////////////////////////////////////////
   HypreParMatrix *M       = m->ParallelAssemble();
   HypreParMatrix *K_inv_x = k_inv_x->ParallelAssemble();
   HypreParMatrix *K_inv_y = k_inv_y->ParallelAssemble();
   HypreParMatrix *K_inv_z;

   if (dim == 3)
   {
       K_inv_z = k_inv_z->ParallelAssemble();
   }

   delete m;
   delete k_inv_x, k_inv_y; // Cleanup memory
   if (dim == 3)
   {
       delete k_inv_z; 
   }

   adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, *K_inv_z, 
           *b, *b_aux_x, *b_aux_y, *b_aux_z,
           u_sol, f_inv, aux_sol, aux_grad, f_vis);
 
   if (viscous)
   {
        getAuxVar(dim, u_sol, aux_sol);
        aux_sol.ExchangeFaceNbrData(); 
        b_aux_x->Assemble();
        b_aux_y->Assemble();
    
        getAuxGrad(dim, *K_inv_x, *K_inv_y, *K_inv_z, adv->GetMSolver(), u_sol, *b_aux_x, *b_aux_y, *b_aux_z, aux_grad);
        getVisFlux(dim, u_sol, aux_grad, f_vis);
        aux_grad.ExchangeFaceNbrData(); 
        f_vis.ExchangeFaceNbrData(); 
   }
   /////////////////////////////////////////////////////////////

   t = 0.0; ti = 0; // Initialize time and time iterations
   adv->SetTime(t);

   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK3SSPSolver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
   }

   ode_solver->Init(*adv);

   ///////////////////////////////////////////////////////////////
   //Calculate time step
   MPI_Comm comm = pmesh->GetComm();

   double loc_u_max = getUMax(dim, u_sol);     // Get u_max on local processor
   MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

   if (time_adapt)
   {
       dt = cfl*((h_min/(2.0*order + 1))/glob_u_max); 
   }
   else
   {
       dt = dt_const; 
   }
   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   bool done = false;
   for (ti = 0; !done; )
   {
      Step(); // Step in time

      done = (t >= t_final - 1e-8*dt);
    
      if (ti % 10 == 0) // Check time
      {
          chrono.Stop();
          if (myid == 0)
          {
              cout << "10 Steps took "<< chrono.RealTime() << " s "<< endl;
          }

          chrono.Clear();
          chrono.Start();
      }
      
      if (done || ti % vis_steps == 0) // Visualize
      {
          postProcess(*pmesh, u_sol, ti, t);
      }
   }


//   // Print all nodes in the finite element space 
//   FiniteElementSpace fes_nodes(mesh, vfec, dim);
//   GridFunction nodes(&fes_nodes);
//   mesh->GetNodes(nodes);
//
//   for (int i = 0; i < nodes.Size()/dim; i++)
//   {
//       int offset = nodes.Size()/dim;
//       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
////       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << f_inv(sub1)<< endl;      
////       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << rhs(sub2)<< endl;      
////       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << u_out(sub1) << endl;      
//   }
//

   delete adv;
   delete M, K_inv_x, K_inv_y, K_inv_z;
}


CNS::~CNS()
{
    delete b;
    delete b_aux_x, b_aux_y, b_aux_z;
    delete ode_solver;
    delete pmesh;
    delete fes;
    delete fes_vec;
    delete fes_op;
    delete fes_aux_grad;
}

void CNS::Step()
{
      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u_sol, t, dt_real);
      ti++;

      MPI_Comm comm = pmesh->GetComm();

      double loc_u_max = getUMax(dim, u_sol);     // Get u_max on local processor
      MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

      MPI_Comm_rank(MPI_COMM_WORLD, &myid);
      if (myid == 0)
      {
          cout << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
                t << ", max_speed " << glob_u_max << ", fes_size " << fes->GlobalTrueVSize() << endl;
      }

      if (time_adapt)
      {
          dt = cfl*((h_min/(2.0*order + 1))/glob_u_max); 
      }
      else
      {
          dt = dt_const; 
      }

}





// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            ParLinearForm &_b, ParLinearForm &_b_aux_x, ParLinearForm &_b_aux_y, ParLinearForm &_b_aux_z,
                            ParGridFunction &_u_sol, ParGridFunction &_f_inv, ParGridFunction &_aux_sol, 
                            ParGridFunction &_aux_grad, ParGridFunction &_f_vis)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), K_inv_z(_K_inv_z), 
                            b(_b), b_aux_x(_b_aux_x), b_aux_y(_b_aux_y), b_aux_z(_b_aux_z),
                            u_sol(_u_sol), f_inv(_f_inv), aux_sol(_aux_sol), aux_grad(_aux_grad), f_vis(_f_vis)
{
   M_prec.SetType(HypreSmoother::Jacobi); 
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);
}



void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    int dim = x.Size()/K_inv_x.GetNumRows() - 2;
    int var_dim = dim + 2;

    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector y_temp;
    y_temp.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    u_sol.ExchangeFaceNbrData(); 
    getInvFlux(dim, x, f_inv);

    if (viscous)
    {
        getAuxVar(dim, x, aux_sol);
        aux_sol.ExchangeFaceNbrData(); 
        b_aux_x.Assemble();
        b_aux_y.Assemble();
        if (dim == 3)
        {
            b_aux_z.Assemble();
        }
    
        getAuxGrad(dim, K_inv_x, K_inv_y, K_inv_z, M_solver, x, b_aux_x, b_aux_y, b_aux_z, aux_grad);
        getVisFlux(dim, x, aux_grad, f_vis);
        aux_grad.ExchangeFaceNbrData(); 
        f_vis.ExchangeFaceNbrData(); 
    }
    else
    {
        f_vis = 0.0;    
    }

    b.Assemble();

    subtract(f_vis, f_inv, f_inv);

    int offset  = K_inv_x.GetNumRows();
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
        f_inv.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i], f_x);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f_inv.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i - var_dim], f_x);
    }
    y += y_temp;

    if (dim == 3)
    {
       for(int i = 2*var_dim + 0; i < 3*var_dim; i++)
       {
           f_inv.GetSubVector(offsets[i], f_sol);
           K_inv_z.Mult(f_sol, f_x);
           y_temp.SetSubVector(offsets[i - 2*var_dim], f_x);
       }
       y += y_temp;
    }

    add(b, y, y);
    
    for(int i = 0; i < var_dim; i++)
    {
        y.GetSubVector(offsets[i], f_x);
        M_solver.Mult(f_x, f_x_m); // M^-1.(K(u))
        y.SetSubVector(offsets[i], f_x_m);
    }
}


CGSolver &FE_Evolution::GetMSolver() 
{
    return M_solver;
}


// Get max signal speed 
double getUMax(int dim, const Vector &u)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}

    int offset = u.Size()/var_dim;

    double u_max = 0;

    double rho, vel[dim];
    for(int j = 0; j < offset; j++)
    {
        rho = u[j];
    
        for(int i = 0; i < dim; i++) vel[i] = u[i*offset + j]/rho;

        double v_sq =  0;
        for(int i = 0; i < dim; i++)
        {
            v_sq += pow(vel[i], 2);
        }
        double rho_E = u[(var_dim - 1)*offset + j];

        double e  = (rho_E - 0.5*rho*v_sq)/rho;
        double Cv = R_gas/(gamm - 1);

        double T = e/Cv; // T

        double a = sqrt(gamm*R_gas*T);

        u_max = max(u_max, sqrt(v_sq) + a);
    }
    return u_max;
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
    Vector rho, E;
    u.GetSubVector(offsets[0],           rho   );
    u.GetSubVector(offsets[var_dim - 1],      E);

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



// Get gradient of auxilliary variable 
void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z,
        const CGSolver &M_solver, const Vector &u, 
        const Vector &b_aux_x, const Vector &b_aux_y, const Vector &b_aux_z,
        Vector &aux_grad)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}
    int offset = u.Size()/var_dim;

    Vector aux_sol(aux_dim*offset);

    getAuxVar(dim, u, aux_sol);
        
    Array<int> offsets[dim*aux_dim];
    for(int i = 0; i < dim*aux_dim; i++)
    {
        offsets[i].SetSize(offset);
    }
    for(int j = 0; j < dim*aux_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    aux_grad = 0.0;

    Vector aux_var(offset), aux_x(offset), b_sub(offset);
    for(int i = 0; i < aux_dim; i++)
    {
        aux_sol.GetSubVector(offsets[i], aux_var);

        K_x.Mult(aux_var, aux_x);
        b_aux_x.GetSubVector(offsets[i], b_sub);
        add(b_sub, aux_x, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[          i], aux_x);

        K_y.Mult(aux_var, aux_x);
        b_aux_y.GetSubVector(offsets[i], b_sub);
        add(b_sub, aux_x, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[aux_dim + i], aux_x);

        if (dim == 3)
        {
            K_z.Mult(aux_var, aux_x);
            b_aux_z.GetSubVector(offsets[i], b_sub);
            add(b_sub, aux_x, aux_x);
            M_solver.Mult(aux_x, aux_x);
            aux_grad.SetSubVector(offsets[2*aux_dim + i], aux_x);
        }
    }
}



// Get auxilliary variables
void getAuxVar(int dim, const Vector &u, Vector &aux_sol)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}

    int offset = u.Size()/var_dim;

    Array<int> offsets[var_dim];
    for(int i = 0; i < var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }
    for(int j = 0; j < var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = j*offset + i ;
        }
    }

    Vector rho(offset), u_sol(offset), rho_E(offset);
    u.GetSubVector(offsets[0], rho);
    for(int i = 0; i < dim; i++)
    {
        u.GetSubVector(offsets[1 + i], u_sol); // u, v, w
        for(int j = 0; j < offset; j++)
        {
            aux_sol[i*offset + j] = u_sol[j]/rho[j];
        }
    }
    u.GetSubVector(offsets[var_dim - 1], rho_E); // rho*E
    for(int j = 0; j < offset; j++)
    {
        double v_sq =  0;
        for(int i = 0; i < dim; i++)
        {
            v_sq += pow(aux_sol[i*offset + j], 2);
        }
        double e  = (rho_E(j) - 0.5*rho[j]*v_sq)/rho[j];
        double Cv = R_gas/(gamm - 1);

        aux_sol[(aux_dim - 1)*offset + j] = e/Cv; // T
    }
}


// Aux flux 
void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f)
{
    int var_dim = dim + 2;
    int aux_dim = dim + 1;
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

    Vector rho(offset), rho_u1(offset), rho_u2(offset), E(offset);
    u.GetSubVector(offsets[0],           rho   );
    u.GetSubVector(offsets[var_dim - 1],      E);

    Vector rho_vel[dim];
    for(int i = 0; i < dim; i++) u.GetSubVector(offsets[1 + i], rho_vel[i]);

    for(int i = 0; i < offset; i++)
    {
        double vel[dim];        
        for(int j = 0; j < dim; j++) vel[j]   = rho_vel[j](i)/rho(i);

        double vel_grad[dim][dim];
        for (int k = 0; k < dim; k++)
            for (int j = 0; j < dim; j++)
            {
                vel_grad[j][k]      =  aux_grad[k*(aux_dim)*offset + j*offset + i];
            }

        double divergence = 0.0;            
        for (int k = 0; k < dim; k++) divergence += vel_grad[k][k];

        double tau[dim][dim];
        for (int j = 0; j < dim; j++) 
            for (int k = 0; k < dim; k++) 
                tau[j][k] = mu*(vel_grad[j][k] + vel_grad[k][j]);

        for (int j = 0; j < dim; j++) tau[j][j] -= 2.0*mu*divergence/3.0; 

        double int_en_grad[dim];
        for (int j = 0; j < dim; j++)
        {
            int_en_grad[j] = (R_gas/(gamm - 1))*aux_grad[j*(aux_dim)*offset + (aux_dim - 1)*offset + i] ; // Cv*T_x
        }

        for (int j = 0; j < dim ; j++)
        {
            f(j*var_dim*offset + i)       = 0.0;

            for (int k = 0; k < dim ; k++)
            {
                f(j*var_dim*offset + (k + 1)*offset + i)       = tau[j][k];
            }
            f(j*var_dim*offset + (var_dim - 1)*offset + i)     =  (mu/Pr)*gamm*int_en_grad[j]; 
            for (int k = 0; k < dim ; k++)
            {
                f(j*var_dim*offset + (var_dim - 1)*offset + i)+= vel[k]*tau[j][k]; 
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

       if (problem == 0) // Isentropic vortex; exact solution for Euler
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
       else if (problem == 1) // Taylor Green Vortex; exact solution of incompressible NS
       {
           rho =  1.0;
           p   =  100 + (rho/4.0)*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)));
           u1  =      sin(M_PI*x(0))*cos(M_PI*x(1));
           u2  = -1.0*cos(M_PI*x(0))*sin(M_PI*x(1));
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
           p   =  100 + (rho/16.0)*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)))*(cos(2.0*M_PI*x(2) + 2));
           u1  =      sin(M_PI*x(0))*cos(M_PI*x(1))*cos(M_PI*x(2));
           u2  = -1.0*cos(M_PI*x(0))*sin(M_PI*x(1))*cos(M_PI*x(2));
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

//   Exact solution of problem 
double exactSol(const Vector &x, double t)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       if (problem == 0) // Isentropic vortex; exact solution for Euler
       {
           double rho, u1, u2, p, T, M;
           double R, beta, omega, f, du, dv, dT;
    
           beta = 5;

           double x0 = t;
           if (x0 > 5) x0 = x0 - 10; // Shift it periodically
    
           f     = (pow((x(0) - x0), 2) + pow(x(1), 2));
           omega = (beta/(2*M_PI))*exp(0.5*(1 - f));
           du    = -x(1)*omega;
           dv    =  (x(0))*omega;
           dT    = - (gamm - 1)*beta*beta/(8*gamm*M_PI*M_PI) * exp(1 - f);
    
           u1    = 1 + du;
           u2    = dv;
           T     = 1 + dT;
           rho   = pow(T, 1/(gamm - 1));
    
           return rho;
       }
       else if (problem == 1) // Taylor Green Vortex; exact solution of incompressible NS
       {
           double rho =  1.0;
           double fac =  pow((exp(-2*M_PI*M_PI*mu*t)), 2);
           double p   =  100 + (rho/4.0)*(cos(2.0*M_PI*x(0)) + cos(2.0*M_PI*x(1)))*fac;

           return p;
       }
   }

}


void getFields(const GridFunction &u_sol, Vector &rho, Vector &M, Vector &p)
{
    int vDim    = u_sol.VectorDim();
    int  dim    = vDim - 2;
    int dofs    = u_sol.Size()/vDim;

    int aux_dim = vDim - 1;

    for (int i = 0; i < dofs; i++)
    {
        rho[i]   = u_sol[         i];        
        double vel[dim]; 
        for (int j = 0; j < dim; j++)
        {
            vel[j] =  u_sol[(1 + j)*dofs + i]/rho[i];        
        }
        double E  = u_sol[(vDim - 1)*dofs + i];        

        double v_sq = 0.0;    
        for (int j = 0; j < dim; j++)
        {
            v_sq += pow(vel[j], 2); 
        }

        p[i]     = (E - 0.5*rho[i]*v_sq)*(gamm - 1);

        M[i]     = sqrt(v_sq)/sqrt(gamm*p[i]/rho[i]);

    }
}


void postProcess(ParMesh &mesh, ParGridFunction &u_sol, 
                int cycle, double time)
{
   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   DG_FECollection fec(order, dim);

   VisItDataCollection dc("CNS", &mesh);
   dc.SetPrecision(8);
 
   ParFiniteElementSpace fes_fields(&mesh, &fec);
   ParGridFunction rho(&fes_fields);
   ParGridFunction M(&fes_fields);
   ParGridFunction p(&fes_fields);

   dc.RegisterField("rho", &rho);
   dc.RegisterField("M", &M);
   dc.RegisterField("p", &p);

   getFields(u_sol, rho, M, p);

   dc.SetCycle(cycle);
   dc.SetTime(time);
   dc.Save();
  
}
