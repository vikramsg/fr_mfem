#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.1;
const double R_gas = 287;
const double   Pr  = 0.72;

//Run parameters
const char *mesh_file        =  "periodic-cube.mesh";
const int    order           =  2;
const double t_final         =  0.00100;
const int    problem         =  1;
const int    ref_levels      =  1;

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

double getUMax(int dim, const Vector &u);

void getInvFlux(int dim, const Vector &u, Vector &f);
void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z,
        const CGSolver &M_solver, const Vector &u, Vector &aux_grad);
void getAuxVar(int dim, const Vector &u, Vector &aux_sol);

double ComputeTKE(FiniteElementSpace &fes, const Vector &uD);

void getFields(const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &u1, Vector &u2, 
                Vector &E, Vector &u_x, Vector &u_y, Vector &v_x, Vector &v_y);
void postProcess(ParMesh &mesh, ParGridFunction &u_sol, ParGridFunction &aux_grad,
        int cycle, double time);

void getEleOrder(FiniteElementSpace &fes, Array<int> &newEleOrder, GridFunction &order);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form is M du/dt = K u + b, where M and K are the mass
    and operator matrices, and b describes the face correction terms. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   HypreParMatrix &M, &K_inv_x, &K_inv_y, &K_inv_z;
   HypreParMatrix     &K_vis_x, &K_vis_y, &K_vis_z;
   const Vector &b;
   HypreSmoother M_prec;
   CGSolver M_solver;

public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            HypreParMatrix &_K_vis_x, HypreParMatrix &_K_vis_y, HypreParMatrix &_K_vis_z,
                            Vector &_b);

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

    ParMesh *pmesh ;

    ParFiniteElementSpace  *fes, *fes_vec, *fes_op;
   
    ParBilinearForm *m, *k_inv_x, *k_inv_y, *k_inv_z;
    ParBilinearForm     *k_vis_x, *k_vis_y, *k_vis_z;

    ODESolver *ode_solver; 

    int dim;

    ParGridFunction *u_sol, *f_inv;   
    HypreParVector  *U;
    ParLinearForm   *b;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, t, glob_u_max;
    int ti;

    ofstream tke_file;
public:
   CNS();

   void Step();

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

   fes_vec = new ParFiniteElementSpace(pmesh, &fec, dim*var_dim);
       
   f_inv = new ParGridFunction(fes_vec);
   getInvFlux(dim, *u_sol, *f_inv);

   ParFiniteElementSpace fes_aux_grad(pmesh, &fec, dim*aux_dim);
   ParGridFunction aux_grad(&fes_aux_grad);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim); 
   dir = 0.0; dir(0) = 1.0;
   VectorConstantCoefficient x_dir(dir);
   dir = 0.0; dir(1) = 1.0;
   VectorConstantCoefficient y_dir(dir);

   fes_op = new ParFiniteElementSpace(pmesh, &fec);
   m      = new ParBilinearForm(fes_op);
   m->AddDomainIntegrator(new MassIntegrator);
   k_inv_x = new ParBilinearForm(fes_op);
   k_inv_x->AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));
   k_inv_y = new ParBilinearForm(fes_op);
   k_inv_y->AddDomainIntegrator(new ConvectionIntegrator(y_dir, -1.0));

   m->Assemble();
   m->Finalize();
   int skip_zeros = 1;
   k_inv_x->Assemble(skip_zeros);
   k_inv_x->Finalize(skip_zeros);
   k_inv_y->Assemble(skip_zeros);
   k_inv_y->Finalize(skip_zeros);

   /////////////////////////////////////////////////////////////

   VectorGridFunctionCoefficient u_vec(u_sol);
   VectorGridFunctionCoefficient f_vec(f_inv);

   u_sol->ExchangeFaceNbrData(); //Exchange data across processors
   f_inv->ExchangeFaceNbrData();

   /////////////////////////////////////////////////////////////
   // Linear form
   b = new ParLinearForm(fes);
   b->AddFaceIntegrator(
      new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, var_dim, -1.0));
   b->Assemble();
   ///////////////////////////////////////////////////////////

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for viscous terms 
   k_vis_x = new ParBilinearForm(fes_op);
   k_vis_x->AddDomainIntegrator(new ConvectionIntegrator(x_dir,  1.0));
   k_vis_x->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(x_dir, -1.0,  0.0)));// Beta 0 means central flux

   k_vis_y = new ParBilinearForm(fes_op);
   k_vis_y->AddDomainIntegrator(new ConvectionIntegrator(y_dir,  1.0));
   k_vis_y->AddInteriorFaceIntegrator(
      new TransposeIntegrator(new DGTraceIntegrator(y_dir, -1.0,  0.0)));// Beta 0 means central flux

   k_vis_x->Assemble(skip_zeros);
   k_vis_x->Finalize(skip_zeros);
   k_vis_y->Assemble(skip_zeros);
   k_vis_y->Finalize(skip_zeros);

   if (dim == 3)
   {
       dir = 0.0; dir(2) = 1.0;
       VectorConstantCoefficient z_dir(dir);
      
       k_inv_z = new ParBilinearForm(fes_op);
       k_inv_z->AddDomainIntegrator(new ConvectionIntegrator(z_dir, -1.0));
    
       k_inv_z->Assemble(skip_zeros);
       k_inv_z->Finalize(skip_zeros);
       
       k_vis_z = new ParBilinearForm(fes_op);
       k_vis_z->AddDomainIntegrator(new ConvectionIntegrator(z_dir,  1.0));
       k_vis_z->AddInteriorFaceIntegrator(
          new TransposeIntegrator(new DGTraceIntegrator(z_dir, -1.0,  0.0)));// Beta 0 means central flux
    
       k_vis_z->Assemble(skip_zeros);
       k_vis_z->Finalize(skip_zeros);
   }

   ///////////////////////////////////////////////////////////////
   //Setup time stepping objects and do initial post-processing
   
   FE_Evolution *adv;

   HypreParMatrix *M       = m->ParallelAssemble();
   HypreParMatrix *K_inv_x = k_inv_x->ParallelAssemble();
   HypreParMatrix *K_inv_y = k_inv_y->ParallelAssemble();

   HypreParMatrix *K_vis_x = k_vis_x->ParallelAssemble();
   HypreParMatrix *K_vis_y = k_vis_y->ParallelAssemble();

   HypreParMatrix *K_inv_z;
   HypreParMatrix *K_vis_z;

   if (dim == 3)
   {
       K_inv_z = k_inv_z->ParallelAssemble();
       K_vis_z = k_vis_z->ParallelAssemble();

       adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, *K_inv_z, 
                   *K_vis_x, *K_vis_y, *K_vis_z, *b);
       
       getAuxGrad(dim, *K_vis_x, *K_vis_y, *K_vis_z, 
           adv->GetMSolver(), *U, aux_grad);
   }
   else if (dim == 2)
   {
       HypreParMatrix dummyMat;
       adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, dummyMat,
                            *K_vis_x, *K_vis_y, dummyMat, *b);
       
       getAuxGrad(dim, *K_vis_x, *K_vis_y, dummyMat, 
           adv->GetMSolver(), *U, aux_grad);
   }
   delete m;
   delete k_inv_x, k_inv_y, k_inv_z;
   delete k_vis_x, k_vis_y, k_vis_z; // Cleanup memory

   t = 0.0; ti = 0; // Initialize time and time iterations
   postProcess(*pmesh, *u_sol, aux_grad, ti, t);

   adv->SetTime(t);

   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK3SSPSolver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
   }

   ode_solver->Init(*adv);

   dt = dt_const; 

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   tke_file.open("tke.dat");

   bool done = false;
   for (ti = 0; !done; )
   {
      Step(); // Step in time

      done = (t >= t_final - 1e-8*dt);

      if (ti % 25 == 0) // Check time
      {
          chrono.Stop();
          if (myid == 0)
          {
              cout << "25 Steps took "<< chrono.RealTime() << " s "<< endl;
          }

          chrono.Clear();
          chrono.Start();
      }
    
      if (done || ti % vis_steps == 0) // Visualize
      {
          if (dim == 2)
          {
              HypreParMatrix dummyMat;
              getAuxGrad(dim, *K_vis_x, *K_vis_y, dummyMat, 
                      adv->GetMSolver(), *U, aux_grad);
          }
          else if (dim == 3)
          {
              getAuxGrad(dim, *K_vis_x, *K_vis_y, *K_vis_z, 
                      adv->GetMSolver(), *U, aux_grad);
          }

          postProcess(*pmesh, *u_sol, aux_grad, ti, t);
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
////       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << rhs(sub2)<< endl;      
////       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << u_out(sub1) << endl;      
//   }
//
   
   tke_file.close();
   delete adv;
   delete M, K_inv_x, K_inv_y, K_inv_z;
   delete K_vis_x, K_vis_y, K_vis_z;
}


CNS::~CNS()
{
    delete u_sol, f_inv;
    delete U;
    delete b;
    delete ode_solver;
    delete pmesh;
    delete fes;
    delete fes_vec;
    delete fes_op;
}

void CNS::Step()
{
    *u_sol = *U;
    u_sol->ExchangeFaceNbrData();
    getInvFlux(dim, *u_sol, *f_inv); // To update f_vec

    b->Assemble();

    double dt_real = min(dt, t_final - t);
    ode_solver->Step(*U, t, dt_real);
    ti++;

    MPI_Comm comm = pmesh->GetComm();
      
    double loc_u_max = getUMax(dim, *U); // Get u_max on local processor
    MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors
    
    double loc_tke   = ComputeTKE(*fes, *U); // Get TKE 
    double glob_tke;
    MPI_Allreduce(&loc_tke, &glob_tke, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors

    if (myid == 0)
    {
        tke_file << setprecision(12) << ti << "\t" << t << "\t" << glob_tke << endl;
        cout << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
            t << ", max_speed " << glob_u_max << ", fes_size " << fes->GlobalTrueVSize() << endl;
    }

    dt = dt_const; 

}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            HypreParMatrix &_K_vis_x, HypreParMatrix &_K_vis_y, HypreParMatrix &_K_vis_z,
                            Vector &_b)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), K_inv_z(_K_inv_z),
                            K_vis_x(_K_vis_x), K_vis_y(_K_vis_y), K_vis_z(_K_vis_z), b(_b),
                            M_solver(M.GetComm())
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
    y_temp.SetSize(x.Size()); 

    Vector f(dim*x.Size());
    getInvFlux(dim, x, f);

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
        f.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
        b.GetSubVector(offsets[i], b_sub);
        f_x += b_sub; // Needs to be added only once
        y_temp.SetSubVector(offsets[i], f_x);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i - var_dim], f_x);
    }
    y += y_temp;


    if (dim == 3)
    {
        for(int i = 2*var_dim + 0; i < 3*var_dim; i++)
        {
            f.GetSubVector(offsets[i], f_sol);
            K_inv_z.Mult(f_sol, f_x);
            y_temp.SetSubVector(offsets[i - 2*var_dim], f_x);
        }
        y += y_temp;
    }

    //////////////////////////////////////////////
    //Get viscous contribution
    Vector f_vis(dim*x.Size());
    Vector aux_grad(dim*(dim + 1)*offset);

    getAuxGrad(dim, K_vis_x, K_vis_y, K_vis_z, M_solver, x, aux_grad);
    getVisFlux(dim, x, aux_grad, f_vis);

    for(int i = 0; i < var_dim; i++)
    {
        f_vis.GetSubVector(offsets[i], f_sol);
        K_vis_x.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i], f_x);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f_vis.GetSubVector(offsets[i], f_sol);
        K_vis_y.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i - var_dim], f_x);
    }
    y += y_temp;

    if (dim == 3)
    {
        for(int i = 2*var_dim + 0; i < 3*var_dim; i++)
        {
            f_vis.GetSubVector(offsets[i], f_sol);
            K_vis_z.Mult(f_sol, f_x);
            y_temp.SetSubVector(offsets[i - 2*var_dim], f_x);
        }
        y += y_temp;
    }

    for(int i = 0; i < var_dim; i++)
    {
        y.GetSubVector(offsets[i], f_x);
        M_solver.Mult(f_x, f_x_m);
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
        const CGSolver &M_solver, const Vector &u, Vector &aux_grad)
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

    Vector aux_var(offset), aux_x(offset);
    for(int i = 0; i < aux_dim; i++)
    {
        aux_sol.GetSubVector(offsets[i], aux_var);

        K_x.Mult(aux_var, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[0*aux_dim + i], aux_x);

        K_y.Mult(aux_var, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[1*aux_dim + i], aux_x);
    
        if (dim == 3)
        {
            K_z.Mult(aux_var, aux_x);
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


void getFields(const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &M,
                Vector &p, Vector &vort, Vector &q)
{

    int vDim    = u_sol.VectorDim();
    int dofs    = u_sol.Size()/vDim;
    int dim     = vDim - 2;

    int aux_dim = vDim - 1;

    double u_grad[dim][dim], omega_sq, s_sq;

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
        
        for (int j = 0; j < dim; j++)
        {
            for (int k = 0; k < dim; k++)
            {
                u_grad[j][k] = aux_grad[(k*aux_dim + j  )*dofs + i];
            }
        }
    
        if (dim == 2)
        {
           vort[i] = u_grad[1][0] - u_grad[0][1];     
        }
        else if (dim == 3)
        {
            double w_x  = u_grad[2][1] - u_grad[1][2];
            double w_y  = u_grad[0][2] - u_grad[2][0];
            double w_z  = u_grad[1][0] - u_grad[0][1];
            double w_sq = pow(w_x, 2) + pow(w_y, 2) + pow(w_z, 2); 

            vort[i]   = sqrt(w_sq);
        }
        if (dim == 2)
        {
            double s_z      = u_grad[1][0] + u_grad[0][1];     
                   s_sq     = pow(s_z, 2); 
                   omega_sq = s_sq; // q criterion makes sense in 3D only
        }
        else if (dim == 3)
        {
            double omega_x  = 0.5*(u_grad[2][1] - u_grad[1][2]);
            double omega_y  = 0.5*(u_grad[0][2] - u_grad[2][0]);
            double omega_z  = 0.5*(u_grad[1][0] - u_grad[0][1]);
                   omega_sq = 2*(pow(omega_x, 2) + pow(omega_y, 2) + pow(omega_z, 2)); 

            double s_23  = 0.5*(u_grad[2][1] + u_grad[1][2]);
            double s_13  = 0.5*(u_grad[0][2] + u_grad[2][0]);
            double s_12  = 0.5*(u_grad[1][0] + u_grad[0][1]);

            double s_11  = u_grad[0][0]; 
            double s_22  = u_grad[1][1]; 
            double s_33  = u_grad[2][2]; 

                   s_sq = 2*(pow(s_12, 2) + pow(s_13, 2) + pow(s_23, 2)) + s_11*s_11 + s_22*s_22 + s_33*s_33; 
        }
            
        q[i]      = 0.5*(omega_sq - s_sq);
    }
}




void postProcess(ParMesh &mesh, ParGridFunction &u_sol, ParGridFunction &aux_grad,
                int cycle, double time)
{
   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   DG_FECollection fec(order + 1, dim);
   ParFiniteElementSpace fes_post(&mesh, &fec, var_dim);
   ParFiniteElementSpace fes_post_grad(&mesh, &fec, (dim+1)*dim);

   ParGridFunction u_post(&fes_post);
   u_post.GetValuesFrom(u_sol); // Create a temp variable to get the previous space solution
 
   ParGridFunction aux_grad_post(&fes_post_grad);
   aux_grad_post.GetValuesFrom(aux_grad); // Create a temp variable to get the previous space solution

   fes_post.Update();
   u_post.Update();
   aux_grad_post.Update();

   VisItDataCollection dc("CNS", &mesh);
   dc.SetPrecision(8);
 
   ParFiniteElementSpace fes_fields(&mesh, &fec);
   ParGridFunction rho(&fes_fields);
   ParGridFunction M(&fes_fields);
   ParGridFunction p(&fes_fields);
   ParGridFunction vort(&fes_fields);
   ParGridFunction q(&fes_fields);

   dc.RegisterField("rho", &rho);
   dc.RegisterField("M", &M);
   dc.RegisterField("p", &p);
   dc.RegisterField("vort", &vort);
   dc.RegisterField("q_criterion", &q);

   getFields(u_post, aux_grad_post, rho, M, p, vort, q);

   dc.SetCycle(cycle);
   dc.SetTime(time);
   dc.Save();
  
}


// Returns TKE 
double ComputeTKE(FiniteElementSpace &fes, const Vector &uD)
{
   const FiniteElement *el;
   ElementTransformation *T;

   int dim;

   double tke = 0.0;
   for (int i = 0; i < fes.GetNE(); i++)
   {
       T  = fes.GetElementTransformation(i);
       el = fes.GetFE(i);
   
       dim = el->GetDim();

       int dof = el->GetDof();
       Array<int> vdofs;
       fes.GetElementVDofs(i, vdofs);

       const IntegrationRule *ir ;
       int   order;

       order = 2*el->GetOrder() + 1;
       ir    = &IntRules.Get(el->GetGeomType(), order);

       double local_ke = 0.0;
       for (int p = 0; p < ir->GetNPoints(); p++)
       {
           const IntegrationPoint &ip = ir->IntPoint(p);

           double rho    = uD[vdofs[p]];
           double irho   = 1.0/rho; 

           double point_ke = 0.0;       
           for (int j = 0; j < dim; j++)
           {
               point_ke +=  irho*0.5*(uD[vdofs[(1 + j)*dof + p]] * uD[vdofs[(1 + j)*dof + p]]);
           }
           local_ke += ip.weight*T->Weight()*point_ke;

           tke += ip.weight*T->Weight()*point_ke;
       }
   }

   return tke;
}

