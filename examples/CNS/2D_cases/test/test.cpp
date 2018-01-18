#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.000625;
const double R_gas = 287;
const double   Pr  = 0.72;

//Run parameters
//const char *mesh_file        =  "periodic-cube.mesh";
const char *mesh_file        =  "per_5.mesh";
const int    order           =  3;
const double t_final         =  1.0    ;
const int    problem         =  1;
const int    ref_levels      =  3;

const bool   time_adapt      =  false;
const double cfl             =  1.1  ;
const double dt_const        =  0.001  ;
const int    ode_solver_type =  3; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       = 50;

const bool   adapt           =  false;
const int    adapt_iter      =  200  ; // Time steps after which adaptation is done
const double tolerance       =  5e-4 ; // Tolerance for adaptation


// Velocity coefficient
void init_function(const Vector &x, Vector &v);
void char_bnd_cnd(const Vector &x, Vector &v);
void wall_bnd_cnd(const Vector &x, Vector &v);
void wall_adi_bnd_cnd(const Vector &x, Vector &v);

double getUMax(int dim, const Vector &u);

int GetFacePtsSize(Mesh &mesh, FiniteElementSpace &fes);
void AssembleFaceMatrices(Mesh &mesh, FiniteElementSpace &fes, FiniteElementSpace &fes_var,
        Vector &u, Vector &f,
        SparseMatrix &face_project_l, SparseMatrix &face_project_r, SparseMatrix &wts,
        Vector &nor_face);
void getVectorLFFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f);
void getFaceDotNorm(int dim, const Vector &f, const Vector &nor_face, Vector &face_f);
void getEulerDGTranspose(int dim, SparseMatrix &face_project_l, SparseMatrix &face_project_r, 
        SparseMatrix &wts, Vector &nor,
        const Vector &u, const Vector &f, Vector &b);


void getInvFlux(int dim, const Vector &u, Vector &f);
void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z,
        const CGSolver &M_solver, const Vector &u, 
        const Vector &b_aux_x, const Vector &b_aux_y, const Vector &b_aux_z,        
        Vector &aux_grad);
void getAuxVar(int dim, const Vector &u, Vector &aux_sol);

void ComputeLift(Mesh &mesh, FiniteElementSpace &fes, GridFunction &uD, GridFunction &f_vis_D, 
                    const Array<int> &bdr, const double gamm, Vector &force);

void getFields(const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &u1, Vector &u2, 
                Vector &E, Vector &u_x, Vector &u_y, Vector &v_x, Vector &v_y);
void postProcess(Mesh &mesh, GridFunction &u_sol, GridFunction &aux_grad,
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
   HypreParMatrix &K_vis_x, &K_vis_y, &K_vis_z;
   ParLinearForm  &b, &b_aux_x, &b_aux_y, &b_aux_z;

   SparseMatrix   &face_proj_l, &face_proj_r, &wts;
   Vector         &nor_face;

   HypreSmoother M_prec;
   CGSolver M_solver;
                            
   ParGridFunction &u, &u_aux, &u_grad, &f_I, &f_V;
                            
public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            HypreParMatrix &_K_vis_x, HypreParMatrix &_K_vis_y, HypreParMatrix &_K_vis_z,
                            ParGridFunction &u_,   ParGridFunction &u_aux, ParGridFunction &u_grad, 
                            ParGridFunction &f_I_, ParGridFunction &f_V_,
                            ParLinearForm &_b_aux_x, ParLinearForm &_b_aux_y, ParLinearForm &_b_aux_z, 
                            ParLinearForm &_b,
                            SparseMatrix &face_proj_l_, SparseMatrix &face_proj_r_, SparseMatrix &wts_,
                            Vector &nor_face_);

   void GetSize() ;

   CGSolver &GetMSolver() ;

   void Update();

   virtual void Mult(const ParGridFunction &x, ParGridFunction &y) const;

   virtual ~FE_Evolution() { }
};

class CNS 
{
private:
    int num_procs, myid;

    ParMesh *pmesh ;

    ParFiniteElementSpace  *fes, *fes_vec, *fes_op;
   
    ParGridFunction *u_sol, *f_inv, *f_vis;   
    ParGridFunction *aux_sol, *aux_grad;   
    ParGridFunction *u_b, *f_I_b;   // Used in calculating b
    ParLinearForm   *b, *b_aux_x, *b_aux_y, *b_aux_z;

    ODESolver *ode_solver; 
    FE_Evolution *adv;
    ParGridFunction *u_t, *k_t, *y_t;   

    SparseMatrix *face_proj_l, *face_proj_r, *wts;
    Vector nor_face;

    int dim;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, dt_real, t;
    int ti;

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
   Mesh *mesh  = new Mesh(mesh_file, 1, 1);
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

   fes_vec = new ParFiniteElementSpace(pmesh, &fec, dim*var_dim);
       
   f_inv = new ParGridFunction(fes_vec);
   getInvFlux(dim, *u_sol, *f_inv);

   f_vis = new ParGridFunction(fes_vec);

   ParFiniteElementSpace fes_aux(pmesh, &fec, aux_dim);
   aux_sol = new ParGridFunction(&fes_aux);

   ParFiniteElementSpace fes_aux_grad(pmesh, &fec, dim*aux_dim);
   aux_grad = new ParGridFunction(&fes_aux_grad);

   // For time stepping
   u_t = new ParGridFunction(fes);
   k_t = new ParGridFunction(fes);
   y_t = new ParGridFunction(fes);
   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector xdir(dim), ydir(dim), zdir(dim); 
   xdir = 0.0; xdir(0) = 1.0;
   VectorConstantCoefficient x_dir(xdir);
   ydir = 0.0; ydir(1) = 1.0;
   VectorConstantCoefficient y_dir(ydir);

   ParBilinearForm *m, *k_inv_x, *k_inv_y, *k_inv_z;
   ParBilinearForm     *k_vis_x, *k_vis_y, *k_vis_z;

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

   u_b    = new ParGridFunction(fes);
   f_I_b  = new ParGridFunction(fes_vec);

   VectorGridFunctionCoefficient u_vec(u_b);
   VectorGridFunctionCoefficient f_vec(f_I_b);

   *u_b   = *u_sol;
   *f_I_b = *f_inv;

   u_b  ->ExchangeFaceNbrData(); //Exchange data across processors
   f_I_b->ExchangeFaceNbrData();
   
   aux_grad->ExchangeFaceNbrData();

   ///////////////////////////////////////////////////
   // Linear form representing the Euler boundary non-linear term
   b = new ParLinearForm(fes);
   b->AddFaceIntegrator(
      new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, var_dim, -1.0));

   b->Assemble();

   getAuxVar(dim, *u_sol, *aux_sol);
   VectorGridFunctionCoefficient aux_vec(aux_sol);

   b_aux_x = new ParLinearForm(&fes_aux);
   b_aux_y = new ParLinearForm(&fes_aux);

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

   VectorConstantCoefficient z_dir(ydir);
   if (dim == 3)
   {
       zdir = 0.0; zdir(2) = 1.0;
       VectorConstantCoefficient temp_z_dir(zdir);
       z_dir = temp_z_dir;
      
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
       
       b_aux_z = new ParLinearForm(&fes_aux);
   }

   HypreParMatrix *M       = m->ParallelAssemble();
   HypreParMatrix *K_inv_x = k_inv_x->ParallelAssemble();
   HypreParMatrix *K_inv_y = k_inv_y->ParallelAssemble();

   HypreParMatrix *K_vis_x = k_vis_x->ParallelAssemble();
   HypreParMatrix *K_vis_y = k_vis_y->ParallelAssemble();

   HypreParMatrix *K_inv_z;
   HypreParMatrix *K_vis_z;

   int n_face_pts = GetFacePtsSize(*pmesh, *fes_op); 
   face_proj_l    = new SparseMatrix(n_face_pts, fes_op->GetVSize());
   face_proj_r    = new SparseMatrix(n_face_pts, fes_op->GetVSize());
   wts            = new SparseMatrix(n_face_pts);
   nor_face.SetSize(dim*n_face_pts);
   AssembleFaceMatrices(*pmesh, *fes_op, *fes, *u_sol, *f_inv, 
           *face_proj_l, *face_proj_r, *wts, nor_face);

   ///////////////////////////////////////////////////////////////
   //Setup time stepping objects and do initial post-processing
   if (dim == 3)
   {
       K_inv_z = k_inv_z->ParallelAssemble();
       K_vis_z = k_vis_z->ParallelAssemble();

       adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, *K_inv_z,
                            *K_vis_x, *K_vis_y, *K_vis_z, 
                            *u_b, *aux_sol, *aux_grad, *f_I_b, *f_vis,
                            *b_aux_x, *b_aux_y, *b_aux_z, 
                            *b,
                            *face_proj_l, *face_proj_r, *wts,
                            nor_face);
   }
   else if (dim == 2)
   {
       adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, *K_inv_z,
                            *K_vis_x, *K_vis_y, *K_vis_z, 
                            *u_b, *aux_sol, *aux_grad, *f_I_b, *f_vis,
                            *b_aux_x, *b_aux_y, *b_aux_z, 
                            *b,
                            *face_proj_l, *face_proj_r, *wts,
                            nor_face);
   }
   delete m, k_inv_x, k_inv_y, k_inv_z;
   delete    k_vis_x, k_vis_y, k_vis_z;


   {// Post process initially
       ti = 0; t = 0;
       postProcess(*pmesh, *u_sol, *aux_grad, ti, t);
   }

   if (time_adapt == false)
   {
       dt = dt_const; 
   }
   else
   {
       MPI_Comm comm = pmesh->GetComm();

       double loc_u_max, glob_u_max; 
       loc_u_max = getUMax(dim, *u_sol); // Get u_max on local processor
       MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

       dt = cfl*((h_min/(2.0*order + 1))/glob_u_max);
   }
    
   MPI_Comm comm = pmesh->GetComm();

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   Step();

//   bool done = false;
//   for (ti = 0; !done; )
//   {
//      Step(); // Step in time
//
//      done = (t >= t_final - 1e-8*dt);
//
//      if ((ti % 10 == 0) && (myid == 0)) // Check time
//      {
//          chrono.Stop();
//          cout << "10 Steps took "<< chrono.RealTime() << " s "<< endl;
//
//          chrono.Clear();
//          chrono.Start();
//      }
//    
//      if (done || ti % vis_steps == 0) // Visualize
//      {
//       
//          getAuxGrad(dim, *K_vis_x, *K_vis_y, *K_vis_z, 
//          adv->GetMSolver(), *u_b,
//          *b_aux_x, *b_aux_y, *b_aux_z, 
//          *aux_grad);
//
//          postProcess(*pmesh, *u_sol, *aux_grad, ti, t);
//      }
//  
//   }
   
   
   delete adv;
   delete K_inv_x, K_vis_x;
   delete K_inv_y, K_vis_y;
   delete K_inv_z, K_vis_z;
}


CNS::~CNS()
{
    delete pmesh;
    delete u_b, f_I_b;
    delete u_t, k_t, y_t;
    delete u_sol, f_inv, f_vis, aux_sol, aux_grad ;
    delete b, b_aux_x, b_aux_y, b_aux_z;
    delete fes;
    delete fes_vec;
    delete fes_op;

    delete face_proj_l, face_proj_r, wts;

    MPI_Finalize();
}

void CNS::Step()
{
    dt_real = min(dt, t_final - t);
    
    // Euler or RK3SSP
    ////////////////////
    *u_t = *u_sol;

    if (ode_solver_type == 1)
    {
        adv->Mult(*u_t, *k_t);
        add(*u_t, dt_real, *k_t, *u_sol);
    }
    else if (ode_solver_type == 2)
    {
        adv->Mult(*u_t, *k_t);
    
        // x1 = x + k0, t1 = t + dt, k1 = dt*f(t1, x1)
        add(*u_t, dt_real, *k_t, *y_t);
        adv->Mult(*y_t, *k_t);
     
        // x2 = 3/4*x + 1/4*(x1 + k1), t2 = t + 1/2*dt, k2 = dt*f(t2, x2)
        y_t->Add(dt_real, *k_t);
        add(3./4, *u_t, 1./4, *y_t, *y_t);
        adv->Mult(*y_t, *k_t);
    
        // x3 = 1/3*x + 2/3*(x2 + k2), t3 = t + dt
        y_t->Add(dt_real, *k_t);
        add(1./3, *u_t, 2./3, *y_t, *u_sol);
    }
    else if (ode_solver_type == 3)
    {
        // SSPRK43
        adv->Mult(*u_t, *k_t);

        add(*u_t, (1./2)*dt_real, *k_t, *y_t);
        adv->Mult(*y_t, *k_t);

        add(*y_t, (1./2)*dt_real, *k_t, *y_t);
        adv->Mult(*y_t, *k_t); 

        add(2./3, *u_t, 1./3, *y_t, *y_t);
        add(*y_t, (1./6)*dt_real, *k_t, *y_t);
        adv->Mult(*y_t, *k_t);

        add(*y_t, (1./2)*dt_real, *k_t, *u_sol);
    }

    t += dt_real;
    ////////////////////

    ti++;

    MPI_Comm comm = pmesh->GetComm();

    double loc_u_max, glob_u_max; 
    loc_u_max = getUMax(dim, *u_sol); // Get u_max on local processor
    MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

    if (myid == 0)
    {
        cout << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
            t << ", max_speed " << glob_u_max << ", fes_size " << fes->GlobalTrueVSize() << endl;
    }

    if (time_adapt == false)
    {
        dt = dt_const; 
    }
    else
    {
        dt = cfl*((h_min/(2.0*order + 1))/glob_u_max);
    }

}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            HypreParMatrix &_K_vis_x, HypreParMatrix &_K_vis_y, HypreParMatrix &_K_vis_z,
                            ParGridFunction &u_,   ParGridFunction &u_aux_, ParGridFunction &u_grad_, 
                            ParGridFunction &f_I_, ParGridFunction &f_V_,
                            ParLinearForm &_b_aux_x, ParLinearForm &_b_aux_y, ParLinearForm &_b_aux_z, 
                            ParLinearForm &_b,
                            SparseMatrix &face_proj_l_, SparseMatrix &face_proj_r_, SparseMatrix &wts_,
                            Vector &nor_face_)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), K_inv_z(_K_inv_z),
                            K_vis_x(_K_vis_x), K_vis_y(_K_vis_y), K_vis_z(_K_vis_z), 
                            u(u_), u_aux(u_aux_), u_grad(u_grad_), f_I(f_I_), f_V(f_V_), 
                            b_aux_x(_b_aux_x), b_aux_y(_b_aux_y), b_aux_z(_b_aux_z), b(_b),
                            face_proj_l(face_proj_l_), face_proj_r(face_proj_r_), wts(wts_),
                            nor_face(nor_face_)

{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);
}



void FE_Evolution::Mult(const ParGridFunction &x, ParGridFunction &y) const
{
    int dim = x.Size()/K_inv_x.GetNumRows() - 2;
    int var_dim = dim + 2;

    u = x;
    getInvFlux(dim, u, f_I); // To update f_vec
    u.ExchangeFaceNbrData();
    
    getAuxVar(dim, x, u_aux);
    u_aux.ExchangeFaceNbrData();

    b_aux_x *= 0.0;    
    b_aux_y *= 0.0;    
    if (dim == 3)
        b_aux_z *= 0.0;
    getAuxGrad(dim, K_vis_x, K_vis_y, K_vis_z, M_solver, x, 
            b_aux_x, b_aux_y, b_aux_z,
            u_grad);
    getVisFlux(dim, x, u_grad, f_V);

    b.Assemble();

    getEulerDGTranspose(dim, face_proj_l, face_proj_r, 
        wts, nor_face, u, f_I, b);
    
    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector y_temp;
    y_temp.SetSize(x.Size()); 

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
        f_I.GetSubVector(offsets[i], f_sol);
        K_inv_x.Mult(f_sol, f_x);
        b.GetSubVector(offsets[i], b_sub);
        f_x += b_sub; // Needs to be added only once
        y_temp.SetSubVector(offsets[i], f_x);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f_I.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i - var_dim], f_x);
    }
    y += y_temp;


    if (dim == 3)
    {
        for(int i = 2*var_dim + 0; i < 3*var_dim; i++)
        {
            f_I.GetSubVector(offsets[i], f_sol);
            K_inv_z.Mult(f_sol, f_x);
            y_temp.SetSubVector(offsets[i - 2*var_dim], f_x);
        }
        y += y_temp;
    }

    //////////////////////////////////////////////
    //Get viscous contribution
    for(int i = 0; i < var_dim; i++)
    {
        f_V.GetSubVector(offsets[i], f_sol);
        K_vis_x.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i], f_x);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f_V.GetSubVector(offsets[i], f_sol);
        K_vis_y.Mult(f_sol, f_x);
        y_temp.SetSubVector(offsets[i - var_dim], f_x);
    }
    y += y_temp;

    if (dim == 3)
    {
        for(int i = 2*var_dim + 0; i < 3*var_dim; i++)
        {
            f_V.GetSubVector(offsets[i], f_sol);
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
        aux_grad.SetSubVector(offsets[0*aux_dim + i], aux_x);

        K_y.Mult(aux_var, aux_x);
        b_aux_y.GetSubVector(offsets[i], b_sub);
        add(b_sub, aux_x, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[1*aux_dim + i], aux_x);
    
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
           u3    = 0;
           T     = 1 + dT;
           rho   = pow(T, 1/(gamm - 1));
           p     = rho*R_gas*T;
       }
       
 

       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = rho * u3;                //rho * v
       v(4) = p/(gamm - 1) + 0.5*rho*v_sq;
   }

}

void char_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p;
       rho = 1;
       u1  = 3.0924; u2 = 0.2162; 
       p   = 172.2;
    
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
   else if (dim == 3)
   {
       double rho, u1, u2, u3, p;
       rho = 1;
       u1  = 3.0924; u2 = 0.2162; u3 = 0.0;
       p   =172.2;
    
       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = rho * u3;                //rho * w
       v(4) = p/(gamm - 1) + 0.5*rho*v_sq;
   }

}


void wall_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.6;  // Temp 
   }
   else if (dim == 3)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.0;  // z velocity 
       v(3) = 0.6;  // Temp 
   }
}

void wall_adi_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
   }
   if (dim == 3)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.0;  // y velocity 
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




void postProcess(Mesh &mesh, GridFunction &u_sol, GridFunction &aux_grad,
                int cycle, double time)
{
   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   DG_FECollection fec(order , dim);
   FiniteElementSpace fes_post(&mesh, &fec, var_dim);
   FiniteElementSpace fes_post_grad(&mesh, &fec, (dim+1)*dim);

   GridFunction u_post(&fes_post);
   u_post.GetValuesFrom(u_sol); // Create a temp variable to get the previous space solution
 
   GridFunction aux_grad_post(&fes_post_grad);
   aux_grad_post.GetValuesFrom(aux_grad); // Create a temp variable to get the previous space solution

   VisItDataCollection dc("CNS", &mesh);
   dc.SetPrecision(8);
 
   FiniteElementSpace fes_fields(&mesh, &fec);
   GridFunction rho(&fes_fields);
   GridFunction M(&fes_fields);
   GridFunction p(&fes_fields);
   GridFunction vort(&fes_fields);
   GridFunction q(&fes_fields);

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


void ComputeLift(Mesh &mesh, FiniteElementSpace &fes, GridFunction &uD, GridFunction &f_vis_D, 
                    const Array<int> &bdr, const double gamm, Vector &force)
{
   force = 0.0; 

   const FiniteElement *el;
   FaceElementTransformations *T;

   int dim;
   Vector nor;

   Vector vals, vis_vals;
   
   for (int i = 0; i < fes.GetNBE(); i++)
   {
       const int bdr_attr = mesh.GetBdrAttribute(i);

       if (bdr[bdr_attr-1] == 0) { continue; } // Skip over non-active boundaries

       T = mesh.GetBdrFaceTransformations(i);
 
       el = fes.GetFE(T -> Elem1No);
   
       dim = el->GetDim();
      
       const IntegrationRule *ir ;
       int order;
       
       order = T->Elem1->OrderW() + 2*el->GetOrder();
       
       if (el->Space() == FunctionSpace::Pk)
       {
          order++;
       }
       ir = &IntRules.Get(T->FaceGeom, order);

       for (int p = 0; p < ir->GetNPoints(); p++)
       {
          const IntegrationPoint &ip = ir->IntPoint(p);
          IntegrationPoint eip;
          T->Loc1.Transform(ip, eip);
    
          T->Face->SetIntPoint(&ip);
          T->Elem1->SetIntPoint(&eip);

          nor.SetSize(dim);          
          if (dim == 1)
          {
             nor(0) = 2*eip.x - 1.0;
          }
          else
          {
             CalcOrtho(T->Face->Jacobian(), nor);
          }
 
          Vector nor_dim(dim);
          double nor_l2 = nor.Norml2();
          nor_dim.Set(1/nor_l2, nor);
     
          uD.GetVectorValue(T->Elem1No, eip, vals);

          Vector vel(dim);    
          double rho = vals(0);
          double v_sq  = 0.0;
          for (int j = 0; j < dim; j++)
          {
              vel(j) = vals(1 + j)/rho;      
              v_sq    += pow(vel(j), 2);
          }
          double pres = (gamm - 1)*(vals(dim + 1) - 0.5*rho*v_sq);

          // nor is measure(face) so area is included
          // The normal is always going into the boundary element
          // F = -p n_j with n_j going out
          force(0) += pres*nor(0)*ip.weight; // Quadrature of pressure 
          force(1) += pres*nor(1)*ip.weight; // Quadrature of pressure 

          f_vis_D.GetVectorValue(T->Elem1No, eip, vis_vals);
        
          double tau[dim][dim];
          for(int i = 0; i < dim ; i++)
              for(int j = 0; j < dim ; j++) tau[i][j] = vis_vals[i*(dim + 2) + 1 + j];


          // The normal is always going into the boundary element
          // F = sigma_{ij}n_j with n_j going out
          for(int i = 0; i < dim ; i++)
          {
              force(0) -= tau[0][i]*nor(i)*ip.weight; // Quadrature of shear stress 
              force(1) -= tau[1][i]*nor(i)*ip.weight; // Quadrature of shear stress 
          }
       } // p loop
   } // NBE loop
}


int GetFacePtsSize(Mesh &mesh, FiniteElementSpace &fes)
{
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;

   int dim, var_dim, ndof1, ndof2;
   int order;

   double w; // weight
   double alpha =  -1.0;

   Vector shape1, shape2;

   Vector nor;
   Array<int> vdofs, vdofs2;

   int n_face_pts = 0;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
       T = mesh.GetInteriorFaceTransformations(i);
 
       el1 = fes.GetFE(T -> Elem1No);
       el2 = fes.GetFE(T -> Elem2No);

       dim     = el1->GetDim();
       var_dim = dim + 2;

       const IntegrationRule *ir ;

       order = (std::min(T->Elem1->OrderW(), T->Elem2->OrderW()) +
               2*std::max(el1->GetOrder(), el2->GetOrder()));
       
       ir = &IntRules.Get(T->FaceGeom, order);

       n_face_pts += ir->GetNPoints(); 
   }
   return n_face_pts;
}


void AssembleFaceMatrices(Mesh &mesh, FiniteElementSpace &fes, FiniteElementSpace &fes_var,
        Vector &u, Vector &f,
        SparseMatrix &face_project_l, SparseMatrix &face_project_r, SparseMatrix &wts,
        Vector &nor_face)

{
   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;

   int dim, var_dim, ndof1, ndof2;
   int order;

   double w; // weight
   double alpha =  -1.0;

   Vector shape1, shape2;

   Vector nor;
   Array<int> vdofs, vdofs2;

   int n_face_pts = wts.Size();
   int dofs       = fes.GetVSize();

   dim     = mesh.SpaceDimension();
   var_dim = dim + 2;

   int face_coun = 0;
   for (int i = 0; i < mesh.GetNumFaces(); i++)
   {
       T = mesh.GetInteriorFaceTransformations(i);

       fes.GetElementVDofs (T -> Elem1No, vdofs);
       fes.GetElementVDofs (T -> Elem2No, vdofs2);
 
       el1 = fes.GetFE(T -> Elem1No);
       el2 = fes.GetFE(T -> Elem2No);

       ndof1   = el1->GetDof();
       ndof2   = el2->GetDof();

       shape1.SetSize(ndof1);
       shape2.SetSize(ndof2);
       
       nor.SetSize(dim);

       const IntegrationRule *ir ;

       order = (std::min(T->Elem1->OrderW(), T->Elem2->OrderW()) +
               2*std::max(el1->GetOrder(), el2->GetOrder()));
       
       ir = &IntRules.Get(T->FaceGeom, order);

       for (int p = 0; p < ir->GetNPoints(); p++)
       {
          const IntegrationPoint &ip = ir->IntPoint(p);
          IntegrationPoint eip1, eip2;
          T->Loc1.Transform(ip, eip1);
          T->Loc2.Transform(ip, eip2);

          T->Face->SetIntPoint(&ip);
          T->Elem1->SetIntPoint(&eip1);
          T->Elem2->SetIntPoint(&eip2);

          el1->CalcShape(eip1, shape1);
          el2->CalcShape(eip2, shape2);

          for(int i=0; i < shape1.Size(); i++)
          {
              if (std::abs(shape1(i)) < std::numeric_limits<double>::epsilon() )
                  shape1(i) = 0.0;
          }
          for(int i=0; i < shape2.Size(); i++)
          {
              if (std::abs(shape2(i)) < std::numeric_limits<double>::epsilon() )
                  shape2(i) = 0.0;
          }

          face_project_l.SetRow(face_coun, vdofs,  shape1);
          face_project_r.SetRow(face_coun, vdofs2, shape2);

          w = ip.weight * alpha; 
          wts.Set(face_coun, face_coun, w);

          CalcOrtho(T->Face->Jacobian(), nor);
          for(int i=0; i < dim; i++)
          {
              nor_face(i*n_face_pts + face_coun) = nor(i);          
          }

          face_coun++;

     }// p loop 

   } // NumFaces loop

   face_project_l.Finalize();
   face_project_r.Finalize();
   wts.Finalize();

}

void getEulerDGTranspose(int dim, SparseMatrix &face_project_l, SparseMatrix &face_project_r, 
        SparseMatrix &wts, Vector &nor_face,
        const Vector &u, const Vector &f, Vector &b)
{
   int var_dim    = dim + 2;
   int n_face_pts = face_project_l.Size();
   int dofs       = u.Size()/var_dim;

   Array<int> offsets[dim*var_dim], offsets_face[dim*var_dim];
   for(int i = 0; i < dim*var_dim; i++)
   {
       offsets_face[i].SetSize(n_face_pts);
       offsets     [i].SetSize(dofs);
   }
   for(int j = 0; j < dim*var_dim; j++)
   {
       for(int i = 0; i < n_face_pts; i++)
       {
           offsets_face[j][i] = j*n_face_pts + i ;
       }
       for(int i = 0; i < dofs; i++)
       {
           offsets     [j][i] = j*dofs + i ;
       }
   }

   Vector u_l(var_dim*n_face_pts), u_r(var_dim*n_face_pts);
   Vector u_f_sub(n_face_pts), u_sub(dofs);
   for(int j = 0; j < var_dim; j++)
   {
       u.GetSubVector(offsets[j], u_sub);

       face_project_l.Mult(u_sub, u_f_sub);
       u_l.SetSubVector(offsets_face[j], u_f_sub);

       face_project_r.Mult(u_sub, u_f_sub);
       u_r.SetSubVector(offsets_face[j], u_f_sub);
   }
   Vector f_l(dim*var_dim*n_face_pts), f_r(dim*var_dim*n_face_pts);
   for(int j = 0; j < dim*var_dim; j++)
   {
       f.GetSubVector(offsets[j], u_sub);

       face_project_l.Mult(u_sub, u_f_sub);
       f_l.SetSubVector(offsets_face[j], u_f_sub);

       face_project_r.Mult(u_sub, u_f_sub);
       f_r.SetSubVector(offsets_face[j], u_f_sub);
   }
   Vector f_com(dim*var_dim*n_face_pts);
   getVectorLFFlux(R_gas, gamm, dim, u_l, u_r, nor_face, f_com);
   subtract(f_com, f_l, f_l);
   subtract(f_com, f_r, f_r);
   
   Vector face_f_l(var_dim*n_face_pts), face_f_r(var_dim*n_face_pts);
   getFaceDotNorm(dim, f_l, nor_face, face_f_l);
   getFaceDotNorm(dim, f_r, nor_face, face_f_r);

   Vector f_sub(n_face_pts), f_dofs(dofs);
   for(int j = 0; j < var_dim; j++)
   {
       face_f_l.GetSubVector(offsets_face[j], u_f_sub);
       wts.Mult(u_f_sub, f_sub);
       face_project_l.MultTranspose(f_sub, f_dofs);

       face_f_r.GetSubVector(offsets_face[j], u_f_sub);
       wts.Mult(u_f_sub, f_sub);
       face_project_r.AddMultTranspose(f_sub, f_dofs, -1.0);

       b.SetSubVector(offsets[j], f_dofs);
   }
}




/*
 * Get Interaction flux using the local Lax Friedrichs Riemann  
 * solver, u1 is the left value and u2 the right value
 */
void getVectorLFFlux(const double R, const double gamm, const int dim, 
        const Vector &u1, const Vector &u2, const Vector &nor, Vector &f_com)
{
    int var_dim = dim + 2; 
    double Cv   = R/(gamm - 1);

    int num_pts = u1.Size()/var_dim;

    Vector fl(dim*var_dim*num_pts), fr(dim*var_dim*num_pts);
    getInvFlux(dim, u1, fl);
    getInvFlux(dim, u2, fr);
    add(0.5, fl, fr, f_com);

    double rho_L, E_L, vel_sq_L, T_L, a_L;
    double rho_R, E_R, vel_sq_R, T_R, a_R;
    Vector vel_L(dim);
    Vector vel_R(dim);

    Vector nor_in(dim), nor_dim(dim);
    double nor_l2;

    double vnl, vnr;
    double u_max;
    for(int p = 0; p < num_pts; p++)
    {
        rho_L = u1(p);
        
        for (int i = 0; i < dim; i++)
        {
            vel_L(i) = u1((1 + i)*num_pts + p)/rho_L;    
        }
        E_L   = u1((var_dim - 1)*num_pts + p);

        vel_sq_L = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vel_sq_L += pow(vel_L(i), 2) ;
        }

        T_L   = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
        a_L   = sqrt(gamm * R * T_L);

        rho_R = u2(p);
        for (int i = 0; i < dim; i++)
        {
            vel_R(i) = u2((1 + i)*num_pts + p)/rho_R;    
        }
        E_R   = u2((var_dim - 1)*num_pts + p);
    
        vel_sq_R = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vel_sq_R += pow(vel_R(i), 2) ;
        }
        T_R   = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
        a_R   = sqrt(gamm * R * T_R);

        for (int i = 0; i < dim; i++)
        {
            nor_in(i) = nor(i*num_pts + p);
        }
        nor_l2 = nor_in.Norml2();
        nor_dim.Set(1/nor_l2, nor_in);

        vnl   = 0.0; vnr = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vnl += vel_L[i]*nor_dim(i); 
            vnr += vel_R[i]*nor_dim(i); 
        }

        u_max = std::max(a_L + std::abs(vnl), a_R + std::abs(vnr));

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < var_dim; j++)
            {
                f_com((i*var_dim + j)*num_pts + p) += 
                    -0.5*u_max*nor_dim(i)*(u2(j*num_pts + p) - u1(j*num_pts + p)); 
            }
        }
    }
}


void getFaceDotNorm(int dim, const Vector &f, const Vector &nor_face, Vector &face_f)
{

    int var_dim = dim + 2;
    int num_pts = f.Size()/(dim*var_dim);

    face_f = 0.0;
    for(int p = 0; p < num_pts; p++)
    {
        for(int i = 0; i < dim; i++)
        {
            for(int j = 0; j < var_dim; j++)
            {
                face_f(j*num_pts + p) +=  f((i*var_dim + j)*num_pts + p)*nor_face(i*num_pts + p);
            }
        }
    }
}
