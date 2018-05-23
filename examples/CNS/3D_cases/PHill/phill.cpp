#include "mfem.hpp"
#include "cns.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.0021049677238282345; 
const double R_gas = 287;
const double   Pr  = 0.71;

//Run parameters
const char *mesh_file        =  "curved_test_c1.mesh";
const int    order           =  2;
const double t_final         =2000.0001 ;
const int    problem         =  0;
const int    ref_levels      =  0;

const bool   time_adapt      =  true ;
const double cfl             =  0.6 ;
const double dt_const        =  0.0001 ;
const int    ode_solver_type =  3; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       = 2000;

const bool   adapt           =  false;
const int    adapt_iter      =  200  ; // Time steps after which adaptation is done
const double tolerance       =  5e-4 ; // Tolerance for adaptation

//Source Term
const bool addSource         =  true;  // Add source term
const bool constForcing      = false;  // Is the forcing constant 
const bool mdotForcing       =  true;  // Forcing for constant mass flow 
const double fx              =  1.00 ; // Force x 
const double mdot_pres       =   6.0 ; // Prescribed mass flow 
      double mdot_fx         =  0.00 ; // Global variable for mass flow forcing 

//Restart parameters
const bool restart           =  true ;
const int  restart_freq      =  3000; // Create restart file after every 1000 time steps
const int  restart_cycle     =  3000; // File number used for restart

////////////////////////////////////////////////////////////////////////


// Velocity coefficient
void init_function(const Vector &x, Vector &v);
//void char_bnd_cnd(const Vector &x, Vector &v);
void wall_bnd_cnd(const Vector &x, Vector &v);
void wall_adi_bnd_cnd(const Vector &x, Vector &v);

void getVectorLFFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f);
void getFaceDotNorm(int dim, const Vector &f, const Vector &nor_face, Vector &face_f);

double getUMax(int dim, const Vector &u);

void getInvFlux(int dim, const Vector &u, Vector &f);
void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z,
        const CGSolver &M_solver, const Vector &u, 
        const Vector &b_aux_x, const Vector &b_aux_y, const Vector &b_aux_z,        
        Vector &aux_grad);
void getAuxVar(int dim, const Vector &u, Vector &aux_sol);

void ComputeWallForces(FiniteElementSpace &fes, VectorGridFunctionCoefficient &uD, 
                       VectorGridFunctionCoefficient &f_vis_D, 
                       const Array<int> &bdr, const double gamm, Vector &force);

void FilterWallVars(FiniteElementSpace &fes, GridFunction &uD, 
                    const Array<int> &bdr);

void getPRhoMin(int dim, const Vector &u, double &rho_min, double &p_min);

void getMDot_Faces(ParFiniteElementSpace &fes, vector<int> &int_faces, vector<int> &sh_faces);
void getMDot(ParFiniteElementSpace &fes, const vector<int> &int_faces, const vector<int> &sh_faces,
            ParGridFunction &u, double &mdot, double &m_area);
void GetUniqueX(const FiniteElementSpace &fes, vector<double> &x);
void Compute_PHill_Wall_Quant(FiniteElementSpace &fes, GridFunction &uD, GridFunction &f_vis_D, 
                    const double gamm, const vector<double> &x_uni,
                    vector<double> &p_wall, vector<double> &tau_wall, vector<int> &x_coun);
void Compute_Global_Average(MPI_Comm &comm, const vector<double> &local_quant, const vector<int> &local_count, 
                            vector<double> &global_quant);
void Compute_Time_Average(int ti_in, int ti,
                         const vector<double> &quant, vector<double> &time_average);
void Write_Time_Average(int ti, const vector<double> &x_uni, 
                        const vector<double> &inst_p_wall, const vector<double> &inst_tau_wall,
                        const vector<double> &p_wall,      const vector<double> &tau_wall);

void ComputeMflow(const ParGridFunction &uD, double &mflow);

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

   HypreSmoother M_prec;
   CGSolver M_solver;
                            
   ParGridFunction &u, &u_aux, &u_grad, &f_I, &f_V;

   Vector nor_face, wts;

   SparseMatrix   face_t_r;
   HypreParMatrix *glob_proj_l, *glob_proj_r;
                           
public:
   FE_Evolution(HypreParMatrix &_M, HypreParMatrix &_K_inv_x, HypreParMatrix &_K_inv_y, HypreParMatrix &_K_inv_z, 
                            HypreParMatrix &_K_vis_x, HypreParMatrix &_K_vis_y, HypreParMatrix &_K_vis_z,
                            ParGridFunction &u_,   ParGridFunction &u_aux, ParGridFunction &u_grad, 
                            ParGridFunction &f_I_, ParGridFunction &f_V_,
                            ParLinearForm &_b_aux_x, ParLinearForm &_b_aux_y, ParLinearForm &_b_aux_z, 
                            ParLinearForm &_b);

   void GetSize() ;

   CGSolver &GetMSolver() ;

   void Update();

   virtual void Mult(const ParGridFunction &x, ParGridFunction &y) const;

   void Source(const ParGridFunction &x, ParGridFunction &y) const;

   void GetMomentum(const ParGridFunction &x, double &Fx);

   void AssembleSharedFaceMatrices(const ParGridFunction &x) ;
   void getParEulerDGTranspose(const ParGridFunction &u_sol, const ParGridFunction &f_inv,
                                          Vector &b_nl) const;


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
   
   DG_FECollection fec(order, dim);

   /////////////////////////////////////////////////////////////////////////
   //Get the x locations on the wall where we need to average for 
   //periodic hill statistics
   FiniteElementSpace *fes_temp = new FiniteElementSpace(mesh, &fec, var_dim);

   vector<double> x_uni;
   GetUniqueX(*fes_temp, x_uni);

   delete fes_temp;
   /////////////////////////////////////////////////////////////////////////

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   double kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   fes = new ParFiniteElementSpace(pmesh, &fec, var_dim);

   ///////////////////////////////////////////////////////
   //Get Faces for calculating mass flow 
   vector<int> int_faces, sh_faces;
   getMDot_Faces(*fes, int_faces, sh_faces);
   ////////////////////////////////////////////////////////
   
   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   VectorFunctionCoefficient u0(var_dim, init_function);
   
   u_sol = new ParGridFunction(fes);

   double r_t; int r_ti;
   if (restart == false)
       u_sol->ProjectCoefficient(u0);
   else
       doRestart(restart_cycle, *pmesh, *u_sol, r_t, r_ti);

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

   ///////////////////////////////////////////////////
   // Define boundary terms
   // Each boundary should have its own marker since they are passed by ref
   Array<int> dir_bdr_wall(pmesh->bdr_attributes.Max());
   dir_bdr_wall    = 0; // Deactivate all boundaries
   dir_bdr_wall[0] = 1; // Activate wall bdy 

   VectorFunctionCoefficient u_wall_bnd(aux_dim, wall_bnd_cnd); // Defines wall boundary condition
   
   // Linear form representing the Euler boundary non-linear term
   b = new ParLinearForm(fes);
//   b->AddFaceIntegrator(
//      new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, var_dim, -1.0));
   b->AddBdrFaceIntegrator(
          new DG_Euler_NoSlip_Isotherm_Integrator(
              R_gas, gamm, u_vec, f_vec, u_wall_bnd, -1.0), dir_bdr_wall); 

   b->Assemble();


   getAuxVar(dim, *u_sol, *aux_sol);
   VectorGridFunctionCoefficient aux_vec(aux_sol);

   b_aux_x = new ParLinearForm(&fes_aux);
   b_aux_y = new ParLinearForm(&fes_aux);
   b_aux_x->AddBdrFaceIntegrator(
          new DG_CNS_Aux_Integrator(
              x_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
   b_aux_x->Assemble();

   b_aux_y->AddBdrFaceIntegrator(
           new DG_CNS_Aux_Integrator(
               y_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
   b_aux_y->Assemble();

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
       b_aux_z->AddBdrFaceIntegrator(
           new DG_CNS_Aux_Integrator(
               z_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
       b_aux_z->Assemble();
   }

   HypreParMatrix *M       = m->ParallelAssemble();
   HypreParMatrix *K_inv_x = k_inv_x->ParallelAssemble();
   HypreParMatrix *K_inv_y = k_inv_y->ParallelAssemble();

   HypreParMatrix *K_vis_x = k_vis_x->ParallelAssemble();
   HypreParMatrix *K_vis_y = k_vis_y->ParallelAssemble();

   HypreParMatrix *K_inv_z;
   HypreParMatrix *K_vis_z;

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
                            *b);
   }
   else if (dim == 2)
   {
       adv  = new FE_Evolution(*M, *K_inv_x, *K_inv_y, *K_inv_z,
                            *K_vis_x, *K_vis_y, *K_vis_z, 
                            *u_b, *aux_sol, *aux_grad, *f_I_b, *f_vis,
                            *b_aux_x, *b_aux_y, *b_aux_z, 
                            *b);
   }
   delete m, k_inv_x, k_inv_y, k_inv_z;
   delete    k_vis_x, k_vis_y, k_vis_z;


   getVisFlux(dim, *u_sol, *aux_grad, *f_vis);

   VectorGridFunctionCoefficient vis_vec(f_vis);
   VectorGridFunctionCoefficient aux_grad_vec(aux_grad);

   b->AddBdrFaceIntegrator(
           new DG_CNS_Vis_Isotherm_Integrator(
               R_gas, gamm, u_vec, vis_vec, aux_grad_vec, u_wall_bnd, mu, Pr, 1.0), dir_bdr_wall);

   b->Assemble();

   int ti_in; double t_in;
   if (restart == true)
   {
       ti_in = restart_cycle ;
       t_in  = r_t ;
   }
   else
   {
       ti_in = 0; t_in = 0;
   }

   {// Post process initially
       ti = ti_in; t = t_in;
       postProcess(*pmesh, order, gamm, R_gas, 
                   *u_sol, *aux_grad, ti, t);
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

   ofstream flo_file;
   flo_file.open("flow_char.dat");

   double glob_fx, glob_fy;
   double temp_fx, glob_vol_fx;
   double glob_tk, glob_ub, glob_vb, glob_wb, glob_vol;

   Vector forces(dim);
   ofstream force_file;
   force_file.open ("forces.dat");
   force_file << "Iteration \t dt \t time \t F_x \t F_y \n";

   vector<double> p_wall(x_uni.size()), tau_wall(x_uni.size()); 
   vector<int   > x_coun(x_uni.size()); 
   vector<double> glob_p_wall(x_uni.size()), glob_tau_wall(x_uni.size()); 
   vector<int   > glob_x_coun(x_uni.size()); 
   vector<double> time_p_wall(x_uni.size()), time_tau_wall(x_uni.size()); 

   bool done = false;
   for (ti = ti_in; !done; )
   {
      double tk = ComputeTKE(*fes, *u_sol);
      double ub, vb, wb, vol;
      ComputeUb(*u_sol, ub, vb, wb, vol);
      MPI_Allreduce(&tk,  &glob_tk,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&ub,  &glob_ub,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vb,  &glob_vb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&wb,  &glob_wb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vol, &glob_vol, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors


      double loc_m_dot, loc_m_area ;
      getMDot(*fes, int_faces, sh_faces, *u_sol, loc_m_dot, loc_m_area);
      double glob_m_dot, glob_m_area;
      MPI_Allreduce(&loc_m_dot,   &glob_m_dot,   1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&loc_m_area,  &glob_m_area,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors

      double loc_mbulk, glob_mbulk, mbulk_0;
      ComputeMflow(*u_sol, loc_mbulk);
      MPI_Allreduce(&loc_mbulk,   &glob_mbulk,   1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      double mdot_0, fx_0;
      if (ti < ti_in + 1)
      {
          mdot_fx = fx;
          fx_0    = mdot_fx;

//          mdot_0  = glob_m_dot;
          mdot_0  = glob_mbulk/9.;
      }
      else
      {
//          mdot_fx = fx_0 + 0.3*(mdot_pres - 2*glob_m_dot + mdot_0)/(glob_m_area*dt);
          mdot_fx = fx_0 + 0.3*(mdot_pres - 2*(glob_mbulk/9.) + mdot_0)/(glob_m_area*dt);
//          mdot_0  = glob_m_dot;
          mdot_0  = glob_mbulk/9.;
          fx_0    = mdot_fx;

      }

      Step(); // Step in time

      done = (t >= t_final - 1e-8*dt);

      if ((ti % 25 == 0) && (myid == 0)) // Check time
      {
          chrono.Stop();
          cout << "25 Steps took "<< chrono.RealTime() << " s "<< endl;

          chrono.Clear();
          chrono.Start();
      }
    
      if (done || ti % vis_steps == 0) // Visualize
      {
          getAuxGrad(dim, *K_vis_x, *K_vis_y, *K_vis_z, 
          adv->GetMSolver(), *u_b,
          *b_aux_x, *b_aux_y, *b_aux_z, 
          *aux_grad);

          postProcess(*pmesh, order, gamm, R_gas, 
                   *u_sol, *aux_grad, ti, t);

      }
      
      if (done || ti % restart_freq == 0) // Write restart file 
      {
          writeRestart(*pmesh, *u_sol, ti, t);
      }

      Compute_PHill_Wall_Quant(*fes, *u_sol, *f_vis, 
                               gamm, x_uni, p_wall, tau_wall, x_coun);

      Compute_Global_Average(comm, p_wall,   x_coun, glob_p_wall);
      Compute_Global_Average(comm, tau_wall, x_coun, glob_tau_wall);

      int c_ti = 1;
      if (restart == true)
          c_ti = restart_cycle + 1;

      Compute_Time_Average(c_ti, ti, glob_p_wall,   time_p_wall);
      Compute_Time_Average(c_ti, ti, glob_tau_wall, time_tau_wall);

      if (done || ti % vis_steps == 0) // 
      {
          if (myid == 0)
              Write_Time_Average(ti, x_uni, glob_p_wall, glob_tau_wall, time_p_wall, time_tau_wall);
      }

      adv->GetMomentum(*y_t, temp_fx);
      MPI_Allreduce(&temp_fx, &glob_vol_fx, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors

      if (myid == 0)
      {
          glob_ub = glob_ub/glob_vol;
          glob_vb = glob_vb/glob_vol;
          glob_wb = glob_wb/glob_vol;
          flo_file << setprecision(6) << ti << "\t" << t << "\t" 
              << glob_ub << "\t" << glob_vol_fx << "\t" << glob_vb << "\t" << glob_wb 
//              << "\t"    << glob_m_dot << "\t" << mdot_fx<< endl;
              << "\t"    << glob_mbulk/9. << "\t" << glob_m_dot << "\t" << mdot_fx<< endl;
      }

      ComputeWallForces(*fes, *u_sol, *f_vis, dir_bdr_wall, gamm, forces);
      double wall_fx = forces(0), wall_fy = forces(1);
      MPI_Allreduce(&wall_fx, &glob_fx, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&wall_fy, &glob_fy, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors

      if (myid == 0)
      {
          force_file << ti << "\t" <<  dt_real << "\t" << t << "\t" 
                     << glob_fx << "\t" << glob_fy  << "\t" << mdot_fx <<  endl;
      }
      
//      if (ti % 50 == 0) // Filter every 10 steps 
//          FilterWallVars(*fes, *u_sol, dir_bdr_wall);
  
   }
   flo_file.close();
   force_file.close();
 
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

    double loc_rho_min, glob_rho_min; 
    double loc_p_min, glob_p_min; 
    getPRhoMin(dim, *u_sol, loc_rho_min, loc_p_min); 
    MPI_Allreduce(&loc_rho_min, &glob_rho_min, 1, MPI_DOUBLE, MPI_MIN, comm); 
    MPI_Allreduce(&loc_p_min,   &glob_p_min,   1, MPI_DOUBLE, MPI_MIN, comm); 

    if (myid == 0)
    {
        cout << setprecision(6) << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
            t << ", max_speed " << glob_u_max << ", rho_min "<< glob_rho_min << ", p_min "<< glob_p_min << ", fes_size " << fes->GlobalTrueVSize() << endl;
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
                            ParLinearForm &_b)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), K_inv_z(_K_inv_z),
                            K_vis_x(_K_vis_x), K_vis_y(_K_vis_y), K_vis_z(_K_vis_z), 
                            u(u_), u_aux(u_aux_), u_grad(u_grad_), f_I(f_I_), f_V(f_V_), 
                            b_aux_x(_b_aux_x), b_aux_y(_b_aux_y), b_aux_z(_b_aux_z), b(_b)
{
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);
   
   AssembleSharedFaceMatrices(u);
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
    
    b_aux_x.Assemble();
    b_aux_y.Assemble();
    if (dim == 3)
    {
        b_aux_z.Assemble();
    }


    getAuxGrad(dim, K_vis_x, K_vis_y, K_vis_z, M_solver, x, 
            b_aux_x, b_aux_y, b_aux_z,
            u_grad);
    getVisFlux(dim, x, u_grad, f_V);

    b.Assemble();
    Vector b_nl(x.Size());
    getParEulerDGTranspose(u, f_I, b_nl);

    add(b, -1, b_nl, b);

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

    if (addSource == true)
        Source(x, y);  // Add source to y
}

/*
 * Any source terms that need to be added to the RHS
 * du/dt + dF/dx = S
 * For example if we have a forcing term
 * Since we'll call this at the end of all steps in Mult no Jacobians need be considered
 */
void FE_Evolution::Source(const ParGridFunction &x, ParGridFunction &y) const
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    ParFiniteElementSpace *fes = x.ParFESpace();
    MPI_Comm comm              = fes->GetComm();

    double ub, vb, wb, vol;
    ComputeUb(x, ub, vb, wb, vol);
    double glob_ub, glob_vol;
    MPI_Allreduce(&ub,  &glob_ub,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors
    MPI_Allreduce(&vol, &glob_vol, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors

    ub = glob_ub/glob_vol;

    int dim     = x.Size()/K_inv_x.GetNumRows() - 2;
    int var_dim = dim + 2;

    int offset  = K_inv_x.GetNumRows();
   
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

    Vector s(offset), y_temp(var_dim*offset);
    for(int i = 0; i < var_dim; i++)
    {
        s = 0.0;
        double forcing = 0.0;
        if (constForcing == true)
            forcing = fx;
        else if (mdotForcing == true)
            forcing = mdot_fx;
        if (i == 1)
            s = forcing; 
        else if (i == var_dim - 1)
            s = forcing*ub;
        y_temp.SetSubVector(offsets[i], s);
    }

    add(y_temp, y, y);
}

// Generate matrices for projecting data to faces
// This should in theory speed up calculations
void FE_Evolution::AssembleSharedFaceMatrices(const ParGridFunction &x) 
{
   ParFiniteElementSpace *fes_temp = x.ParFESpace();
   ParMesh *pmesh = fes_temp->GetParMesh();
   MPI_Comm comm              = fes_temp->GetComm();

   int dim     = pmesh->SpaceDimension();
   int var_dim = dim + 2;

   /////////////////////////////////////////////
   //Defined only for getting number of face pts
   DG_Interface_FECollection fec_hdg(order, dim);
   ParFiniteElementSpace fes_hdg(pmesh, &fec_hdg);

   int glob_nFacePts = fes_hdg.GlobalTrueVSize();
   /////////////////////////////////////////////
   
   ParFiniteElementSpace fes(pmesh, fes_temp->FEColl());
   fes.ExchangeFaceNbrData();

   const FiniteElement *el1, *el2;
   FaceElementTransformations *T;

   int ndof1, ndof2;
   int order;

   double w; // weight
   double alpha =  -1.0;

   Vector shape1, shape2;

   Array<int> vdofs, vdofs2;

   int dofs       = fes.GetVSize();

   Vector nor(dim);

   int nfaces = pmesh->GetNumFaces(); // Total number of faces on this processor (including shared faces)

   int n_sh_faces = pmesh->GetNSharedFaces(); // Shared faces number

   Array<int> faceCounter(nfaces), facePts(nfaces); // Get face no and num of face points for each face
   for (int i = 0; i < nfaces; i++)
   {
       T = pmesh->GetInteriorFaceTransformations(i);

       facePts[i] = 0;

       if(T != NULL)
       {
           el1 = fes.GetFE(T -> Elem1No);
           el2 = fes.GetFE(T -> Elem2No);

           const IntegrationRule *ir ;

           int order = 2*std::max(el1->GetOrder(), el2->GetOrder());
       
           ir = &IntRules.Get(T->FaceGeom, order);
           
           facePts[i]   = ir->GetNPoints();
       } // If loop
   }

   for (int i = 0; i < n_sh_faces; i++)
   {
       T = pmesh->GetSharedFaceTransformations(i);
 
       el1 = fes.GetFE(T -> Elem1No);
       el2 = fes.GetFaceNbrFE(T -> Elem2No);
  
       fes.GetElementVDofs (T -> Elem1No, vdofs);
       fes.GetFaceNbrElementVDofs (T -> Elem2No, vdofs2);

       const IntegrationRule *ir ;

       int order = 2*std::max(el1->GetOrder(), el2->GetOrder());
       
       ir = &IntRules.Get(T->FaceGeom, order);
       
       facePts[pmesh->GetSharedFace(i)] = ir->GetNPoints();
   }
   int nFacePts = facePts.Sum();
   
   nor_face.SetSize(dim*nFacePts); wts.SetSize(nFacePts); 

   faceCounter[0] = facePts[0]; // Running counter for number of face points
   for (int i = 1; i < nfaces; i++)
   {
       faceCounter[i] = faceCounter[i - 1] + facePts[i];   
   }

   int nbr_size = fes.GetFaceNbrVSize();
   int loc_size = fes.GetVSize(); 

   SparseMatrix face_proj_l(nFacePts, loc_size + nbr_size); // Second subscript refers to internal dofs and shared dofs
   SparseMatrix face_proj_r(nFacePts, loc_size + nbr_size);
   
   SparseMatrix temp_face(nFacePts, loc_size);
   face_t_r.Swap(temp_face);

//   double eps = std::numeric_limits<double>::epsilon();
   double eps = 1E-12; 

   for (int i = 0; i < nfaces; i++)
   {
       T = pmesh->GetInteriorFaceTransformations(i);

       if(T != NULL)
       {
           fes.GetElementVDofs (T -> Elem1No, vdofs);
           fes.GetElementVDofs (T -> Elem2No, vdofs2);

           el1 = fes.GetFE(T -> Elem1No);
           el2 = fes.GetFE(T -> Elem2No);

           const IntegrationRule *ir ;

           int order = 2*std::max(el1->GetOrder(), el2->GetOrder());
       
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
 
              ndof1   = el1->GetDof();
              ndof2   = el2->GetDof();

              shape1.SetSize(ndof1);
              shape2.SetSize(ndof2);
   
              el1->CalcShape(eip1, shape1);
              el2->CalcShape(eip2, shape2);
    
              for(int j=0; j < shape1.Size(); j++)
              {
                  if (std::abs(shape1(j)) < eps )
                      shape1(j) = 0.0;
              }
              for(int j=0; j < shape2.Size(); j++)
              {
                  if (std::abs(shape2(j)) < eps )
                      shape2(j) = 0.0;
              }

              CalcOrtho(T->Face->Jacobian(), nor);

              int face_coun;
              if (i == 0)
                  face_coun = p;
              else
                  face_coun = faceCounter[i - 1] + p;

              face_proj_l.SetRow(face_coun, vdofs,  shape1);
              face_proj_r.SetRow(face_coun, vdofs2, shape2);

              face_t_r.SetRow(face_coun, vdofs2, shape2);

              for(int i=0; i < dim; i++)
                  nor_face(i*nFacePts + face_coun) = nor(i);

              wts(face_coun) = ip.weight;

         }// p loop 

       } // If loop

   }

   for (int i = 0; i < n_sh_faces; i++)
   {
       T = pmesh->GetSharedFaceTransformations(i);
 
       el1 = fes.GetFE(T -> Elem1No);
       el2 = fes.GetFaceNbrFE(T -> Elem2No);
  
       fes.GetElementVDofs        (T -> Elem1No, vdofs);
       fes.GetFaceNbrElementVDofs (T -> Elem2No, vdofs2);
   
       for (int j = 0; j < vdofs2.Size(); j++)
       {
           vdofs2[j] += loc_size; // Neighbour dofs start after local dofs       
       }

       const IntegrationRule *ir ;

       int order = 2*std::max(el1->GetOrder(), el2->GetOrder());
       
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
 
          ndof1   = el1->GetDof();
          ndof2   = el2->GetDof();

          shape1.SetSize(ndof1);
          shape2.SetSize(ndof2);
   
          el1->CalcShape(eip1, shape1);
          el2->CalcShape(eip2, shape2);
    
          for(int j=0; j < shape1.Size(); j++)
          {
              if (std::abs(shape1(j)) < eps )
                  shape1(j) = 0.0;
          }
          for(int j=0; j < shape2.Size(); j++)
          {
              if (std::abs(shape2(j)) < eps )
                  shape2(j) = 0.0;
          }
              
          CalcOrtho(T->Face->Jacobian(), nor);

          int face_coun;
          int globFaceNo = pmesh->GetSharedFace(i);
          if (globFaceNo == 0)
              face_coun = p;
          else
              face_coun = faceCounter[globFaceNo - 1] + p;

          face_proj_l.SetRow(face_coun, vdofs,  shape1);
          face_proj_r.SetRow(face_coun, vdofs2, shape2);
          
          for(int i=0; i < dim; i++)
                  nor_face(i*nFacePts + face_coun) = nor(i);

          wts(face_coun) = ip.weight;
              
       } // p loop
      
   }

   face_proj_l.Finalize();
   face_proj_r.Finalize();

   face_t_r.Finalize();

   //Now generate the Hypre matrices


   // handle the case when 'a' contains offdiagonal
   int lvsize = fes.GetVSize();
   const HYPRE_Int *face_nbr_glob_ldof = fes.GetFaceNbrGlobalDofMap();
   HYPRE_Int ldof_offset = fes.GetMyDofOffset();

   Array<HYPRE_Int> glob_J_l(face_proj_l.NumNonZeroElems());
   int *J_l = face_proj_l.GetJ();

   for (int i = 0; i < glob_J_l.Size(); i++)
   {
      if (J_l[i] < lvsize)
      {
         glob_J_l[i] = J_l[i] + ldof_offset;
      }
      else
      {
         glob_J_l[i] = face_nbr_glob_ldof[J_l[i] - lvsize];
      }
   }

   Array<HYPRE_Int> glob_J_r(face_proj_r.NumNonZeroElems());
   int *J_r = face_proj_r.GetJ();

   for (int i = 0; i < glob_J_r.Size(); i++)
   {
      if (J_r[i] < lvsize)
      {
         glob_J_r[i] = J_r[i] + ldof_offset;
      }
      else
      {
         glob_J_r[i] = face_nbr_glob_ldof[J_r[i] - lvsize];
      }
   }

   HYPRE_Int ldof = nFacePts; 
   Array<HYPRE_Int> dof_off, *offsets[1] = { &dof_off };

   pmesh->GenerateOffsets(1, &ldof, offsets);

   glob_proj_l = new HypreParMatrix(fes.GetComm(), nFacePts, glob_nFacePts,
                            fes.GlobalVSize(), face_proj_l.GetI(), glob_J_l,
                            face_proj_l.GetData(), dof_off,
                            fes.GetDofOffsets());

   glob_proj_r = new HypreParMatrix(fes.GetComm(), nFacePts, glob_nFacePts,
                            fes.GlobalVSize(), face_proj_r.GetI(), glob_J_r,
                            face_proj_r.GetData(), dof_off,
                            fes.GetDofOffsets());
}


void FE_Evolution::getParEulerDGTranspose(const ParGridFunction &u_sol, const ParGridFunction &f_inv,
                                          Vector &b_nl) const
{
   int var_dim    = u_sol.VectorDim(); 
   int dim        = var_dim - 2;
   int n_face_pts = wts.Size();
   int dofs       = u_sol.Size()/var_dim;

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
       u_sol.GetSubVector(offsets[j], u_sub);

       glob_proj_l->Mult(u_sub, u_f_sub);
       u_l.SetSubVector(offsets_face[j], u_f_sub);
       
       glob_proj_r->Mult(u_sub, u_f_sub);
       u_r.SetSubVector(offsets_face[j], u_f_sub);
   }
   Vector f_l(dim*var_dim*n_face_pts), f_r(dim*var_dim*n_face_pts);
   for(int j = 0; j < dim*var_dim; j++)
   {
       f_inv.GetSubVector(offsets[j], u_sub);

       glob_proj_l->Mult(u_sub, u_f_sub);
       f_l.SetSubVector(offsets_face[j], u_f_sub);

       glob_proj_r->Mult(u_sub, u_f_sub);
       f_r.SetSubVector(offsets_face[j], u_f_sub);
   }
       
   Vector f_com(dim*var_dim*n_face_pts);
   Vector f_left(dim*var_dim*n_face_pts), f_rght(dim*var_dim*n_face_pts) ;
   getVectorLFFlux(R_gas, gamm, dim, u_l, u_r, nor_face, f_com);
   subtract(f_com, f_l, f_left);
   subtract(f_com, f_r, f_rght);

   Vector face_f_l(var_dim*n_face_pts), face_f_r(var_dim*n_face_pts);
   getFaceDotNorm(dim, f_left, nor_face, face_f_l);
   getFaceDotNorm(dim, f_rght, nor_face, face_f_r);

   Vector f_sub(n_face_pts), temp1(dofs), temp2(dofs), f_dofs(dofs);
   for(int j = 0; j < var_dim; j++)
   {
       face_f_l.GetSubVector(offsets_face[j], u_f_sub);
       for(int pt = 0;  pt < n_face_pts; pt++)
           f_sub(pt) = wts(pt)*u_f_sub(pt);
       glob_proj_l->MultTranspose(f_sub, temp1);

       face_f_r.GetSubVector(offsets_face[j], u_f_sub);
       for(int pt = 0;  pt < n_face_pts; pt++)
           f_sub(pt) = wts(pt)*u_f_sub(pt);
       face_t_r.MultTranspose(f_sub, temp2);

       subtract(temp1, temp2, f_dofs);

       b_nl.SetSubVector(offsets[j], f_dofs);
   }
}


/*
 * We'll assume a structured grid in terms of shape for now to make it easier
 */
void getMDot(ParFiniteElementSpace &fes, const vector<int> &int_faces, const vector<int> &sh_faces,
             ParGridFunction &u, double &mdot, double &m_area)
{
   double eps = 1E-11;

   fes.ExchangeFaceNbrData();

   ParMesh *mesh = fes.GetParMesh();
   int   dim     = mesh->Dimension();
   int var_dim   = dim + 2;
   int meshNE    = mesh->GetNE();

   int nfaces     = int_faces.size();
   int n_sh_faces = sh_faces.size();

   int num_procs, myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Vector nor(dim), vals(var_dim), vel(dim);

   double int_area = 0.0, m_int = 0.0;
   double sh_area  = 0.0, m_sh  = 0.0;
   for(int j = 0; j < nfaces; j++)
   {
       FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(int_faces.at(j)); 
       if (tr != NULL)
       {
           const FiniteElement *el1       = fes.GetFE(tr -> Elem1No);
           int order                      = 2*el1->GetOrder();

           const IntegrationRule *ir      = &IntRules.Get(tr -> FaceGeom, order);
           for (int p = 0; p < ir->GetNPoints(); p++)
           {
               const IntegrationPoint &ip     = ir->IntPoint(p);
    
               IntegrationPoint eip;
               tr->Loc1.Transform(ip, eip);
    
               tr->Face->SetIntPoint(&ip);
               CalcOrtho(tr->Face->Jacobian(), nor);

               u.GetVectorValue(tr->Elem1No, eip, vals);
               Vector vel(dim);
               double rho = vals(0);
               double v_sq  = 0.0;
               for (int j = 0; j < dim; j++)
               {
                   vel(j) = vals(1 + j)/rho;
               }

               m_int    += vel(0)*std::abs(nor(0))*ip.weight;
               int_area += std::abs(nor(0))*ip.weight;
           }
       }
   }
   for (int j = 0; j < n_sh_faces; j++)
   {
       FaceElementTransformations *tr = mesh->GetSharedFaceTransformations(sh_faces.at(j)); 
       if (tr != NULL)
       {
           const FiniteElement *el1       = fes.GetFE(tr -> Elem1No);
           int order                      = 2*el1->GetOrder();

           const IntegrationRule *ir      = &IntRules.Get(tr -> FaceGeom, order);
           for (int p = 0; p < ir->GetNPoints(); p++)
           {
               const IntegrationPoint &ip     = ir->IntPoint(p);
    
               IntegrationPoint eip;
               tr->Loc1.Transform(ip, eip);
    
               tr->Face->SetIntPoint(&ip);
               CalcOrtho(tr->Face->Jacobian(), nor);
    
               u.GetVectorValue(tr->Elem1No, eip, vals);
               Vector vel(dim);
               double rho = vals(0);
               double v_sq  = 0.0;
               for (int j = 0; j < dim; j++)
               {
                   vel(j) = vals(1 + j)/rho;
               }

               m_sh    += vel(0)*std::abs(nor(0))*ip.weight;
               sh_area += std::abs(nor(0))*ip.weight;
           }
       }
   }

   mdot   = m_int + m_sh;
   m_area = int_area + sh_area;
//   cout << int_area << "\t" << m_int << "\t" << sh_area << "\t" << m_sh << endl;

}



/*
 * We'll assume a structured grid in terms of shape for now to make it easier
 */
void getMDot_Faces(ParFiniteElementSpace &fes, vector<int> &int_faces, vector<int> &sh_faces)
{
   double eps = 1E-11;

   ParMesh *mesh = fes.GetParMesh();

   fes.    ExchangeFaceNbrData();
   mesh -> ExchangeFaceNbrData();

   int   dim     = mesh->Dimension();
   int meshNE    = mesh->GetNE();

   int nfaces     = mesh->GetNumFaces();
   int n_sh_faces = mesh->GetNSharedFaces();

   int num_procs, myid;
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   Vector loc(dim), nor(dim);
   for(int j = 0; j < nfaces; j++)
   {
       FaceElementTransformations *tr = mesh->GetInteriorFaceTransformations(j); 
       if (tr != NULL)
       {
           const FiniteElement *el1       = fes.GetFE(tr -> Elem1No);
           int order                      = 2*el1->GetOrder();

           const IntegrationRule *ir      = &IntRules.Get(tr -> FaceGeom, order);
           const IntegrationPoint &ip     = ir->IntPoint(0);

           IntegrationPoint eip;
           tr->Loc1.Transform(ip, eip);

           tr->Elem1->Transform(eip, loc);

           if (std::abs(loc[0]) < eps)
           {
               int_faces.push_back(j);
           }
       }
   }
   for (int j = 0; j < n_sh_faces; j++)
   {
       FaceElementTransformations *tr = mesh->GetSharedFaceTransformations(j); 
       if (tr != NULL)
       {
           const FiniteElement *el1       = fes.GetFE(tr -> Elem1No);
           int order                      = 2*el1->GetOrder();

           const IntegrationRule *ir      = &IntRules.Get(tr -> FaceGeom, order);
           const IntegrationPoint &ip     = ir->IntPoint(0);

           IntegrationPoint eip;
           tr->Loc1.Transform(ip, eip);

           tr->Elem1->Transform(eip, loc);
           
           if (std::abs(loc[0]) < eps)
           {
               sh_faces.push_back(j);
           }
       }
   }

}


/*
 * We'll assume a structured grid in terms of shape for now to make it easier
 * This is for the Periodic hill
 * the spanwise directions
 * The algorithm is sensitive to Tolerance levels
 * We simply get all unique x values 
 */
void GetUniqueX(const FiniteElementSpace &fes, vector<double> &x_uni)
{
   double eps = 5E-05;

   const FiniteElement *el;
   FaceElementTransformations *T;

   Mesh *mesh = fes.GetMesh();
   int   dim  = mesh->Dimension();
   int meshNE = mesh->GetNE();

   Vector loc(dim);
   std::vector<double> x, y;

   for (int i = 0; i < fes.GetNBE(); i++)
   {
       T = mesh->GetBdrFaceTransformations(i);
 
       el = fes.GetFE(T -> Elem1No);
   
       const IntegrationRule *ir ;
       int order;
       
       order = 2*el->GetOrder();
       
       ir = &IntRules.Get(T->FaceGeom, order);

       for (int p = 0; p < ir->GetNPoints(); p++)
       {
          const IntegrationPoint &ip = ir->IntPoint(p);
          IntegrationPoint eip;
          T->Loc1.Transform(ip, eip);
    
          T->Face->SetIntPoint(&ip);
          T->Elem1->SetIntPoint(&eip);
           
          T->Elem1->Transform(eip, loc);

          if (loc(1) < 1 + eps) // Only measure at lower wall
          {
              x.push_back(loc(0));
              y.push_back(loc(1));
          }
       }
   }

   std::sort(x.begin(), x.end(), std::less<double>());
   double xMin = *(std::min_element(x.begin(), x.end()));

   std::vector<double> x_unique;
   x_unique.push_back(x.at(0));

   int xCoun = 0; //Count the number of co-ordinates in y direction on which we'll average
   for(int i = 0; i < x.size(); i++)
   {
       double last_x = x_unique.back(); // Last unique y added
       if (abs(x.at(i) - last_x) > eps)
           x_unique.push_back(x[i]);

       if (abs(x.at(i) - xMin) < eps)
           xCoun++;
   }

   x_uni = x_unique;
}


void Compute_PHill_Wall_Quant(FiniteElementSpace &fes, GridFunction &uD, GridFunction &f_vis_D, 
                    const double gamm, const vector<double> &x_uni,
                    vector<double> &p_wall, vector<double> &tau_wall, vector<int> &x_coun)
{
   double eps = 5E-05;

   const FiniteElement *el;
   FaceElementTransformations *T;

   Mesh *mesh = fes.GetMesh();
   int   dim  = mesh->Dimension();

   Vector nor(dim), nor_dim(dim), loc(dim);

   Vector vals, vis_vals;

   for (int j = 0; j < x_uni.size(); j++)
   {
       p_wall.  at(j) = 0;
       tau_wall.at(j) = 0;
       x_coun.  at(j) = 0;
   }

   for (int i = 0; i < fes.GetNBE(); i++)
   {
       T = mesh->GetBdrFaceTransformations(i);
 
       el = fes.GetFE(T -> Elem1No);
   
       dim = el->GetDim();
      
       const IntegrationRule *ir ;
       int order;
       
       order = 2*el->GetOrder();
       
       ir = &IntRules.Get(T->FaceGeom, order);

       for (int p = 0; p < ir->GetNPoints(); p++)
       {
          const IntegrationPoint &ip = ir->IntPoint(p);
          IntegrationPoint eip;
          T->Loc1.Transform(ip, eip);
    
          T->Face->SetIntPoint(&ip);
          T->Elem1->SetIntPoint(&eip);

          CalcOrtho(T->Face->Jacobian(), nor);
               
          T->Elem1->Transform(eip, loc);

//          cout << i << "\t" << p << "\t" << loc(0) << "\t" << loc(1) << "\t" << loc(2) << endl; 
          double nor_l2 = nor.Norml2();
          nor_dim.Set(1/nor_l2, nor);
  
          int x_uni_coun; // Identify subscript of x_uni where present point matches
          for (int j = 0; j < x_uni.size(); j++)
          {
              if (std::abs(x_uni.at(j) - loc(0)) < eps)
                  x_uni_coun = j;
//                  cout << i << "\t" << p << "\t" << j << "\t" << loc(0) << "\t" << loc(1) << endl; 
          }


          if (loc(1) < 1 + eps) // Only measure at lower wall
          {
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

              f_vis_D.GetVectorValue(T->Elem1No, eip, vis_vals);
            
              double tau[dim][dim];
              for(int i = 0; i < dim ; i++)
                  for(int j = 0; j < dim ; j++) tau[i][j] = vis_vals[i*(dim + 2) + 1 + j];

              p_wall.at(x_uni_coun)   += pres;
              tau_wall.at(x_uni_coun) += tau[0][1];
              x_coun.at(x_uni_coun)   += 1;
    
          } // If loop

       } // p loop
   } // NBE loop

}

// Global sum across MPI processes
void Compute_Global_Average(MPI_Comm &comm, const vector<double> &local_quant, const vector<int> &local_count, 
                            vector<double> &global_quant)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    vector<double> global;
    int node_coun = local_quant.size();

    for(int i = 0; i < node_coun; i++)
    {
        double loc  = local_quant.at(i);    
        double glob;

        int loc_cn  = local_count.at(i);
        int glob_cn;

        MPI_Allreduce(&loc,    &glob,    1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors
        MPI_Allreduce(&loc_cn, &glob_cn, 1, MPI_INT   , MPI_SUM, comm); // Get global u_max across processors

        global.push_back(glob/glob_cn);
    }

    global_quant = global;
}

void Compute_Time_Average(int ti_in, int ti,
                         const vector<double> &quant, vector<double> &time_average)
{
    if (ti == ti_in)
        time_average = quant;
    else
    {
        int node_coun = quant.size();    
        for(int i = 0; i < node_coun; i++)
        {
            time_average.at(i) = time_average.at(i)*(ti - ti_in) + quant.at(i);

            time_average.at(i) = time_average.at(i)/(ti - ti_in + 1);
        }
    }
}

void Write_Time_Average(int ti, const vector<double> &x_uni, 
                        const vector<double> &inst_p_wall, const vector<double> &inst_tau_wall,
                        const vector<double> &p_wall,      const vector<double> &tau_wall)
{
    std::ostringstream oss;
    oss << std::setw(7) << std::setfill('0') << ti;

    std::string f_name = "u_mean_" + oss.str();

    ofstream f_file;
    f_file.open(f_name.c_str());

    int node_coun = x_uni.size();

    for(int i = 0; i < node_coun; i++)
    {
        f_file << setprecision(6) << x_uni.at(i) << "\t"  
               << inst_p_wall.at(i)  << "\t" << inst_tau_wall.at(i) << "\t" 
               << p_wall.at(i)       << "\t" << tau_wall.at(i) << endl; 
    }
    f_file.close();

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
    
        for(int i = 0; i < dim; i++) vel[i] = u[i*(1 + offset) + j]/rho;

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

// Get max signal speed 
void getPRhoMin(int dim, const Vector &u, double &rho_min, double &p_min)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}

    int offset = u.Size()/var_dim;

    rho_min = u[0];
    p_min   = 1.0E16;

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

        double p = rho*R_gas*T; 

        rho_min = min(rho_min, rho);
        p_min   = min(p_min,   p);
    }

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

   if (dim == 3)
   {
       double rho, u1, u2, u3, p;
       /*
        * Discrete filter operators for large-eddy simulation using high-order spectral difference methods
        */
       rho = 1;

       double amp = 0.1;
       
       u1  =  4.7;
       u2  = 0.0;
       u3  = 0.0;
       p   =2500 ;

       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = rho * u3;                //rho * v
       v(4) = p/((gamm - 1)*rho) + 0.5*rho*v_sq;
   }
   else if (dim == 2)
   {
       double rho, u1, u2, u3, p;
       /*
        * Discrete filter operators for large-eddy simulation using high-order spectral difference methods
        */
       rho = 1;

       double amp = 0.1;
       
       u1  =  4.5;
       u2  = 0.0;
       u3  = 0.0;
       p   =2500 ;

       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/((gamm - 1)*rho) + 0.5*rho*v_sq;
   }

}


void wall_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 3)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.0;  // z velocity 
       v(3) = 8.710801393728223;  // Temp 
   }
   else if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 8.710801393728223;  // Temp 
   }

}




void FE_Evolution::GetMomentum(const ParGridFunction &x, double &Fx) 
{
    int dim = x.Size()/K_inv_x.GetNumRows() - 2;
    int var_dim = dim + 2;

    Vector y_temp, y;
    y_temp.SetSize(x.Size()); 
    y.SetSize(x.Size()); 

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

    const FiniteElementSpace *fes = x.FESpace();
    Mesh *mesh                    = fes->GetMesh();

    const FiniteElement *el;
 
    Fx  = 0.0;
    for (int i = 0; i < fes->GetNE(); i++)
    {
        ElementTransformation *T  = fes->GetElementTransformation(i);
        el = fes->GetFE(i);
 
        dim = el->GetDim();
 
        int dof = el->GetDof();
        Array<int> vdofs;
        fes->GetElementVDofs(i, vdofs);
 
        const IntegrationRule *ir ;
        int   order;
 
        order = 2*el->GetOrder() + 1;
        ir    = &IntRules.Get(el->GetGeomType(), order);
 
        for (int p = 0; p < ir->GetNPoints(); p++)
        {
            const IntegrationPoint &ip = ir->IntPoint(p);
            T->SetIntPoint(&ip);
 
            Fx  += ip.weight*T->Weight()*(y[vdofs[dof + p]]);
        }
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

    double vsl, vsr;
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

        vsl   = sqrt(vel_sq_L); vsr = sqrt(vel_sq_R);

        u_max = std::max(a_L + std::abs(vsl), a_R + std::abs(vsr));

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
    for(int i = 0; i < dim; i++)
    {
        for(int j = 0; j < var_dim; j++)
        {
            for(int p = 0; p < num_pts; p++)
            {
                face_f(j*num_pts + p) +=  f((i*var_dim + j)*num_pts + p)*nor_face(i*num_pts + p);
            }
        }
    }
}




// Get bulk flow 
void ComputeMflow(const ParGridFunction &uD, double &mflow)
{
   const FiniteElementSpace *fes = uD.FESpace();
   Mesh *mesh                    = fes->GetMesh();
   
   const FiniteElement *el;

   int dim;

   mflow  = 0.0;
   for (int i = 0; i < fes->GetNE(); i++)
   {
       ElementTransformation *T  = fes->GetElementTransformation(i);
       el = fes->GetFE(i);

       dim = el->GetDim();

       int dof = el->GetDof();
       Array<int> vdofs;
       fes->GetElementVDofs(i, vdofs);

       const IntegrationRule *ir ;
       int   order;

       order = 2*el->GetOrder() + 1;
       ir    = &IntRules.Get(el->GetGeomType(), order);

       for (int p = 0; p < ir->GetNPoints(); p++)
       {
           const IntegrationPoint &ip = ir->IntPoint(p);
           T->SetIntPoint(&ip);

           double rho    = uD[vdofs[p]];
           double irho   = 1.0/rho; 
       
           mflow  += ip.weight*T->Weight()*(uD[vdofs[1*dof + p]]);
       }
   }

}


