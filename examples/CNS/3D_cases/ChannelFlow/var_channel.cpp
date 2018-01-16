#include "mfem.hpp"
#include "cns.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.00031746031746031746; 
const double R_gas = 287;
const double   Pr  = 0.71;

//Run parameters
const char *mesh_file        =  "channel_tiny.mesh";
const int    order           =  2;
const double t_final         =   0.200  ;
const int    problem         =  0;
const int    ref_levels      =  0;

const bool   time_adapt      =  false;
const double cfl             =  0.50 ;
const double dt_const        =  0.002  ;
const int    ode_solver_type =  3; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       = 2000;

// Variable p
const bool   variable_p      =  true;
const bool   adapt           =  false;
const int    adapt_iter      =  200  ; // Time steps after which adaptation is done
const double tolerance       =  5e-4 ; // Tolerance for adaptation

//Source Term
const bool addSource         =  true;                   // Add source term
const double fx              =  0.0029653061224489793; // Force x 

//Restart parameters
const bool restart           = false ;
const int  restart_freq      =  5000; // Create restart file after every 1000 time steps
const int  restart_cycle     =     4; // File number used for restart

//Boundary parameters
const int num_bnd            =     1; // Number of boundaries
const int no_wall_bnd        =     0; // Number of wall boundaries
////////////////////////////////////////////////////////////////////////


// Velocity coefficient
void init_function(const Vector &x, Vector &v);
//void char_bnd_cnd(const Vector &x, Vector &v);
void wall_bnd_cnd(const Vector &x, Vector &v);
void wall_adi_bnd_cnd(const Vector &x, Vector &v);

double getUMax(int dim, const Vector &u);

void getInvFlux(int dim, const Vector &u, Vector &f);
void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z,
        const CGSolver &M_solver, const Vector &u, 
        const Vector &b_aux_x, const Vector &b_aux_y, const Vector &b_aux_z,        
        Vector &aux_grad);
void getAuxVar(int dim, const Vector &u, Vector &aux_sol);


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

   void GetMomentum(const ParGridFunction &x, double &fx);

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
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);

   double kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   /////////////////////////////////////////////////////////////////////////
   // Need to get unique y values for turbulent channel flow before partitioning
   // BUILDING Channel flow analysis variables 
   Array<int> eleOrder_temp;
   int ne_temp = mesh->GetNE();
   eleOrder_temp.SetSize(ne_temp);
   for (int i = 0; i < ne_temp; i++)
   {
       eleOrder_temp[i] = order; 
   }

   FaceElementTransformations *T_temp;
   int nbfaces_temp = mesh->GetNBE();
   for (int i = 0; i < nbfaces_temp; i++)
   {
       T_temp = mesh->GetBdrFaceTransformations(i);
   
       eleOrder_temp[T_temp -> Elem1No] = order + 1; 
   }
 
   VarL2_FiniteElementCollection *vfec_temp = new VarL2_FiniteElementCollection(mesh, eleOrder_temp);
   FiniteElementSpace *fes_temp = new FiniteElementSpace(mesh, vfec_temp, var_dim);

   vector<double> y_uni;
   GetUniqueY(*fes_temp, *vfec_temp, y_uni);

   delete fes_temp;
   delete vfec_temp;
   // Now we can do CNS part
   /////////////////////////////////////////////////////////////////////////
   
   delete mesh;

   Array<int> eleOrder;

   ////////////////////////////////////////
   // Uniform element order
   int ne = pmesh->GetNE();
   eleOrder.SetSize(ne);
   for (int i = 0; i < ne; i++)
   {
       eleOrder[i] = order; 
   }
   ////////////////////////////////////////
   // Wall element order
   FaceElementTransformations *T;
   int nbfaces = pmesh->GetNBE();
   for (int i = 0; i < nbfaces; i++)
   {
       T = pmesh->GetBdrFaceTransformations(i);
   
       eleOrder[T -> Elem1No] = order + 1; 
   }
   ////////////////////////////////////////

   VarL2_FiniteElementCollection vfec(pmesh, eleOrder);

   fes = new ParFiniteElementSpace(pmesh, &vfec, var_dim);

   ///////////////////////////////////////////////////////
   //Get periodic ids for turbulent channel flow
   vector< vector<int> > ids;
   GetPeriodicIds(*fes, vfec, y_uni, ids);
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
   {
       doRestart(restart_cycle, *pmesh, *u_sol, r_t, r_ti);
   }

   fes_vec = new ParFiniteElementSpace(pmesh, &vfec, dim*var_dim);
       
   f_inv = new ParGridFunction(fes_vec);
   getInvFlux(dim, *u_sol, *f_inv);

   f_vis = new ParGridFunction(fes_vec);

   ParFiniteElementSpace fes_aux(pmesh, &vfec, aux_dim);
   aux_sol = new ParGridFunction(&fes_aux);

   ParFiniteElementSpace fes_aux_grad(pmesh, &vfec, dim*aux_dim);
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
   VectorConstantCoefficient z_dir(ydir);
   if (dim == 3)
   {
       zdir = 0.0; zdir(2) = 1.0;
       VectorConstantCoefficient temp_z_dir(zdir);
       z_dir = temp_z_dir;
   }

   ParBilinearForm *m, *k_inv_x, *k_inv_y, *k_inv_z;
   ParBilinearForm     *k_vis_x, *k_vis_y, *k_vis_z;

   fes_op = new ParFiniteElementSpace(pmesh, &vfec);
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

   if (dim == 3)
   {
       k_inv_z = new ParBilinearForm(fes_op);
       k_inv_z->AddDomainIntegrator(new ConvectionIntegrator(z_dir, -1.0));
    
       k_inv_z->Assemble(skip_zeros);
       k_inv_z->Finalize(skip_zeros);
   }

   /////////////////////////////////////////////////////////////

   u_b    = new ParGridFunction(fes);
   f_I_b  = new ParGridFunction(fes_vec);

   VectorGridFunctionCoefficient u_vec(u_b);
   VectorGridFunctionCoefficient f_vec(f_I_b);

   *u_b   = *u_sol;
   *f_I_b = *f_inv;

   u_b  ->ExchangeFaceNbrData(); //Exchange data across processors
   f_I_b->ExchangeFaceNbrData();

   getAuxVar(dim, *u_sol, *aux_sol);
   VectorGridFunctionCoefficient aux_vec(aux_sol);

   getVisFlux(dim, *u_sol, *aux_grad, *f_vis);

   VectorGridFunctionCoefficient vis_vec(f_vis);
   VectorGridFunctionCoefficient aux_grad_vec(aux_grad);

   // Linear form representing the transpose boundary non-linear term as well as boundary terms
   // of the Euler part
   b       = new ParLinearForm(fes);

   // Linear form for boundary conditions on aux variables
   b_aux_x     = new ParLinearForm(&fes_aux);
   b_aux_y     = new ParLinearForm(&fes_aux);
   b_aux_z     = new ParLinearForm(&fes_aux);

   (*b_aux_x) *= 0.0; // Initialize to 0
   (*b_aux_y) *= 0.0;
   (*b_aux_z) *= 0.0;

   b->AddFaceIntegrator(
      new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, var_dim, -1.0));

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
       k_vis_z = new ParBilinearForm(fes_op);
       k_vis_z->AddDomainIntegrator(new ConvectionIntegrator(z_dir,  1.0));
       k_vis_z->AddInteriorFaceIntegrator(
          new TransposeIntegrator(new DGTraceIntegrator(z_dir, -1.0,  0.0)));// Beta 0 means central flux
    
       k_vis_z->Assemble(skip_zeros);
       k_vis_z->Finalize(skip_zeros);
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


   ///////////////////////////////////////////////////
   // Define boundary terms
   // Each boundary should have its own marker since they are passed by ref
   Array<int> dir_bdr_wall(pmesh->bdr_attributes.Max());
   dir_bdr_wall    = 0; // Deactivate all boundaries
   dir_bdr_wall[0] = 1; // Activate wall bdy 
      
   // For isothermal walls
   VectorFunctionCoefficient u_wall_bnd(aux_dim, wall_bnd_cnd); // Defines wall boundary condition

   // Will only be activated if there are boundaries
   if (num_bnd > 0)
   {
       b->AddBdrFaceIntegrator(
          new DG_Euler_NoSlip_Isotherm_Integrator(
              R_gas, gamm, u_vec, f_vec, u_wall_bnd, -1.0), dir_bdr_wall); 
       b->AddBdrFaceIntegrator(
           new DG_CNS_Vis_Isotherm_Integrator(
               R_gas, gamm, u_vec, vis_vec, aux_grad_vec, u_wall_bnd, mu, Pr, 1.0), dir_bdr_wall);

       b_aux_x->AddBdrFaceIntegrator(
              new DG_CNS_Aux_Integrator(
                  x_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
       b_aux_x->Assemble();
      
       b_aux_y->AddBdrFaceIntegrator(
               new DG_CNS_Aux_Integrator(
                   y_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
       b_aux_y->Assemble();
   
       if (dim == 3)
       {
           b_aux_z->AddBdrFaceIntegrator(
               new DG_CNS_Aux_Integrator(
                   z_dir, aux_vec, u_wall_bnd,  1.0), dir_bdr_wall);
           b_aux_z->Assemble();
       }
   }


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

   {   // Post process initially
       // Initial gradients will be garbage
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

   Vector forces(dim);
   ofstream force_file;
   force_file.open ("forces.dat");
   force_file << "Iteration \t dt \t time \t F_x \t F_y \n";

   vector<double> loc_u_mean, glob_u_mean, temp_u_mean;
   vector<double> loc_uu_mean, glob_uu_mean, loc_vv_mean, glob_vv_mean;
   vector<double> loc_ww_mean, glob_ww_mean, loc_uv_mean, glob_uv_mean;
   vector<double> temp_uu_mean, temp_vv_mean, temp_ww_mean, temp_uv_mean;

   bool done = false;
   for (ti = ti_in; !done; )
   {
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
          if (variable_p)
              writeRestart(*pmesh, order, *u_sol, ti, t);
          else
              writeRestart(*pmesh, *u_sol, ti, t);
      }

      double tk = ComputeTKE(*fes, *u_sol);
      double ub, vb, wb, vol;
      ComputeUb(*u_sol, ub, vb, wb, vol);
      double glob_tk, glob_ub, glob_vb, glob_wb, glob_vol;
      MPI_Allreduce(&tk,  &glob_tk,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&ub,  &glob_ub,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vb,  &glob_vb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&wb,  &glob_wb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vol, &glob_vol, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors

      double temp_fx, glob_vol_fx;
      adv->GetMomentum(*y_t, temp_fx);
      MPI_Allreduce(&temp_fx, &glob_vol_fx, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors

      if (myid == 0)
      {
          glob_ub = glob_ub/glob_vol;
          glob_vb = glob_vb/glob_vol;
          glob_wb = glob_wb/glob_vol;
          flo_file << setprecision(8) << ti << "\t" << t << "\t" << glob_tk << "\t" 
              << glob_ub << "\t" << glob_vol_fx << "\t" << glob_vb << "\t" << glob_wb<< endl;
      }

      ComputeWallForces(*fes, *u_sol, *f_vis, dir_bdr_wall, gamm, forces);
      double fx = forces(0), fy = forces(1);
      double glob_fx, glob_fy;
      MPI_Allreduce(&fx, &glob_fx, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&fy, &glob_fy, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors

      if (myid == 0)
      {
          force_file << ti << "\t" <<  dt_real << "\t" << t << "\t" << glob_fx << "\t" << glob_fy <<  endl;
      }
      
      ComputePeriodicMean(dim, *u_sol, ids, loc_u_mean, loc_uu_mean, loc_vv_mean, 
                         loc_ww_mean, loc_uv_mean);
      ComputeGlobPeriodicMean(comm, ids, loc_u_mean, loc_uu_mean, loc_vv_mean,
                              loc_ww_mean, loc_uv_mean, 
                              glob_u_mean, glob_uu_mean, glob_vv_mean, 
                              glob_ww_mean, glob_uv_mean);

      int c_ti = 1;
      if (restart == true)
          c_ti = restart_cycle + 1;
      if (ti == c_ti)
      {
          temp_u_mean  = glob_u_mean;
          temp_uu_mean = glob_uu_mean;
          temp_vv_mean = glob_vv_mean;
          temp_ww_mean = glob_ww_mean;
          temp_uv_mean = glob_uv_mean;
      }
      else if (ti > c_ti)
      {
          int vert_nodes = glob_u_mean.size();      
          for(int i = 0; i < vert_nodes; i++)
          {
              temp_u_mean.at(i)  = temp_u_mean.at(i)*(ti - c_ti)  + glob_u_mean.at(i);          
              temp_uu_mean.at(i) = temp_uu_mean.at(i)*(ti - c_ti) + glob_uu_mean.at(i);          
              temp_vv_mean.at(i) = temp_vv_mean.at(i)*(ti - c_ti) + glob_vv_mean.at(i);          
              temp_ww_mean.at(i) = temp_ww_mean.at(i)*(ti - c_ti) + glob_ww_mean.at(i);          
              temp_uv_mean.at(i) = temp_uv_mean.at(i)*(ti - c_ti) + glob_uv_mean.at(i);          

              temp_u_mean.at(i)  = temp_u_mean.at(i)/double(ti - c_ti + 1);
              temp_uu_mean.at(i) = temp_uu_mean.at(i)/double(ti - c_ti + 1);
              temp_vv_mean.at(i) = temp_vv_mean.at(i)/double(ti - c_ti + 1);
              temp_ww_mean.at(i) = temp_ww_mean.at(i)/double(ti - c_ti + 1);
              temp_uv_mean.at(i) = temp_uv_mean.at(i)/double(ti - c_ti + 1);
          }
          if (done || ti % vis_steps == 0) // Write mean u
          {
              if (myid == 0)
                  writeUMean(ti, y_uni, glob_u_mean, temp_u_mean, temp_uu_mean,
                            temp_vv_mean, temp_ww_mean, temp_uv_mean);

          }
      }
  
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
        if (i == 1)
            s = fx; 
        else if (i == var_dim - 1)
            s = fx*ub;
        y_temp.SetSubVector(offsets[i], s);
    }

    add(y_temp, y, y);

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

   if (dim == 3)
   {
       double rho, u1, u2, u3, p;
       /*
        * Discrete filter operators for large-eddy simulation using high-order spectral difference methods
        */
       rho = 1;
       
       double re_tau = 180;
       double y_tau  = 1/re_tau;
       double u_tau  = mu*re_tau;
       double kappa  = 0.38, C = 4.1;

       double yplus  = x(1)/y_tau;
       if (x(1) > 1)
           yplus  = std::abs(x(1) - 2)/y_tau;
       double uplus  = (1/kappa)*log(1 + kappa*yplus) + 
                       (C - (1/kappa)*log(kappa))*(1 - exp(-yplus/11.0) - (yplus/11.0)*exp(-yplus/3.0));

       double amp    =  0.1;

       u1  = 1.6*pow( (1 - pow( x(1) - 1, 2)), 2);
       u2  = amp*exp(-pow((x(0) - M_PI)/(2*M_PI), 2))*exp(-pow((x(1)/2.0), 2))*cos(4*M_PI*x(2)/M_PI);
       u3  = 0.0;
       p   = 71.42857142857143;
       
       u1 += amp*sin(10*M_PI*x(1)/2)*sin(10*M_PI*x(2)/M_PI);
       u1 += amp*sin(20*M_PI*x(1)/2)*sin(20*M_PI*x(2)/M_PI);
       u1 += amp*sin(30*M_PI*x(1)/2)*sin(30*M_PI*x(2)/M_PI);
       u1 += amp*sin(40*M_PI*x(1)/2)*sin(40*M_PI*x(2)/M_PI);

       u2 += amp*sin(10*M_PI*x(0)/(2*M_PI))*sin(10*M_PI*x(2)/M_PI);
       u2 += amp*sin(20*M_PI*x(0)/(2*M_PI))*sin(20*M_PI*x(2)/M_PI);
       u2 += amp*sin(30*M_PI*x(0)/(2*M_PI))*sin(30*M_PI*x(2)/M_PI);
       u2 += amp*sin(40*M_PI*x(0)/(2*M_PI))*sin(40*M_PI*x(2)/M_PI);

       u3 += amp*sin(10*M_PI*x(0)/(2*M_PI))*sin(10*M_PI*x(1)/2);
       u3 += amp*sin(20*M_PI*x(0)/(2*M_PI))*sin(20*M_PI*x(1)/2);
       u3 += amp*sin(30*M_PI*x(0)/(2*M_PI))*sin(30*M_PI*x(1)/2);
       u3 += amp*sin(40*M_PI*x(0)/(2*M_PI))*sin(40*M_PI*x(1)/2);

       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = rho * u3;                //rho * v
       v(4) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
   else if (dim == 2)
   {
       double rho, u1, u2, p;
       /*
        * Discrete filter operators for large-eddy simulation using high-order spectral difference methods
        */
       rho = 1;
       u1  = 1.9*(1 - pow( x(1) - 1, 2));
       u2  = 0.1*1.0*exp(-pow((x(0) - M_PI)/(2*M_PI), 2))*exp(-pow((x(1)/2.0), 2)); 
       p   = 71.42857142857143;

       double v_sq = pow(u1, 2) + pow(u2, 2) ;
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
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
       v(3) = 0.2488800398208064;  // Temp 
   }
   else if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.2488800398208064;  // Temp 
   }

}


//void char_bnd_cnd(const Vector &x, Vector &v)
//{
//   //Space dimensions 
//   int dim = x.Size();
//
//   if (dim == 2)
//   {
//       double rho, u1, u2, p;
//       rho = 1;
//       u1  = 3.0924; u2 = 0.2162; 
//       p   = 172.2;
//    
//       double v_sq = pow(u1, 2) + pow(u2, 2);
//    
//       v(0) = rho;                     //rho
//       v(1) = rho * u1;                //rho * u
//       v(2) = rho * u2;                //rho * v
//       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
//   }
//   else if (dim == 3)
//   {
//       double rho, u1, u2, u3, p;
//       rho = 1;
//       u1  = 3.0924; u2 = 0.2162; u3 = 0.0;
//       p   =172.2;
//    
//       double v_sq = pow(u1, 2) + pow(u2, 2) + pow(u3, 2);
//    
//       v(0) = rho;                     //rho
//       v(1) = rho * u1;                //rho * u
//       v(2) = rho * u2;                //rho * v
//       v(3) = rho * u3;                //rho * w
//       v(4) = p/(gamm - 1) + 0.5*rho*v_sq;
//   }
//
//}

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

    Vector s(offset);
    for(int i = 0; i < var_dim; i++)
    {
        s = 0.0;
        if (i == 1)
            s = fx; 
        else if (i == var_dim - 1)
            s = fx;
        y_temp.SetSubVector(offsets[i], s);
    }
    add(y_temp, y, y);

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


