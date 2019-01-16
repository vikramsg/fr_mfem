#include "mfem.hpp"
#include "cns.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.00076; 
const double R_gas = 287.;
const double   Pr  = 0.71;

//Run parameters
const char *mesh_file        =  "channel_p1.mesh";
const int    order           =  3; int np = order + 1;
const double t_final         =2000.0001 ;
const int    problem         =  0;
const int    ref_levels      =  0;

const bool   time_adapt      =  true ;
const double cfl             =  0.65 ;
const double dt_const        =  0.0001 ;
const int    ode_solver_type =  3; // 1. Forward Euler 2. TVD SSP 3 Stage

const int    vis_steps       = 5500;

const bool   adapt           =  false;
const int    adapt_iter      =  200  ; // Time steps after which adaptation is done
const double tolerance       =  5e-4 ; // Tolerance for adaptation

//Source Term
const bool addSource         =  true;  // Add source term
const bool constForcing      = false;  // Is the forcing constant 
const bool mdotForcing       =  true;  // Forcing for constant mass flow 
const double fx              =  0.088; // Force x 
const double mdot_pres       =  5.025; // Prescribed mass flow 
      double mdot_fx         =  0.00 ; // Global variable for mass flow forcing 

//Restart parameters
const bool restart           =  false;
const int  restart_freq      =  9500; // Create restart file after every 1000 time steps
const int  restart_cycle     =  9500; // File number used for restart

//Splitting Paramaeter
const bool   split           =   true ;
const string splitform       =   "KG";

//Riemann solver
const string riemann         =   "KG";

// Freestream
const double M_inf           =    0.2 ;
const double rho_inf         =    1.0;

////////////////////////////////////////////////////////////////////////

double bnd_u_max = 0.0;
double mach_max  = 0.0;

// FIXME: It seems that the equally partitioned KE correction is leading to problems in
// wall normal direction velocity accumulation. This could be problematic


// Velocity coefficient
void init_function(const Vector &x, Vector &v);
//void char_bnd_cnd(const Vector &x, Vector &v);
void wall_bnd_cnd(const Vector &x, Vector &v);
void wall_adi_bnd_cnd(const Vector &x, Vector &v);

void getVectorLFFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f);
void getVectorRoeFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f);
void getVectorKGFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f);

SparseMatrix getSparseMat(const Vector &x);

void getKGSplitDx(int dim, 
        const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z, 
        const Vector &u, Vector &f_dx); 

void getFaceDotNorm(int dim, const Vector &f, const Vector &nor_face, Vector &face_f);

double getUMax(int dim, const Vector &u);
double getMMax(int dim, const Vector &u);
void getPRhoMinMax(int dim, const Vector &u, double &rho_min, double &p_min, double &rho_max, double &p_max);

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
   void getKGRestriction(const ParGridFunction &u, Vector &f_l, Vector &f_r) const;
   void getKGCorrection(const ParGridFunction &u, Vector &f_ke_corr) const;

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

    double glob_rho_min, glob_rho_max;
    double glob_p_min,   glob_p_max;

public:
   CNS();

   void Step();

   ~CNS(); 
};



/** Boundary face Riemann integrator
    */
class Bnd_Euler_NoSlip_Isotherm_Integrator: public LinearFormIntegrator, public EulerIntegrator
{
protected:
   VectorCoefficient &uD;
   VectorCoefficient &fD;
   VectorCoefficient &u_bnd;

   double alpha; // b = alpha*b

   double gamm;
   double R   ;

public:
   Bnd_Euler_NoSlip_Isotherm_Integrator(double R_, double gamm_, VectorCoefficient &uD_, VectorCoefficient &fD_, 
                                VectorCoefficient &u_bnd_, double alpha_)
      : uD(uD_), fD(fD_), u_bnd(u_bnd_), alpha(alpha_), R(R_), gamm(gamm_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);
};



void Bnd_Euler_NoSlip_Isotherm_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}



void Bnd_Euler_NoSlip_Isotherm_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, ndof, aux_dim;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   aux_dim = dim + 1;
   var_dim = dim + 2;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   Vector u1_dir(var_dim), u2_dir(var_dim);
   Vector u2_bnd(aux_dim);
   Vector vel_L(dim);    
   Vector vel_R(dim);   
   Vector f_dir(dim*var_dim);
   Vector f1_dir(dim*var_dim);
   Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
         
      order = 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      CalcOrtho(Trans.Face->Jacobian(), nor);

      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Trans.Elem1, eip);

      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);
      double E_L = u1_dir(var_dim - 1);

      double rho_R = rho_L; // Extrapolate pressure
      v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = - vel_L(j);      
          v_sq    += pow(vel_R(j), 2);
      }
      double E_R   = E_L;

      u2_dir(0) = rho_R;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_R*vel_R(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;

      getVectorLFFlux(R, gamm, dim, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs

      fD.Eval(f1_dir, *Trans.Elem1, eip); // Get discontinuous flux at face

      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}



/** Boundary viscous adiabatic integrator
    */
class Bnd_CNS_Vis_Isotherm_Integrator: public LinearFormIntegrator, public CNSIntegrator
{
protected:
   VectorCoefficient &uD;
   VectorCoefficient &fD;
   VectorCoefficient &auxD;
   VectorCoefficient &u_bnd;

   double gamm;
   double R   ;

   double alpha; // b = alpha*b

   double mu, Pr;


public:
   Bnd_CNS_Vis_Isotherm_Integrator(double R_, double gamm_, VectorCoefficient &uD_, VectorCoefficient &fD_, 
                                    VectorCoefficient &auxD_, 
                                    VectorCoefficient &u_bnd_,  double mu_, double Pr_, double alpha_) 
      : uD(uD_), fD(fD_), auxD(auxD_), u_bnd(u_bnd_), alpha(alpha_), mu(mu_), Pr(Pr_) , R(R_), gamm(gamm_) { }

   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       ElementTransformation &Tr,
                                       Vector &elvect);
   virtual void AssembleRHSElementVect(const FiniteElement &el,
                                       FaceElementTransformations &Tr,
                                       Vector &elvect);

};



void Bnd_CNS_Vis_Isotherm_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DG_CNS_Vis_Ad_Integrator::AssembleRHSElementVect");
}


void Bnd_CNS_Vis_Isotherm_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, var_dim, aux_dim, ndof;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   var_dim = dim + 2;
   aux_dim = dim + 1;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   Vector nor_dim(dim);
   Vector u1_dir(var_dim), u2_dir(var_dim);
   Vector u2_bnd(aux_dim);
   Vector aux1_dir(dim*aux_dim), aux2_dir(dim*aux_dim);
   Vector vel_L(dim);    
   Vector vel_R(dim);   
   Vector fL_dir(dim*var_dim), fR_dir(dim*var_dim);
   Vector f_dir(dim*var_dim);
   Vector f1_dir(dim*var_dim);
   Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
     
      order = 2*el.GetOrder();
      
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      Tr.Face->SetIntPoint(&ip);
      Tr.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      double nor_l2 = nor.Norml2();
      nor_dim.Set(1/nor_l2, nor);

      uD.Eval(u1_dir, *Tr.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Tr.Elem1, eip);

      auxD.Eval(aux1_dir, *Tr.Elem1, eip);

      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);
      double T_L = p_L/(rho_L*R); 

      double rho_R = rho_L; // Extrapolate pressure
      v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = 2*u2_bnd(j) - vel_L(j);      
          v_sq    += pow(vel_R(j), 2);
      }
      double T_R   = 2*u2_bnd(aux_dim - 1) - T_L;
      double p_R   = rho_R*R*T_R;
      double E_R   = p_R/(gamm - 1) + 0.5*rho_R*v_sq;

      u2_dir(0) = rho_R;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_R*vel_R(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;

      aux2_dir = aux1_dir;

      getViscousCNSFlux(R, gamm, u1_dir, aux1_dir, mu, Pr, fL_dir);
      getViscousCNSFlux(R, gamm, u2_dir, aux2_dir, mu, Pr, fR_dir);
      add(0.5, fL_dir, fR_dir, f_dir);

      fD.Eval(f1_dir, *Tr.Elem1, eip);        // Get discontinuous flux at face

      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}





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

   DG_FECollection fec(order, dim);
   
   /////////////////////////////////////////////////////////////////////////
   // Need to get unique y values for turbulent channel flow before partitioning
   FiniteElementSpace *fes_temp = new FiniteElementSpace(mesh, &fec, var_dim);

   vector<double> y_uni;
   GetUniqueY(*fes_temp, y_uni);

   delete fes_temp;
   /////////////////////////////////////////////////////////////////////////

   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   double kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   fes = new ParFiniteElementSpace(pmesh, &fec, var_dim);

   ///////////////////////////////////////////////////////
   //Get periodic ids for turbulent channel flow
   vector< vector<int> > ids;
   GetPeriodicIds(*fes, y_uni, ids);
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
   /////////////////////////////////////////////////////
   // Define boundary terms
   // Each boundary should have its own marker since they are passed by ref
   Array<int> dir_bdr_wall(pmesh->bdr_attributes.Max());
   dir_bdr_wall    = 0; // Deactivate all boundaries
   dir_bdr_wall[0] = 1; // Activate wall bdy 

   VectorFunctionCoefficient u_wall_bnd(aux_dim, wall_bnd_cnd); // Defines wall boundary condition
   
   DGEulerIntegrator *dgeuler = new DGEulerIntegrator(R_gas, gamm, u_vec, f_vec, var_dim, -1.0);

   Bnd_Euler_NoSlip_Isotherm_Integrator *dg_ns_iso = new Bnd_Euler_NoSlip_Isotherm_Integrator(
              R_gas, gamm, u_vec, f_vec, u_wall_bnd, -1.0); 

   // Linear form representing the Euler boundary non-linear term
   b = new ParLinearForm(fes);
   
   b->AddBdrFaceIntegrator(dg_ns_iso, dir_bdr_wall); 

   b->Assemble();

   getAuxVar(dim, *u_sol, *aux_sol);
   VectorGridFunctionCoefficient aux_vec(aux_sol);

   DG_CNS_Aux_Integrator *dg_aux_x =  new DG_CNS_Aux_Integrator(
              x_dir, aux_vec, u_wall_bnd,  1.0);
   DG_CNS_Aux_Integrator *dg_aux_y =  new DG_CNS_Aux_Integrator(
              y_dir, aux_vec, u_wall_bnd,  1.0);

   b_aux_x = new ParLinearForm(&fes_aux);
   b_aux_y = new ParLinearForm(&fes_aux);
   b_aux_z = new ParLinearForm(&fes_aux);
   b_aux_x->AddBdrFaceIntegrator(dg_aux_x, dir_bdr_wall);
   b_aux_x->Assemble();

   b_aux_y->AddBdrFaceIntegrator(dg_aux_y, dir_bdr_wall);
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
 
       DG_CNS_Aux_Integrator *dg_aux_z =  new DG_CNS_Aux_Integrator(
              z_dir, aux_vec, u_wall_bnd,  1.0);

       b_aux_z->AddBdrFaceIntegrator(dg_aux_z, dir_bdr_wall);
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

   Bnd_CNS_Vis_Isotherm_Integrator *dg_vis_iso = new Bnd_CNS_Vis_Isotherm_Integrator(
               R_gas, gamm, u_vec, vis_vec, aux_grad_vec, u_wall_bnd, mu, Pr, 1.0);

   b->AddBdrFaceIntegrator(dg_vis_iso, dir_bdr_wall);

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
       loc_u_max = std::max(loc_u_max, bnd_u_max); 
       MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

       dt = cfl*((h_min/(2.0*order + 1))/glob_u_max);
   }
    
   MPI_Comm comm = pmesh->GetComm();

   ofstream flo_file;
   flo_file.open("flow_char.dat");

   StopWatch chrono;
   chrono.Clear();
   chrono.Start();

   Vector forces(dim);
   ofstream force_file;
   force_file.open ("forces.dat");
   force_file << "Iteration \t dt \t time \t F_x \t F_y \n";
 
   vector<double> loc_u_mean, loc_v_mean, loc_w_mean;
   vector<double> temp_u_mean, temp_v_mean, temp_w_mean ;
   vector<double> glob_u_mean, glob_v_mean, glob_w_mean ;
   vector<double> loc_uu_mean, glob_uu_mean, loc_vv_mean, glob_vv_mean;
   vector<double> loc_ww_mean, glob_ww_mean, loc_uv_mean, glob_uv_mean;
   vector<double> temp_uu_mean, temp_vv_mean, temp_ww_mean, temp_uv_mean;

   bool done = false;
   for (ti = ti_in; !done; )
   {
      double tk = ComputeTKE(*fes, *u_sol);
      double ub, vb, wb, vol;
      ComputeUb(*u_sol, ub, vb, wb, vol);
      double glob_tk, glob_ub, glob_vb, glob_wb, glob_vol;
      MPI_Allreduce(&tk,  &glob_tk,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&ub,  &glob_ub,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vb,  &glob_vb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&wb,  &glob_wb,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
      MPI_Allreduce(&vol, &glob_vol, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors

      double mdot_0, fx_0, mdot_now;
      if (ti < ti_in + 1)
      {
          mdot_fx = fx; 
          fx_0    = mdot_fx;
          mdot_0  = glob_ub/glob_vol;
      }
      else
      {
          mdot_now = glob_ub/glob_vol;
          mdot_fx  = fx   + 0.3*(mdot_pres - 2*mdot_now + mdot_0)/(dt); 
          mdot_0   = glob_ub/glob_vol;
          fx_0     = mdot_fx;
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

      double temp_fx, glob_vol_fx;
      adv->GetMomentum(*y_t, temp_fx);
      MPI_Allreduce(&temp_fx, &glob_vol_fx, 1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors

      if (myid == 0)
      {
          glob_ub = glob_ub/glob_vol;
          glob_vb = glob_vb/glob_vol;
          glob_wb = glob_wb/glob_vol;
          flo_file << setprecision(6) << ti << "\t" << t << "\t" << glob_tk << "\t" 
              << glob_ub << "\t" << glob_vol_fx << "\t" << glob_vb << "\t" << glob_wb << "\t"
              << mdot_fx << endl;
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
      
      ComputePeriodicMean(dim, *u_sol, ids, loc_u_mean, loc_v_mean, loc_w_mean, 
                          loc_uu_mean, loc_vv_mean, loc_ww_mean, loc_uv_mean);
      ComputeGlobPeriodicMean(comm, ids, loc_u_mean, loc_v_mean, loc_w_mean, 
                              loc_uu_mean, loc_vv_mean, loc_ww_mean, loc_uv_mean, 
                              glob_u_mean, glob_v_mean, glob_w_mean, 
                              glob_uu_mean, glob_vv_mean, glob_ww_mean, glob_uv_mean);

      int c_ti = 1;
      if (restart == true)
          c_ti = restart_cycle + 1;
      if (ti == c_ti)
      {
          temp_u_mean  = glob_u_mean;
          temp_v_mean  = glob_v_mean;
          temp_w_mean  = glob_w_mean;
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
              temp_v_mean.at(i)  = temp_v_mean.at(i)*(ti - c_ti)  + glob_v_mean.at(i);          
              temp_w_mean.at(i)  = temp_w_mean.at(i)*(ti - c_ti)  + glob_w_mean.at(i);          
              temp_uu_mean.at(i) = temp_uu_mean.at(i)*(ti - c_ti) + glob_uu_mean.at(i);          
              temp_vv_mean.at(i) = temp_vv_mean.at(i)*(ti - c_ti) + glob_vv_mean.at(i);          
              temp_ww_mean.at(i) = temp_ww_mean.at(i)*(ti - c_ti) + glob_ww_mean.at(i);          
              temp_uv_mean.at(i) = temp_uv_mean.at(i)*(ti - c_ti) + glob_uv_mean.at(i);          

              temp_u_mean.at(i)  = temp_u_mean.at(i)/double(ti - c_ti + 1);
              temp_v_mean.at(i)  = temp_v_mean.at(i)/double(ti - c_ti + 1);
              temp_w_mean.at(i)  = temp_w_mean.at(i)/double(ti - c_ti + 1);
              temp_uu_mean.at(i) = temp_uu_mean.at(i)/double(ti - c_ti + 1);
              temp_vv_mean.at(i) = temp_vv_mean.at(i)/double(ti - c_ti + 1);
              temp_ww_mean.at(i) = temp_ww_mean.at(i)/double(ti - c_ti + 1);
              temp_uv_mean.at(i) = temp_uv_mean.at(i)/double(ti - c_ti + 1);
          }
          if (ti % vis_steps == 0) // Write mean u
          {
              if (myid == 0)
                  writeUMean(ti, y_uni, glob_u_mean, temp_u_mean, temp_v_mean, temp_w_mean, 
                            temp_uu_mean, temp_vv_mean, temp_ww_mean, temp_uv_mean);
          }
      }


  
   }
   flo_file.close();
   
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
    loc_u_max = std::max(loc_u_max, bnd_u_max); 
    MPI_Allreduce(&loc_u_max, &glob_u_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

    double loc_m_max, glob_m_max; 
    loc_m_max = getMMax(dim, *u_sol); // Get u_max on local processor
    loc_m_max = std::max(loc_m_max, mach_max); 
    MPI_Allreduce(&loc_m_max, &glob_m_max, 1, MPI_DOUBLE, MPI_MAX, comm); // Get global u_max across processors

    double loc_rho_min, loc_rho_max; 
    double loc_p_min,   loc_p_max; 
    getPRhoMinMax(dim, *u_sol, loc_rho_min, loc_p_min, loc_rho_max, loc_p_max); 
    MPI_Allreduce(&loc_rho_min, &glob_rho_min, 1, MPI_DOUBLE, MPI_MIN, comm); 
    MPI_Allreduce(&loc_rho_max, &glob_rho_max, 1, MPI_DOUBLE, MPI_MAX, comm); 
    MPI_Allreduce(&loc_p_min,   &glob_p_min,   1, MPI_DOUBLE, MPI_MIN, comm); 
    MPI_Allreduce(&loc_p_max,   &glob_p_max,   1, MPI_DOUBLE, MPI_MAX, comm); 

    if (myid == 0)
    {
        cout << setprecision(6) << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
            t << ", max_mach " << glob_m_max << ", rho_min "<< glob_rho_min << ", p_min "<< glob_p_min << ", fes_size " << fes->GlobalTrueVSize() << endl;
    }

    if (myid == 0)
    {
        if ( (glob_p_min < 0) || (glob_rho_min < 0) || (isnan( glob_p_min ) ) || ( isnan(glob_rho_min) ) )
            mfem_error("Negative values encountered"); 
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

    Vector f_dx(var_dim*offset);

    Vector f_sol(offset), f_x(offset), f_x_m(offset);
    Vector b_sub(offset);
    y = 0.0;

    if (split == true)
    {
        if ( splitform == "KG" ) 
        {
            getKGSplitDx(dim, K_inv_x, K_inv_y, K_inv_z, x, f_dx);
            Vector temp(var_dim*offset), f_ke_corr(var_dim*offset);
            getKGCorrection(u, f_ke_corr);
            add(f_dx, f_ke_corr, temp);
            f_dx = temp;
        }
        else
            mfem_error("Split form does not exist");

        for(int i = 0; i < var_dim; i++)
        {
            f_dx.GetSubVector(offsets[i], f_x);
            b.GetSubVector(offsets[i], b_sub);
            f_x += b_sub; // Needs to be added only once
            y_temp.SetSubVector(offsets[i], f_x);
        }
        y += y_temp;
    }
    else
    {
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

   Vector f_l (dim*var_dim*n_face_pts), f_r (dim*var_dim*n_face_pts);

   if (split == true)
   {
        if ( splitform == "KG" ) 
        {
            getKGRestriction(u_sol, f_l, f_r);
        }
        else
        {
           Vector f_l1 (dim*var_dim*n_face_pts), f_r1(dim*var_dim*n_face_pts);
           Vector f_l2 (dim*var_dim*n_face_pts), f_r2(dim*var_dim*n_face_pts);
           for(int j = 0; j < dim*var_dim; j++)
           {
               f_inv.GetSubVector(offsets[j], u_sub);
        
               glob_proj_l->Mult(u_sub, u_f_sub);
               f_l1.SetSubVector(offsets_face[j], u_f_sub);
        
               glob_proj_r->Mult(u_sub, u_f_sub);
               f_r1.SetSubVector(offsets_face[j], u_f_sub);
           }
   
           getInvFlux(dim, u_l, f_l2);
           getInvFlux(dim, u_r, f_r2);

           add(0.5, f_l1, f_l2, f_l);
           add(0.5, f_r1, f_r2, f_r);

        }
   }
   else
   {
       for(int j = 0; j < dim*var_dim; j++)
       {
           f_inv.GetSubVector(offsets[j], u_sub);
    
           glob_proj_l->Mult(u_sub, u_f_sub);
           f_l.SetSubVector(offsets_face[j], u_f_sub);
    
           glob_proj_r->Mult(u_sub, u_f_sub);
           f_r.SetSubVector(offsets_face[j], u_f_sub);
       }

   }
       
   Vector f_com(dim*var_dim*n_face_pts);
   Vector f_left(dim*var_dim*n_face_pts), f_rght(dim*var_dim*n_face_pts) ;

   if (riemann == "LF")
       getVectorLFFlux(R_gas, gamm, dim, u_l, u_r, nor_face, f_com);
   else if (riemann == "Roe")
       getVectorRoeFlux(R_gas, gamm, dim, u_l, u_r, nor_face, f_com);
   else if (riemann == "KG")
       getVectorKGFlux(R_gas, gamm, dim, u_l, u_r, nor_face, f_com);
   else
       mfem_error("Not a valid Riemann solver");
  
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



CGSolver &FE_Evolution::GetMSolver() 
{
    return M_solver;
}



/*
 * For Gauss points, the restriction of f for the RHS C(f_{num} - f) does not guarantee conservation
 * for splittings in general. Therefore we need to instroduce a function to get this restriction
 * for each splitting
 */
void FE_Evolution::getKGRestriction(const ParGridFunction &u, Vector &f_l, Vector &f_r) const
{
   int var_dim    = u.VectorDim(); 
   int dim        = var_dim - 2;
   int n_face_pts = wts.Size();
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

   int offset = u.Size()/var_dim; 

   Vector rho, E;
   u.GetSubVector(offsets[0],           rho   );
   u.GetSubVector(offsets[var_dim - 1],      E);

   Vector rho_vel[dim];
   for(int i = 0; i < dim; i++) u.GetSubVector(offsets[1 + i], rho_vel[i]);

   Vector vel[dim], rho_vel_sq[dim], rho_uv[dim], uv[dim], v_sq[dim];
   for(int i = 0; i < dim; i++) 
   {
       vel[i].SetSize(offset); rho_vel_sq[i].SetSize(offset); rho_uv[i].SetSize(offset);
       uv[i].SetSize(offset);
       v_sq[i].SetSize(offset);
   }

   Vector pres(offset); // (rho*Cv*T + p)*u, p
   Vector pu[dim]; 
   Vector e(offset), eu[dim], rho_eu[dim]; 
   for(int i = 0; i < dim; i++) 
   {
       eu[i]    .SetSize(offset);
       rho_eu[i].SetSize(offset);
       pu[i]    .SetSize(offset);
   }

   for(int i = 0; i < offset; i++)
   {
       double vel_sq = 0.0;
       for(int j = 0; j < dim; j++)
       {
           vel[j][i]        = rho_vel[j](i)/rho(i);
           vel_sq          += pow(vel[j][i], 2);

           rho_vel_sq[j][i] = rho_vel[j](i)*vel[j](i);
           v_sq[j][i]       = vel[j](i)*vel[j](i);
       }

       pres(i)          = (E(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);
       e(i)             =  E(i)/rho(i);

       for(int j = 0; j < dim; j++)
       {
           eu[j](i)            = e(i)*vel[j](i);
           rho_eu[j](i)        = E(i)*vel[j](i);
           
           pu[j](i)            = pres(i)*vel[j](i);
       }

       rho_uv[0](i)  = rho_vel[0](i)*vel[1](i); // rho*u*v
       rho_uv[1](i)  = rho_vel[0](i)*vel[2](i); // rho*u*w
       rho_uv[2](i)  = rho_vel[1](i)*vel[2](i); // rho*v*w
   
       uv[0](i)      = vel[0](i)*vel[1](i); // rho*u*v
       uv[1](i)      = vel[0](i)*vel[2](i); // rho*u*w
       uv[2](i)      = vel[1](i)*vel[2](i); // rho*v*w

   }
   
   Vector rho_vel_Rl[dim]; // Restriction
   Vector vel_Rl[dim], rho_vel_sq_Rl[dim], rho_uv_Rl[dim];
   Vector vel_sq_Rl[dim], uv_Rl[dim];
   Vector pres_Rl(n_face_pts), rho_Rl(n_face_pts); 
   Vector rho_eu_Rl[dim], eu_Rl[dim], E_Rl(n_face_pts), e_Rl(n_face_pts); 
   Vector pu_Rl[dim]; 

   glob_proj_l->Mult(rho ,  rho_Rl );
   glob_proj_l->Mult(pres,  pres_Rl);
   glob_proj_l->Mult(e,                e_Rl);
   glob_proj_l->Mult(E,                E_Rl);
   for(int j = 0; j < dim; j++) 
   {
       rho_vel_Rl[j].SetSize(n_face_pts); vel_Rl[j].SetSize(n_face_pts); rho_uv_Rl[j].SetSize(n_face_pts);
       uv_Rl[j].SetSize(n_face_pts);
       rho_vel_sq_Rl[j].SetSize(n_face_pts); vel_sq_Rl[j].SetSize(n_face_pts);
       rho_eu_Rl[j].SetSize(n_face_pts); eu_Rl[j].SetSize(n_face_pts);
       pu_Rl[j].SetSize(n_face_pts); 

       glob_proj_l->Mult(rho_vel[j],     rho_vel_Rl[j]);
       glob_proj_l->Mult(vel[j],         vel_Rl[j]);
       glob_proj_l->Mult(rho_uv[j],      rho_uv_Rl[j]);
       glob_proj_l->Mult(uv[j],          uv_Rl[j]);
       glob_proj_l->Mult(rho_vel_sq[j],  rho_vel_sq_Rl[j]);
       glob_proj_l->Mult(v_sq[j],        vel_sq_Rl[j]);
       glob_proj_l->Mult(rho_eu[j],      rho_eu_Rl[j]);
       glob_proj_l->Mult(eu[j],          eu_Rl[j]);
       glob_proj_l->Mult(pu[j],          pu_Rl[j]);
   }

   Vector temp(n_face_pts), temp1(n_face_pts), temp2(n_face_pts), temp3(n_face_pts), temp4(n_face_pts);

   {
       // X -dir
       // Density

       temp = rho_vel_Rl[0];

       getSparseMat(vel_Rl[0]).Mult(rho_Rl, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_l.SetSubVector(offsets_face[0], temp);
       
       // rho_u 
       
       temp = rho_vel_sq_Rl[0];

       getSparseMat(vel_Rl[0]).Mult(rho_vel_Rl[0], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rl[0]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rl; 

       f_l.SetSubVector(offsets_face[1], temp);

       // rho_v 

       temp = rho_uv_Rl[0];

       getSparseMat(vel_Rl[0]).Mult(rho_vel_Rl[1], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[1]).Mult(rho_vel_Rl[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[0]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[2], temp);

       // rho_w 

       temp = rho_uv_Rl[1];

       getSparseMat(vel_Rl[0]).Mult(rho_vel_Rl[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[2]).Mult(rho_vel_Rl[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[1]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[3], temp);
   
       // rho_e 

       temp = rho_eu_Rl[0];

       getSparseMat(vel_Rl[0]).Mult(E_Rl, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rl).Mult(eu_Rl[0], temp1);
       temp  += temp1; 

       getSparseMat(e_Rl).Mult(rho_vel_Rl[0], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rl[0];
       
       getSparseMat(vel_Rl[0]).Mult(pres_Rl, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_l.SetSubVector(offsets_face[4], temp);

   }

   {
       // Y -dir
       // Density

       temp = rho_vel_Rl[1];

       getSparseMat(vel_Rl[1]).Mult(rho_Rl, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_l.SetSubVector(offsets_face[var_dim + 0], temp);
       
       // rho_u 
       
       temp = rho_uv_Rl[0];

       getSparseMat(vel_Rl[0]).Mult(rho_vel_Rl[1], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[1]).Mult(rho_vel_Rl[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[0]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[var_dim + 1], temp);

       // rho_v 

       temp = rho_vel_sq_Rl[1];

       getSparseMat(vel_Rl[1]).Mult(rho_vel_Rl[1], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rl[1]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rl; 

       f_l.SetSubVector(offsets_face[var_dim + 2], temp);

       // rho_w 

       temp = rho_uv_Rl[2];

       getSparseMat(vel_Rl[1]).Mult(rho_vel_Rl[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[2]).Mult(rho_vel_Rl[1], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[2]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[var_dim + 3], temp);
   
       // rho_e 

       temp = rho_eu_Rl[1];

       getSparseMat(vel_Rl[1]).Mult(E_Rl, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rl).Mult(eu_Rl[1], temp1);
       temp  += temp1; 

       getSparseMat(e_Rl).Mult(rho_vel_Rl[1], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rl[1];
       
       getSparseMat(vel_Rl[1]).Mult(pres_Rl, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_l.SetSubVector(offsets_face[var_dim + 4], temp);

   }

   {
       // Z -dir
       // Density

       temp = rho_vel_Rl[2];

       getSparseMat(vel_Rl[2]).Mult(rho_Rl, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_l.SetSubVector(offsets_face[2*var_dim + 0], temp);
       
       // rho_u 
       
       temp = rho_uv_Rl[1];

       getSparseMat(vel_Rl[0]).Mult(rho_vel_Rl[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[2]).Mult(rho_vel_Rl[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[1]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[2*var_dim + 1], temp);

       // rho_v 

       temp = rho_uv_Rl[2];

       getSparseMat(vel_Rl[1]).Mult(rho_vel_Rl[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rl[2]).Mult(rho_vel_Rl[1], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rl[2]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_l.SetSubVector(offsets_face[2*var_dim + 2], temp);

       // rho_w 

       temp = rho_vel_sq_Rl[2];

       getSparseMat(vel_Rl[2]).Mult(rho_vel_Rl[2], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rl[2]).Mult(rho_Rl, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rl; 

       f_l.SetSubVector(offsets_face[2*var_dim + 3], temp);
   
       // rho_e 

       temp = rho_eu_Rl[2];

       getSparseMat(vel_Rl[2]).Mult(E_Rl, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rl).Mult(eu_Rl[2], temp1);
       temp  += temp1; 

       getSparseMat(e_Rl).Mult(rho_vel_Rl[2], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rl[2];
       
       getSparseMat(vel_Rl[2]).Mult(pres_Rl, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_l.SetSubVector(offsets_face[2*var_dim + 4], temp);

   }

   Vector rho_vel_Rr[dim]; // Restriction right
   Vector vel_Rr[dim], rho_vel_sq_Rr[dim], rho_uv_Rr[dim];
   Vector vel_sq_Rr[dim], uv_Rr[dim];
   Vector pres_Rr(n_face_pts), rho_Rr(n_face_pts); 
   Vector rho_eu_Rr[dim], eu_Rr[dim], E_Rr(n_face_pts), e_Rr(n_face_pts); 
   Vector pu_Rr[dim]; 

   glob_proj_r->Mult(rho ,  rho_Rr );
   glob_proj_r->Mult(pres,  pres_Rr);
   glob_proj_r->Mult(e,                e_Rr);
   glob_proj_r->Mult(E,                E_Rr);
   for(int j = 0; j < dim; j++) 
   {
       rho_vel_Rr[j].SetSize(n_face_pts); vel_Rr[j].SetSize(n_face_pts); rho_uv_Rr[j].SetSize(n_face_pts);
       uv_Rr[j].SetSize(n_face_pts);
       rho_vel_sq_Rr[j].SetSize(n_face_pts); vel_sq_Rr[j].SetSize(n_face_pts);
       rho_eu_Rr[j].SetSize(n_face_pts); eu_Rr[j].SetSize(n_face_pts);
       pu_Rr[j].SetSize(n_face_pts); 

       glob_proj_r->Mult(rho_vel[j],     rho_vel_Rr[j]);
       glob_proj_r->Mult(vel[j],         vel_Rr[j]);
       glob_proj_r->Mult(rho_uv[j],      rho_uv_Rr[j]);
       glob_proj_r->Mult(uv[j],          uv_Rr[j]);
       glob_proj_r->Mult(rho_vel_sq[j],  rho_vel_sq_Rr[j]);
       glob_proj_r->Mult(v_sq[j],        vel_sq_Rr[j]);
       glob_proj_r->Mult(rho_eu[j],      rho_eu_Rr[j]);
       glob_proj_r->Mult(eu[j],          eu_Rr[j]);
       glob_proj_r->Mult(pu[j],          pu_Rr[j]);
   }

   {
       // X -dir
       // Density

       temp = rho_vel_Rr[0];

       getSparseMat(vel_Rr[0]).Mult(rho_Rr, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_r.SetSubVector(offsets_face[0], temp);
       
       // rho_u 
       
       temp = rho_vel_sq_Rr[0];

       getSparseMat(vel_Rr[0]).Mult(rho_vel_Rr[0], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rr[0]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rr; 

       f_r.SetSubVector(offsets_face[1], temp);

       // rho_v 

       temp = rho_uv_Rr[0];

       getSparseMat(vel_Rr[0]).Mult(rho_vel_Rr[1], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[1]).Mult(rho_vel_Rr[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[0]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[2], temp);

       // rho_w 

       temp = rho_uv_Rr[1];

       getSparseMat(vel_Rr[0]).Mult(rho_vel_Rr[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[2]).Mult(rho_vel_Rr[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[1]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[3], temp);
   
       // rho_e 

       temp = rho_eu_Rr[0];

       getSparseMat(vel_Rr[0]).Mult(E_Rr, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rr).Mult(eu_Rr[0], temp1);
       temp  += temp1; 

       getSparseMat(e_Rr).Mult(rho_vel_Rr[0], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rr[0];
       
       getSparseMat(vel_Rr[0]).Mult(pres_Rr, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_r.SetSubVector(offsets_face[4], temp);

   }

   {
       // Y -dir
       // Density

       temp = rho_vel_Rr[1];

       getSparseMat(vel_Rr[1]).Mult(rho_Rr, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_r.SetSubVector(offsets_face[var_dim + 0], temp);
       
       // rho_u 
       
       temp = rho_uv_Rr[0];

       getSparseMat(vel_Rr[0]).Mult(rho_vel_Rr[1], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[1]).Mult(rho_vel_Rr[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[0]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[var_dim + 1], temp);

       // rho_v 

       temp = rho_vel_sq_Rr[1];

       getSparseMat(vel_Rr[1]).Mult(rho_vel_Rr[1], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rr[1]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rr; 

       f_r.SetSubVector(offsets_face[var_dim + 2], temp);

       // rho_w 

       temp = rho_uv_Rr[2];

       getSparseMat(vel_Rr[1]).Mult(rho_vel_Rr[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[2]).Mult(rho_vel_Rr[1], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[2]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[var_dim + 3], temp);
   
       // rho_e 

       temp = rho_eu_Rr[1];

       getSparseMat(vel_Rr[1]).Mult(E_Rr, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rr).Mult(eu_Rr[1], temp1);
       temp  += temp1; 

       getSparseMat(e_Rr).Mult(rho_vel_Rr[1], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rr[1];
       
       getSparseMat(vel_Rr[1]).Mult(pres_Rr, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_r.SetSubVector(offsets_face[var_dim + 4], temp);

   }

   {
       // Z -dir
       // Density

       temp = rho_vel_Rr[2];

       getSparseMat(vel_Rr[2]).Mult(rho_Rr, temp1);
       temp += temp1; 

       temp *= 0.5;

       f_r.SetSubVector(offsets_face[2*var_dim + 0], temp);
       
       // rho_u 
       
       temp = rho_uv_Rr[1];

       getSparseMat(vel_Rr[0]).Mult(rho_vel_Rr[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[2]).Mult(rho_vel_Rr[0], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[1]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[2*var_dim + 1], temp);

       // rho_v 

       temp = rho_uv_Rr[2];

       getSparseMat(vel_Rr[1]).Mult(rho_vel_Rr[2], temp1);
       temp  += temp1; 

       getSparseMat(vel_Rr[2]).Mult(rho_vel_Rr[1], temp1);
       temp  += temp1; 

       getSparseMat(uv_Rr[2]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       f_r.SetSubVector(offsets_face[2*var_dim + 2], temp);

       // rho_w 

       temp = rho_vel_sq_Rr[2];

       getSparseMat(vel_Rr[2]).Mult(rho_vel_Rr[2], temp1);
       temp1 *= 2.0;
       temp  += temp1; 

       getSparseMat(vel_sq_Rr[2]).Mult(rho_Rr, temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp  += pres_Rr; 

       f_r.SetSubVector(offsets_face[2*var_dim + 3], temp);
   
       // rho_e 

       temp = rho_eu_Rr[2];

       getSparseMat(vel_Rr[2]).Mult(E_Rr, temp1);
       temp  += temp1; 

       getSparseMat(rho_Rr).Mult(eu_Rr[2], temp1);
       temp  += temp1; 

       getSparseMat(e_Rr).Mult(rho_vel_Rr[2], temp1);
       temp  += temp1; 

       temp  *= 0.25;

       temp1  = pu_Rr[2];
       
       getSparseMat(vel_Rr[2]).Mult(pres_Rr, temp2);

       temp1 += temp2;
       temp1 *= 0.5;

       temp  += temp1;

       f_r.SetSubVector(offsets_face[2*var_dim + 4], temp);

   }


}




/*
 * For Gauss points, after we guarantee conservation, there are unbalanced terms in the
 * kinetic energy equation that does not guarantee kinetic energy preservation anymore
 * So, we need to find the unbalanced terms first
 */
void FE_Evolution::getKGCorrection(const ParGridFunction &u, Vector &f_ke_corr) const
{
   ParFiniteElementSpace *fes = u.ParFESpace();
   MPI_Comm comm              = fes->GetComm();

   int var_dim    = u.VectorDim(); 
   int dim        = var_dim - 2;
   int n_face_pts = wts.Size();
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

   int offset = u.Size()/var_dim; 

   Vector rho, E;
   u.GetSubVector(offsets[0],           rho   );
   u.GetSubVector(offsets[var_dim - 1],      E);

   Vector rho_vel[dim];
   for(int i = 0; i < dim; i++) u.GetSubVector(offsets[1 + i], rho_vel[i]);

   Vector vel[dim], rho_vel_sq[dim], rho_uv[dim], uv[dim], v_sq[dim];
   for(int i = 0; i < dim; i++) 
   {
       vel[i].SetSize(offset); rho_vel_sq[i].SetSize(offset); rho_uv[i].SetSize(offset);
       uv[i].SetSize(offset);
       v_sq[i].SetSize(offset);
   }

   Vector pres(offset); // (rho*Cv*T + p)*u, p
   Vector pu[dim]; 
   Vector e(offset), eu[dim], rho_eu[dim];
   Vector vbar_sq(offset), rho_vbar_sq(offset); // rho*(u*u  + v*v + w*w)
   for(int i = 0; i < dim; i++) 
   {
       eu[i]    .SetSize(offset);
       rho_eu[i].SetSize(offset);
       pu[i]    .SetSize(offset);
   }

   for(int i = 0; i < offset; i++)
   {
       double vel_sq = 0.0;
       for(int j = 0; j < dim; j++)
       {
           vel[j][i]        = rho_vel[j](i)/rho(i);
           vel_sq          += pow(vel[j][i], 2);

           rho_vel_sq[j][i] = rho_vel[j](i)*vel[j](i);
           v_sq[j][i]       = vel[j](i)*vel[j](i);
       }

       rho_vbar_sq[i]   = rho[i]*vel_sq;
       vbar_sq[i]       = vel_sq;

       pres(i)          = (E(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);
       e(i)             =  E(i)/rho(i);

       for(int j = 0; j < dim; j++)
       {
           eu[j](i)            = e(i)*vel[j](i);
           rho_eu[j](i)        = E(i)*vel[j](i);
           
           pu[j](i)            = pres(i)*vel[j](i);
       }

       rho_uv[0](i)  = rho_vel[0](i)*vel[1](i); // rho*u*v
       rho_uv[1](i)  = rho_vel[0](i)*vel[2](i); // rho*u*w
       rho_uv[2](i)  = rho_vel[1](i)*vel[2](i); // rho*v*w
   
       uv[0](i)      = vel[0](i)*vel[1](i); // rho*u*v
       uv[1](i)      = vel[0](i)*vel[2](i); // rho*u*w
       uv[2](i)      = vel[1](i)*vel[2](i); // rho*v*w

   }
   
   Vector rho_vel_Rl[dim]; // Restriction
   Vector vel_Rl[dim], rho_vel_sq_Rl[dim], rho_uv_Rl[dim];
   Vector vel_sq_Rl[dim], uv_Rl[dim];
   Vector pres_Rl(n_face_pts), rho_Rl(n_face_pts); 
   Vector rho_eu_Rl[dim], eu_Rl[dim], E_Rl(n_face_pts), e_Rl(n_face_pts); 
   Vector pu_Rl[dim]; 

   glob_proj_l->Mult(rho ,  rho_Rl );
   glob_proj_l->Mult(pres,  pres_Rl);
   glob_proj_l->Mult(e,                e_Rl);
   glob_proj_l->Mult(E,                E_Rl);
   for(int j = 0; j < dim; j++) 
   {
       rho_vel_Rl[j].SetSize(n_face_pts); vel_Rl[j].SetSize(n_face_pts); rho_uv_Rl[j].SetSize(n_face_pts);
       uv_Rl[j].SetSize(n_face_pts);
       rho_vel_sq_Rl[j].SetSize(n_face_pts); vel_sq_Rl[j].SetSize(n_face_pts);
       rho_eu_Rl[j].SetSize(n_face_pts); eu_Rl[j].SetSize(n_face_pts);
       pu_Rl[j].SetSize(n_face_pts); 

       glob_proj_l->Mult(rho_vel[j],     rho_vel_Rl[j]);
       glob_proj_l->Mult(vel[j],         vel_Rl[j]);
       glob_proj_l->Mult(rho_uv[j],      rho_uv_Rl[j]);
       glob_proj_l->Mult(uv[j],          uv_Rl[j]);
       glob_proj_l->Mult(rho_vel_sq[j],  rho_vel_sq_Rl[j]);
       glob_proj_l->Mult(v_sq[j],        vel_sq_Rl[j]);
       glob_proj_l->Mult(rho_eu[j],      rho_eu_Rl[j]);
       glob_proj_l->Mult(eu[j],          eu_Rl[j]);
       glob_proj_l->Mult(pu[j],          pu_Rl[j]);
   }

   Vector ke_face_error_l(n_face_pts), ke_face_error_r(n_face_pts);

   ke_face_error_l = 0.0;
   for (int j = 0; j < n_face_pts; j++)
   {
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[0][j]*vel_sq_Rl[0][j] - rho_vel_Rl[0][j]*vel_Rl[0][j]*vel_Rl[0][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[0][j]*vel_sq_Rl[1][j] - rho_vel_Rl[0][j]*vel_Rl[1][j]*vel_Rl[1][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[0][j]*vel_sq_Rl[2][j] - rho_vel_Rl[0][j]*vel_Rl[2][j]*vel_Rl[2][j] );
       
       ke_face_error_l[j] += 0.25*( uv_Rl[0][j]*rho_vel_Rl[1][j]
                                                      - uv_Rl[0][j]*rho_Rl[j]*vel_Rl[1][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[0][j]*rho_Rl[j]*vel_sq_Rl[1][j]
                                                      - vel_Rl[0][j]*rho_vel_Rl[1][j]*vel_Rl[1][j] );

       ke_face_error_l[j] += 0.25*( uv_Rl[1][j]*rho_vel_Rl[2][j]
                                                      - uv_Rl[1][j]*rho_Rl[j]*vel_Rl[2][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[0][j]*rho_Rl[j]*vel_sq_Rl[2][j]
                                                      - vel_Rl[0][j]*rho_vel_Rl[2][j]*vel_Rl[2][j] );


       ke_face_error_l[j] += 0.5*( rho_vel_Rl[1][j]*vel_sq_Rl[0][j] - rho_vel_Rl[1][j]*vel_Rl[0][j]*vel_Rl[0][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[1][j]*vel_sq_Rl[1][j] - rho_vel_Rl[1][j]*vel_Rl[1][j]*vel_Rl[1][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[1][j]*vel_sq_Rl[2][j] - rho_vel_Rl[1][j]*vel_Rl[2][j]*vel_Rl[2][j] );

       ke_face_error_l[j] += 0.25*( uv_Rl[0][j]*rho_vel_Rl[0][j]
                                                      - uv_Rl[0][j]*rho_Rl[j]*vel_Rl[0][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[1][j]*rho_Rl[j]*vel_sq_Rl[0][j]
                                                      - vel_Rl[1][j]*rho_vel_Rl[0][j]*vel_Rl[0][j] );

       ke_face_error_l[j] += 0.25*( uv_Rl[2][j]*rho_vel_Rl[2][j]
                                                      - uv_Rl[2][j]*rho_Rl[j]*vel_Rl[2][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[1][j]*rho_Rl[j]*vel_sq_Rl[2][j]
                                                      - vel_Rl[1][j]*rho_vel_Rl[2][j]*vel_Rl[2][j] );

       ke_face_error_l[j] += 0.5*( rho_vel_Rl[2][j]*vel_sq_Rl[0][j] - rho_vel_Rl[2][j]*vel_Rl[0][j]*vel_Rl[0][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[2][j]*vel_sq_Rl[1][j] - rho_vel_Rl[2][j]*vel_Rl[1][j]*vel_Rl[1][j] );
       ke_face_error_l[j] += 0.5*( rho_vel_Rl[2][j]*vel_sq_Rl[2][j] - rho_vel_Rl[2][j]*vel_Rl[2][j]*vel_Rl[2][j] );

       ke_face_error_l[j] += 0.25*( uv_Rl[1][j]*rho_vel_Rl[0][j]
                                                      - uv_Rl[1][j]*rho_Rl[j]*vel_Rl[0][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[2][j]*rho_Rl[j]*vel_sq_Rl[0][j]
                                                      - vel_Rl[2][j]*rho_vel_Rl[0][j]*vel_Rl[0][j] );

       ke_face_error_l[j] += 0.25*( uv_Rl[2][j]*rho_vel_Rl[1][j]
                                                      - uv_Rl[2][j]*rho_Rl[j]*vel_Rl[1][j] );
       ke_face_error_l[j] += 0.25*( vel_Rl[2][j]*rho_Rl[j]*vel_sq_Rl[1][j]
                                                    - vel_Rl[2][j]*rho_vel_Rl[1][j]*vel_Rl[1][j] );

   }
 

   Vector rho_vel_Rr[dim]; // Restriction right
   Vector vel_Rr[dim], rho_vel_sq_Rr[dim], rho_uv_Rr[dim];
   Vector vel_sq_Rr[dim], uv_Rr[dim];
   Vector pres_Rr(n_face_pts), rho_Rr(n_face_pts); 
   Vector rho_eu_Rr[dim], eu_Rr[dim], E_Rr(n_face_pts), e_Rr(n_face_pts); 
   Vector pu_Rr[dim]; 

   glob_proj_r->Mult(rho ,  rho_Rr );
   glob_proj_r->Mult(pres,  pres_Rr);
   glob_proj_r->Mult(e,                e_Rr);
   glob_proj_r->Mult(E,                E_Rr);
   for(int j = 0; j < dim; j++) 
   {
       rho_vel_Rr[j].SetSize(n_face_pts); vel_Rr[j].SetSize(n_face_pts); rho_uv_Rr[j].SetSize(n_face_pts);
       uv_Rr[j].SetSize(n_face_pts);
       rho_vel_sq_Rr[j].SetSize(n_face_pts); vel_sq_Rr[j].SetSize(n_face_pts);
       rho_eu_Rr[j].SetSize(n_face_pts); eu_Rr[j].SetSize(n_face_pts);
       pu_Rr[j].SetSize(n_face_pts); 

       glob_proj_r->Mult(rho_vel[j],     rho_vel_Rr[j]);
       glob_proj_r->Mult(vel[j],         vel_Rr[j]);
       glob_proj_r->Mult(rho_uv[j],      rho_uv_Rr[j]);
       glob_proj_r->Mult(uv[j],          uv_Rr[j]);
       glob_proj_r->Mult(rho_vel_sq[j],  rho_vel_sq_Rr[j]);
       glob_proj_r->Mult(v_sq[j],        vel_sq_Rr[j]);
       glob_proj_r->Mult(rho_eu[j],      rho_eu_Rr[j]);
       glob_proj_r->Mult(eu[j],          eu_Rr[j]);
       glob_proj_r->Mult(pu[j],          pu_Rr[j]);
   }

   ke_face_error_r = 0.0;
   for (int j = 0; j < n_face_pts; j++)
   {
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[0][j]*vel_sq_Rr[0][j] - rho_vel_Rr[0][j]*vel_Rr[0][j]*vel_Rr[0][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[0][j]*vel_sq_Rr[1][j] - rho_vel_Rr[0][j]*vel_Rr[1][j]*vel_Rr[1][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[0][j]*vel_sq_Rr[2][j] - rho_vel_Rr[0][j]*vel_Rr[2][j]*vel_Rr[2][j] );
       
       ke_face_error_r[j] += 0.25*( uv_Rr[0][j]*rho_vel_Rr[1][j]
                                                      - uv_Rr[0][j]*rho_Rr[j]*vel_Rr[1][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[0][j]*rho_Rr[j]*vel_sq_Rr[1][j]
                                                      - vel_Rr[0][j]*rho_vel_Rr[1][j]*vel_Rr[1][j] );

       ke_face_error_r[j] += 0.25*( uv_Rr[1][j]*rho_vel_Rr[2][j]
                                                      - uv_Rr[1][j]*rho_Rr[j]*vel_Rr[2][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[0][j]*rho_Rr[j]*vel_sq_Rr[2][j]
                                                      - vel_Rr[0][j]*rho_vel_Rr[2][j]*vel_Rr[2][j] );


       ke_face_error_r[j] += 0.5*( rho_vel_Rr[1][j]*vel_sq_Rr[0][j] - rho_vel_Rr[1][j]*vel_Rr[0][j]*vel_Rr[0][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[1][j]*vel_sq_Rr[1][j] - rho_vel_Rr[1][j]*vel_Rr[1][j]*vel_Rr[1][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[1][j]*vel_sq_Rr[2][j] - rho_vel_Rr[1][j]*vel_Rr[2][j]*vel_Rr[2][j] );

       ke_face_error_r[j] += 0.25*( uv_Rr[0][j]*rho_vel_Rr[0][j]
                                                      - uv_Rr[0][j]*rho_Rr[j]*vel_Rr[0][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[1][j]*rho_Rr[j]*vel_sq_Rr[0][j]
                                                      - vel_Rr[1][j]*rho_vel_Rr[0][j]*vel_Rr[0][j] );

       ke_face_error_r[j] += 0.25*( uv_Rr[2][j]*rho_vel_Rr[2][j]
                                                      - uv_Rr[2][j]*rho_Rr[j]*vel_Rr[2][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[1][j]*rho_Rr[j]*vel_sq_Rr[2][j]
                                                      - vel_Rr[1][j]*rho_vel_Rr[2][j]*vel_Rr[2][j] );

       ke_face_error_r[j] += 0.5*( rho_vel_Rr[2][j]*vel_sq_Rr[0][j] - rho_vel_Rr[2][j]*vel_Rr[0][j]*vel_Rr[0][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[2][j]*vel_sq_Rr[1][j] - rho_vel_Rr[2][j]*vel_Rr[1][j]*vel_Rr[1][j] );
       ke_face_error_r[j] += 0.5*( rho_vel_Rr[2][j]*vel_sq_Rr[2][j] - rho_vel_Rr[2][j]*vel_Rr[2][j]*vel_Rr[2][j] );

       ke_face_error_r[j] += 0.25*( uv_Rr[1][j]*rho_vel_Rr[0][j]
                                                      - uv_Rr[1][j]*rho_Rr[j]*vel_Rr[0][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[2][j]*rho_Rr[j]*vel_sq_Rr[0][j]
                                                      - vel_Rr[2][j]*rho_vel_Rr[0][j]*vel_Rr[0][j] );

       ke_face_error_r[j] += 0.25*( uv_Rr[2][j]*rho_vel_Rr[1][j]
                                                      - uv_Rr[2][j]*rho_Rr[j]*vel_Rr[1][j] );
       ke_face_error_r[j] += 0.25*( vel_Rr[2][j]*rho_Rr[j]*vel_sq_Rr[1][j]
                                                    - vel_Rr[2][j]*rho_vel_Rr[1][j]*vel_Rr[1][j] );

   }

   Vector face_vl(n_face_pts), face_vr(n_face_pts);
   double nor_l2, nor_loc[dim]; 
   for (int k = 0; k < n_face_pts; k++)
   {
       nor_l2 = 0.0;
       for (int j = 0; j < dim; j++)
       {
           nor_l2    += std::pow( nor_face(j*n_face_pts + k), 2 );
           nor_loc[j] = nor_face(j*n_face_pts + k);
       }
       nor_l2 = std::sqrt(nor_l2);

       double temp1 = 0.0; 
       double temp2 = 0.0; 
       for (int j = 0; j < dim; j++)
       {
           temp1 += ke_face_error_l[k]*nor_loc[j];
           temp2 += ke_face_error_r[k]*nor_loc[j];
       }
       ke_face_error_l[k] = temp1;
       ke_face_error_r[k] = temp2;
   }

   Vector face_l(n_face_pts), face_r(n_face_pts);
   Vector ke_corr_error(dofs), ke_l(dofs), ke_r(dofs), temp(dofs);

   getSparseMat(wts).Mult(ke_face_error_l, face_l);
   getSparseMat(wts).Mult(ke_face_error_r, face_r);
       
   glob_proj_l->MultTranspose(face_l, ke_l);
   face_t_r    .MultTranspose(face_r, ke_r);

   subtract(ke_l, ke_r, ke_corr_error); // We need to do F_l - F_r

   Vector one_v(dofs), mass_v(dofs), mass_rho(dofs);
   Vector m_vbar_sq(dofs), m_rho_vbar_sq(dofs);
   Vector m_vel[dim], m_rho_vel[dim], m_rho_vel_sq[dim];
   Vector m_E(dofs);
   
   one_v = 1.0;
   M.Mult(one_v,         mass_v);
   M.Mult(rho  ,         mass_rho);
   M.Mult(vbar_sq,       m_vbar_sq);
   M.Mult(rho_vbar_sq,   m_rho_vbar_sq);
   M.Mult(E,             m_E          );

   for (int j = 0; j < dim; j++)
   {
       m_vel[j].SetSize(dofs);   
       m_rho_vel[j].SetSize(dofs);   
       m_rho_vel_sq[j].SetSize(dofs);   

       M.Mult(vel[j] ,        m_vel[j]    );
       M.Mult(rho_vel[j] ,    m_rho_vel[j]    );
       M.Mult(rho_vel_sq[j],  m_rho_vel_sq[j]    );
   }

   double one_T_ke_corr_loc      = one_v*ke_corr_error; // 1^T R^T B (Ru^2 - R(u)*R(u))
   double rho_t_loc              = one_v*mass_rho;
   double one_T_m_loc            = one_v*mass_v;
   double one_T_vbar_sq_loc      = one_v*m_vbar_sq;
   double one_T_rho_vbar_sq_loc  = one_v*m_rho_vbar_sq;
   double one_T_E_loc            = one_v*m_E;
 
   double one_T_ke_corr, rho_t, one_T_m, one_T_vbar_sq, one_T_rho_vbar_sq, one_T_E;
   
   MPI_Allreduce(&one_T_ke_corr_loc,  &one_T_ke_corr,  1, MPI_DOUBLE, MPI_SUM, comm); // Get global across processors
   MPI_Allreduce(&rho_t_loc,          &rho_t,          1, MPI_DOUBLE, MPI_SUM, comm); 
   MPI_Allreduce(&one_T_m_loc,        &one_T_m,        1, MPI_DOUBLE, MPI_SUM, comm); 
   MPI_Allreduce(&one_T_vbar_sq_loc,  &one_T_vbar_sq,  1, MPI_DOUBLE, MPI_SUM, comm); 
   MPI_Allreduce(&one_T_E_loc,        &one_T_E,        1, MPI_DOUBLE, MPI_SUM, comm); 
   
   double rho_vel_t_loc[dim], one_T_vel_loc[dim], one_T_rho_vel_sq_loc[dim];
   double rho_vel_t[dim], one_T_vel[dim], one_T_rho_vel_sq[dim];
   for (int j = 0; j < dim; j++)
   {
       rho_vel_t_loc[j]          = one_v*m_rho_vel[j];
       one_T_vel_loc[j]          = one_v*m_vel[j]    ;
       one_T_rho_vel_sq_loc[j]   = one_v*m_rho_vel_sq[j]    ;
   }
   for (int j = 0; j < dim; j++)
   {
       MPI_Allreduce(&rho_vel_t_loc[j],         &rho_vel_t[j],  1, MPI_DOUBLE, MPI_SUM, comm); 
       MPI_Allreduce(&one_T_vel_loc[j],         &one_T_vel[j],  1, MPI_DOUBLE, MPI_SUM, comm); 
       MPI_Allreduce(&one_T_rho_vel_sq_loc[j],  &one_T_rho_vel_sq[j],  1, MPI_DOUBLE, MPI_SUM, comm); 
   }
   
   rho_t     = rho_t/one_T_m;
   
   for (int j = 0; j < dim; j++)
   {
       rho_vel_t[j] = rho_vel_t[j]/one_T_m;
   }

   if ( std::abs(one_T_ke_corr) < 1E-14 )
       one_T_ke_corr = 0.0;
   
   double corr_alpha1 = 2.0*one_T_ke_corr/(one_T_rho_vbar_sq - rho_t*one_T_vbar_sq + 1E-16); // Rho
   double corr_alpha2[dim];
   for (int j = 0; j < dim; j++)
       corr_alpha2[j] = one_T_ke_corr/(one_T_rho_vel_sq[j] - rho_vel_t[j]*one_T_vel[j] + 1E-16); // Rho U
  
   Vector temp_ke(dofs);

   f_ke_corr.SetSize(var_dim*dofs);
   f_ke_corr = 0.0;

   {
       temp_ke  = rho;
       temp_ke -= rho_t;
       temp_ke *= 0.00*corr_alpha1;
       
       M.Mult(temp_ke, temp);
       temp_ke = temp;
       
       f_ke_corr.SetSubVector(offsets[0], temp_ke);
   }

   double coeff;
   for (int j = 0; j < dim; j++)
   {
       temp_ke  =  rho_vel[j];
       temp_ke -=  rho_vel_t[j];

       coeff = 0.0;
//       if (mach_max > 0.45)
//           coeff = mach_max;
       temp_ke *=  coeff*corr_alpha2[j];
       
       M.Mult(temp_ke, temp);
       temp_ke = temp;
       
       f_ke_corr.SetSubVector(offsets[1 + j], temp_ke);
   }


}




// Get max signal speed 
double getMMax(int dim, const Vector &u)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}

    int offset = u.Size()/var_dim;

    double m_max = 0;

    double rho, vel[dim];
    for(int j = 0; j < offset; j++)
    {
        rho = u[j];
    
        for(int i = 0; i < dim; i++) vel[i] = u[(1 + i)*offset + j]/rho;

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

        m_max = sqrt(v_sq)/a;
    }
    return m_max;
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
    
        for(int i = 0; i < dim; i++) vel[i] = u[(1 + i)*offset + j]/rho;

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



void getPRhoMinMax(int dim, const Vector &u, double &rho_min, double &p_min, double &rho_max, double &p_max)
{
    int var_dim = dim + 2; 
    int aux_dim = dim + 1; // Auxilliary variables are {u, v, w, T}

    int offset = u.Size()/var_dim;

    rho_min =  u[0];
    rho_max =  u[0];
    p_min   =  1.0E16;
    p_max   = -1.0E16;

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

        rho_min = std::min(rho_min, rho);
        rho_max = std::max(rho_max, rho);
        p_min   = std::min(p_min,   p);
        p_max   = std::max(p_max,   p);
    }

}



SparseMatrix getSparseMat(const Vector &x)
{
    int size = x.Size();

    SparseMatrix a(size, size);

    for(int i = 0; i < size; i++)
        a.Set(i, i, x(i));

    return a;
}





void getKGSplitDx(int dim, 
        const HypreParMatrix &K_x, const HypreParMatrix &K_y, const HypreParMatrix &K_z, 
        const Vector &u, Vector &f_dx) 
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

    Vector vel[dim], rho_vel_sq[dim], rho_uv[dim], uv[dim], v_sq[dim];
    for(int i = 0; i < dim; i++) 
    {
        vel[i].SetSize(offset); rho_vel_sq[i].SetSize(offset); rho_uv[i].SetSize(offset);
        uv[i].SetSize(offset);
        v_sq[i].SetSize(offset);
    }

    Vector pres(offset); // (rho*Cv*T + p)*u, p
    Vector pu[dim]; 
    Vector e(offset), eu[dim], rho_eu[dim]; 
    for(int i = 0; i < dim; i++) 
    {
        eu[i]    .SetSize(offset);
        rho_eu[i].SetSize(offset);
        pu[i]    .SetSize(offset);
    }

    for(int i = 0; i < offset; i++)
    {
        double vel_sq = 0.0;
        for(int j = 0; j < dim; j++)
        {
            vel[j][i]        = rho_vel[j](i)/rho(i);
            vel_sq          += pow(vel[j][i], 2);

            rho_vel_sq[j][i] = rho_vel[j](i)*vel[j](i);
            v_sq[j][i]       = vel[j](i)*vel[j](i);
        }

        pres(i)          = (E(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);
        e(i)             =  E(i)/rho(i);

        for(int j = 0; j < dim; j++)
        {
            eu[j](i)            = e(i)*vel[j](i);
            rho_eu[j](i)        = E(i)*vel[j](i);
            
            pu[j](i)            = pres(i)*vel[j](i);
        }

        rho_uv[0](i)  = rho_vel[0](i)*vel[1](i); // rho*u*v
        rho_uv[1](i)  = rho_vel[0](i)*vel[2](i); // rho*u*w
        rho_uv[2](i)  = rho_vel[1](i)*vel[2](i); // rho*v*w
    
        uv[0](i)      = vel[0](i)*vel[1](i); // rho*u*v
        uv[1](i)      = vel[0](i)*vel[2](i); // rho*u*w
        uv[2](i)      = vel[1](i)*vel[2](i); // rho*v*w

    }

    Vector rho_vel_dx[dim];
    Vector vel_dx[dim], rho_vel_sq_dx(offset), rho_uv_dx[dim];
    Vector vel_sq_dx(offset), uv_dx[dim];
    Vector pres_dx(offset), rho_dx(offset); 
    Vector rho_eu_dx(offset), eu_dx(offset), E_dx(offset), e_dx(offset); 
    Vector pu_dx(offset); 

    K_x.Mult(rho ,  rho_dx );
    K_x.Mult(pres,  pres_dx);
    K_x.Mult(pu[0], pu_dx);
    K_x.Mult(rho_eu[0],        rho_eu_dx);
    K_x.Mult(eu[0],            eu_dx);
    K_x.Mult(e,                e_dx);
    K_x.Mult(E,                E_dx);
    K_x.Mult(rho_vel_sq[0],    rho_vel_sq_dx);
    K_x.Mult(v_sq[0],          vel_sq_dx);
    for(int j = 0; j < dim; j++) 
    {
        rho_vel_dx[j].SetSize(offset); vel_dx[j].SetSize(offset); rho_uv_dx[j].SetSize(offset);
        uv_dx[j].SetSize(offset);

        K_x.Mult(rho_vel[j],    rho_vel_dx[j]);
        K_x.Mult(vel[j],        vel_dx[j]);
        K_x.Mult(rho_uv[j],     rho_uv_dx[j]);
        K_x.Mult(uv[j],         uv_dx[j]);
    }

    Vector fx(var_dim*offset);
    fx = 0.0;

    Vector temp(offset), temp1(offset), temp2(offset), temp3(offset);

    SparseMatrix rho_mat    = getSparseMat(rho   );
    
    {
        // X - derivatives
        
        // rho
        
        temp1  = rho_vel_dx[0];
        rho_mat.Mult(vel_dx[0], temp2);
        temp1 += temp2;
        getSparseMat(vel[0]).Mult(rho_dx,    temp3);
        temp1 += temp3;
        
        temp1 *= 0.5;
        
        fx.SetSubVector(offsets[0], temp1);
        
        // rho_u
        temp   = rho_vel_sq_dx;

        rho_mat.              Mult(vel_sq_dx,     temp1);
        getSparseMat(v_sq[0]).Mult(rho_dx,        temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[0])    .Mult(rho_vel_dx[0], temp1);
        getSparseMat(rho_vel[0]).Mult(vel_dx[0],     temp2);

        temp1 += temp2;
        temp1 *= 2.;

        temp  += temp1;
        temp  *= 0.25;
        temp  += pres_dx;
        
        fx.SetSubVector(offsets[1], temp );
        
        // rho_v
        temp   = rho_uv_dx[0]; 

        rho_mat.            Mult(uv_dx[0],      temp1);
        getSparseMat(uv[0]).Mult(rho_dx,        temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[1])    .Mult(rho_vel_dx[0], temp1);
        getSparseMat(rho_vel[0]).Mult(vel_dx[1],     temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[0])    .Mult(rho_vel_dx[1], temp1);
        getSparseMat(rho_vel[1]).Mult(vel_dx[0],     temp2);

        temp += temp1; temp += temp2; // This is probably done

        temp  *= 0.25;
 
        fx.SetSubVector(offsets[2], temp);
        
        // rho_w
        temp   = rho_uv_dx[1]; 

        getSparseMat(rho_vel[0]).Mult(vel_dx[2],     temp1);
        getSparseMat(    vel[2]).Mult(rho_vel_dx[0], temp2);

        temp += temp1; temp += temp2;
 
        getSparseMat(rho_vel[2]).Mult(vel_dx[0],     temp1);
        getSparseMat(    vel[0]).Mult(rho_vel_dx[2], temp2);

        temp += temp1; temp += temp2;

        rho_mat            .Mult(uv_dx[1],  temp1);
        getSparseMat(uv[1]).Mult(rho_dx,    temp2);

        temp += temp1; temp += temp2;

        temp *= 0.25;
     
        fx.SetSubVector(offsets[3], temp);
    
        // rho_e
        
        temp = rho_eu_dx;
        
        getSparseMat(E)     .Mult(vel_dx[0], temp1);
        getSparseMat(vel[0]).Mult(E_dx,      temp2);
    
        temp += temp1; temp += temp2;
    
        getSparseMat(rho_vel[0]).Mult(e_dx,           temp1);
        getSparseMat(    e)     .Mult(rho_vel_dx[0],  temp2);
        
        temp += temp1; temp += temp2;
     
        rho_mat            .Mult(eu_dx,   temp1);
        getSparseMat(eu[0]).Mult(rho_dx,  temp2);
        
        temp += temp1; temp += temp2;
    
        temp *= 0.25;

        temp1 = pu_dx;

        getSparseMat(pres)  .Mult(vel_dx[0], temp2);
        getSparseMat(vel[0]).Mult(pres_dx,   temp3);

        temp1 += temp2; temp1 += temp3;

        temp1 *= 0.5;

        temp  += temp1;
     
        fx.SetSubVector(offsets[4], temp);
        
    }

    K_y.Mult(rho , rho_dx );
    K_y.Mult(pres, pres_dx);
    K_y.Mult(pu[1], pu_dx);
    K_y.Mult(rho_eu[1],        rho_eu_dx);
    K_y.Mult(eu[1],            eu_dx);
    K_y.Mult(e,                e_dx);
    K_y.Mult(E,                E_dx);
    K_y.Mult(rho_vel_sq[1],    rho_vel_sq_dx);
    K_y.Mult(v_sq[1],          vel_sq_dx);
    for(int j = 0; j < dim; j++) 
    {
        K_y.Mult(rho_vel[j],    rho_vel_dx[j]);
        K_y.Mult(vel[j],        vel_dx[j]);
        K_y.Mult(rho_uv[j],     rho_uv_dx[j]);
        K_y.Mult(uv[j],         uv_dx[j]);
    }

    {
        // Y -deri

        temp1  = rho_vel_dx[1];
        getSparseMat(rho   ).Mult(vel_dx[1], temp2);
        temp1 += temp2;
        getSparseMat(vel[1]).Mult(rho_dx,    temp3);
        temp1 += temp3;
        
        temp1 *= 0.5;
        
        fx.AddElementVector(offsets[0], temp1); // rho
    
        // rho_u
        temp   = rho_uv_dx[0];

        rho_mat.            Mult(uv_dx[0],     temp1);
        getSparseMat(uv[0]).Mult(rho_dx,       temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[0])    .Mult(rho_vel_dx[1], temp1);
        getSparseMat(rho_vel[1]).Mult(vel_dx[0],     temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[1])    .Mult(rho_vel_dx[0], temp1);
        getSparseMat(rho_vel[0]).Mult(vel_dx[1],     temp2);
        
        temp += temp1; temp += temp2;

        temp  *= 0.25;
        
        fx.AddElementVector(offsets[1], temp );

        // rho_v
        temp   = rho_vel_sq_dx; 

        rho_mat.              Mult(vel_sq_dx,     temp1);
        getSparseMat(v_sq[1]).Mult(rho_dx,        temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[1])    .Mult(rho_vel_dx[1], temp1);
        getSparseMat(rho_vel[1]).Mult(vel_dx[1],     temp2);

        temp1 += temp2;
        temp1 *= 2.0;

        temp += temp1;

        temp  *= 0.25;

        temp  += pres_dx;
        
        fx.AddElementVector(offsets[2], temp);
        
        // rho_w
        temp   = rho_uv_dx[2]; 

        getSparseMat(rho_vel[1]).Mult(vel_dx[2],     temp1);
        getSparseMat(    vel[2]).Mult(rho_vel_dx[1], temp2);

        temp += temp1; temp += temp2;
 
        getSparseMat(rho_vel[2]).Mult(vel_dx[1],     temp1);
        getSparseMat(    vel[1]).Mult(rho_vel_dx[2], temp2);

        temp += temp1; temp += temp2;

        rho_mat            .Mult(uv_dx[2],  temp1);
        getSparseMat(uv[2]).Mult(rho_dx,    temp2);

        temp += temp1; temp += temp2;

        temp *= 0.25;
 
        fx.AddElementVector(offsets[3], temp);
        
        // rho_e
        
        temp = rho_eu_dx;
        
        getSparseMat(E)     .Mult(vel_dx[1], temp1);
        getSparseMat(vel[1]).Mult(E_dx,      temp2);
    
        temp += temp1; temp += temp2;
    
        getSparseMat(rho_vel[1]).Mult(e_dx,           temp1);
        getSparseMat(    e)     .Mult(rho_vel_dx[1],  temp2);
        
        temp += temp1; temp += temp2;
     
        rho_mat            .Mult(eu_dx,   temp1);
        getSparseMat(eu[1]).Mult(rho_dx,  temp2);
        
        temp += temp1; temp += temp2;
    
        temp *= 0.25;

        temp1 = pu_dx;

        getSparseMat(pres)  .Mult(vel_dx[1], temp2);
        getSparseMat(vel[1]).Mult(pres_dx,   temp3);

        temp1 += temp2; temp1 += temp3;

        temp1 *= 0.5;

        temp  += temp1;

        fx.AddElementVector(offsets[4], temp);
    }


    K_z.Mult(rho , rho_dx );
    K_z.Mult(pres, pres_dx);
    K_z.Mult(pu[2], pu_dx);
    K_z.Mult(rho_eu[2],        rho_eu_dx);
    K_z.Mult(eu[2],            eu_dx);
    K_z.Mult(e,                e_dx);
    K_z.Mult(E,                E_dx);
    K_z.Mult(rho_vel_sq[2],    rho_vel_sq_dx);
    K_z.Mult(v_sq[2],          vel_sq_dx);
    for(int j = 0; j < dim; j++) 
    {
        K_z.Mult(rho_vel[j],    rho_vel_dx[j]);
        K_z.Mult(vel[j],        vel_dx[j]);
        K_z.Mult(rho_uv[j],     rho_uv_dx[j]);
        K_z.Mult(uv[j],         uv_dx[j]);
    }
    
    {
        // Z -deri
        temp1  = rho_vel_dx[2];
        getSparseMat(rho   ).Mult(vel_dx[2], temp2);
        temp1 += temp2;
        getSparseMat(vel[2]).Mult(rho_dx,    temp3);
        temp1 += temp3;
        
        temp1 *= 0.5;
        
        fx.AddElementVector(offsets[0], temp1); // rho
        
        // rho_u
        temp   = rho_uv_dx[1];

        rho_mat.            Mult(uv_dx[1],     temp1);
        getSparseMat(uv[1]).Mult(rho_dx,       temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[0])    .Mult(rho_vel_dx[2], temp1);
        getSparseMat(rho_vel[2]).Mult(vel_dx[0],     temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[2])    .Mult(rho_vel_dx[0], temp1);
        getSparseMat(rho_vel[0]).Mult(vel_dx[2],     temp2);
        
        temp += temp1; temp += temp2;

        temp  *= 0.25;
        
        fx.AddElementVector(offsets[1], temp);

        // rho_v
        temp   = rho_uv_dx[2]; 

        rho_mat.            Mult(uv_dx[2],      temp1);
        getSparseMat(uv[2]).Mult(rho_dx,        temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[2])    .Mult(rho_vel_dx[1], temp1);
        getSparseMat(rho_vel[1]).Mult(vel_dx[2],     temp2);

        temp += temp1; temp += temp2;

        getSparseMat(vel[1])    .Mult(rho_vel_dx[2], temp1);
        getSparseMat(rho_vel[2]).Mult(vel_dx[1],     temp2);

        temp += temp1; temp += temp2; // This is probably done

        temp  *= 0.25;
 
        fx.AddElementVector(offsets[2], temp);
        
        // rho_w
        temp   = rho_vel_sq_dx; 

        getSparseMat(rho_vel[2]).Mult(vel_dx[2],     temp1);
        getSparseMat(    vel[2]).Mult(rho_vel_dx[2], temp2);

        temp1 += temp2;
        temp1 *= 2.;
        temp  += temp1;

        rho_mat              .Mult(vel_sq_dx, temp1);
        getSparseMat(v_sq[2]).Mult(rho_dx,    temp2);

        temp  += temp1; temp += temp2;

        temp  *= 0.25;
 
        temp  += pres_dx;
        
        fx.AddElementVector(offsets[3], temp);
        
        // rho_e
     
        temp = rho_eu_dx;
        
        getSparseMat(E)     .Mult(vel_dx[2], temp1);
        getSparseMat(vel[2]).Mult(E_dx,      temp2);
    
        temp += temp1; temp += temp2;
    
        getSparseMat(rho_vel[2]).Mult(e_dx,           temp1);
        getSparseMat(    e)     .Mult(rho_vel_dx[2],  temp2);
        
        temp += temp1; temp += temp2;
     
        rho_mat            .Mult(eu_dx,   temp1);
        getSparseMat(eu[2]).Mult(rho_dx,  temp2);
        
        temp += temp1; temp += temp2;
    
        temp *= 0.25;

        temp1 = pu_dx;

        getSparseMat(pres)  .Mult(vel_dx[2], temp2);
        getSparseMat(vel[2]).Mult(pres_dx,   temp3);

        temp1 += temp2; temp1 += temp3;

        temp1 *= 0.5;

        temp  += temp1;

        fx.AddElementVector(offsets[4], temp);
    }

    f_dx = fx;

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

       double amp    =  0.8 ;

       u1  =10.9*pow( (1 - pow( x(1) - 1, 2)), 2);
       u2  = amp*exp(-pow((x(0) - M_PI)/(2*M_PI), 2))*exp(-pow((x(1)/2.0), 2))*cos(4*M_PI*x(2)/M_PI);
       u3  = 0.0;
       p   = (pow(mdot_pres/M_inf, 2))/(gamm);
       
       u1 += amp*sin(20*M_PI*x(1)/2)*sin(20*M_PI*x(2)/M_PI);
       u1 += amp*sin(30*M_PI*x(1)/2)*sin(30*M_PI*x(2)/M_PI);
       u1 += amp*sin(40*M_PI*x(1)/2)*sin(40*M_PI*x(2)/M_PI);

       u2 += amp*sin(20*M_PI*x(0)/(2*M_PI))*sin(20*M_PI*x(2)/M_PI);
       u2 += amp*sin(30*M_PI*x(0)/(2*M_PI))*sin(30*M_PI*x(2)/M_PI);
       u2 += amp*sin(40*M_PI*x(0)/(2*M_PI))*sin(40*M_PI*x(2)/M_PI);

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
       double rho, u1, u2, u3, p;
       /*
        * Discrete filter operators for large-eddy simulation using high-order spectral difference methods
        */
       rho = 1;

       double amp = 0.1;
       
       u1  =  4.5;
       u2  = 0.0;
       u3  = 0.0;
       p   = (pow(mdot_pres/M_inf, 2))/(R_gas*gamm);

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
       v(3) = (pow(mdot_pres/M_inf, 2))/(R_gas*gamm);
   }
   else if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = (pow(mdot_pres/M_inf, 2))/(R_gas*gamm);
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
 * Get Interaction flux using the local KEP solver 
 */
void getVectorKGFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f_com)
{
    int var_dim = dim + 2;
    int num_pts = u1.Size()/var_dim;

    double Cv   = R/(gamm - 1);
    
    double rho_L, E_L, e_L, vel_sq_L, T_L, p_L, a_L, h_L, sqrtRho_L, vnl, mach_L;
    double rho_R, E_R, e_R, vel_sq_R, T_R, p_R, a_R, h_R, sqrtRho_R, vnr, mach_R;
    double beta_L, beta_R;
    double vsl, vsr, u_max;
    Vector vel_L(dim);
    Vector vel_R(dim);

    double sSqrtRho, vel_sq_rho, r_rho, rho_h, rho_a, vn_rho;
    Vector velRho(dim);

    double drho, dp, dvn, du, dv, dw;
    Vector LdU(4), ws(4), dws(4);

    DenseMatrix roeM(dim + 2, 4);
    Vector diss(var_dim); // dissipation 

    double LM_z;
    double dlambda1, dlambda3;

    Vector nor_in(dim), nor_dim(dim);
    double nor_l2;

    bnd_u_max = 0.0;
    mach_max  = 0.0;

    for(int p = 0; p < num_pts; p++)
    {
        rho_L = u1(p);
        
        for (int i = 0; i < dim; i++)
        {
            vel_L(i) = u1((1 + i)*num_pts + p)/rho_L;    
        }
        E_L   = u1((var_dim - 1)*num_pts + p);
        e_L   = E_L/rho_L; 

        vel_sq_L = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vel_sq_L += pow(vel_L(i), 2) ;
        }
        T_L        = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
        p_L        = rho_L*R*T_L;
        a_L        = sqrt(gamm * R * T_L);
        h_L        = (E_L + p_L)/rho_L;
        sqrtRho_L  = std::sqrt(rho_L);
        beta_L     = 1./(2*R_gas*T_L);
        mach_L     = std::sqrt(vel_sq_L)/a_L;

        rho_R = u2(p);
        for (int i = 0; i < dim; i++)
        {
            vel_R(i) = u2((1 + i)*num_pts + p)/rho_R;    
        }
        E_R   = u2((var_dim - 1)*num_pts + p);
        e_R   = E_R/rho_R; 
    
        vel_sq_R = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vel_sq_R += pow(vel_R(i), 2) ;
        }
        T_R       = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
        p_R       = rho_R*R*T_R;
        a_R       = sqrt(gamm * R * T_R);
        h_R       = (E_R + p_R)/rho_R;
        sqrtRho_R = std::sqrt(rho_R);
        beta_R     = 1./(2*R_gas*T_R);
        mach_R     = std::sqrt(vel_sq_R)/a_R;
        
        bnd_u_max = max(bnd_u_max, sqrt(vel_sq_R) + a_R);
        bnd_u_max = max(bnd_u_max, sqrt(vel_sq_L) + a_L);
 
        mach_max = max(mach_max, sqrt(vel_sq_R)/a_R);
        mach_max = max(mach_max, sqrt(vel_sq_L)/a_L);

        for (int i = 0; i < dim; i++)
        {
            double f_mass                                 = 0.25*(rho_L + rho_R)*(vel_L(i) + vel_R(i));
            f_com((i*var_dim + 0)*num_pts + p)            = f_mass;
            
            for (int j = 0; j < dim; j++)
                f_com((i*var_dim + 1 + j)*num_pts + p)    = 0.5*(vel_L(j) + vel_R(j))*f_mass;
            
            f_com((i*var_dim + 1 + i)*num_pts + p)       += 0.5*(p_L + p_R) ;

            f_com((i*var_dim + var_dim - 1)*num_pts + p)  = 0.25*(p_L + p_R)*(vel_L(i) + vel_R(i));
            f_com((i*var_dim + var_dim - 1)*num_pts + p) += 0.125*(rho_L + rho_R)*(e_L + e_R)*
                                                                 (vel_L(i) + vel_R(i));
           
        }
        
    }

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
        T_L        = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
        p_L        = rho_L*R*T_L;
        a_L        = sqrt(gamm * R * T_L);
        h_L        = (E_L + p_L)/rho_L;
        sqrtRho_L  = std::sqrt(rho_L);

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
        T_R       = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
        p_R       = rho_R*R*T_R;
        a_R       = sqrt(gamm * R * T_R);
        h_R       = (E_R + p_R)/rho_R;
        sqrtRho_R = std::sqrt(rho_R);

        for (int i = 0; i < dim; i++)
        {
            nor_in(i) = nor(i*num_pts + p);
        }
        nor_l2 = nor_in.Norml2();
        nor_dim.Set(1/nor_l2, nor_in);

        vnl   = 0.0, vnr = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vnl += vel_L[i]*nor_dim(i); 
            vnr += vel_R[i]*nor_dim(i); 
        }

        sSqrtRho = 1./(sqrtRho_L + sqrtRho_R); 
        vel_sq_rho = 0.0;;
        for (int i = 0; i < dim; i++)
        {
            velRho(i)   = (sqrtRho_L*vel_L(i) + sqrtRho_R*vel_R(i))*sSqrtRho;
            vel_sq_rho += velRho(i)*velRho(i);
        }
        r_rho = (sqrtRho_L*rho_L + sqrtRho_R*rho_R)*sSqrtRho;
        rho_h = (sqrtRho_L*h_L   + sqrtRho_R*h_R)*sSqrtRho;
        rho_a = std::sqrt((gamm - 1)*(rho_h - 0.5*vel_sq_rho));

        vn_rho   = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vn_rho += velRho[i]*nor_dim(i); 
        }

       //Wave Strengths

        drho = rho_R - rho_L ;//Density difference
        dp   = p_R - p_L     ;//Pressure difference
        dvn  = vnr - vnl     ;//Normal velocity difference

        LdU(0) = (dp - r_rho*rho_a*dvn )/(2.*rho_a*rho_a); //Left-moving acoustic wave strength
        LdU(1) =  drho - dp/(rho_a*rho_a);                 //Entropy wave strength
        LdU(2) = (dp + r_rho*rho_a*dvn )/(2.*rho_a*rho_a); //Right-moving acoustic wave strength
        LdU(3) = r_rho;                                    //Shear wave strength 

        // Absolute values of the wave Speeds
        ws(0) = std::abs(vn_rho - rho_a) ;//Left-moving acoustic wave
        ws(1) = std::abs(vn_rho);         //Entropy wave
        ws(2) = std::abs(vn_rho + rho_a) ;//Right-moving acoustic wave
        ws(3) = std::abs(vn_rho) ;        //Shear waves

        //Harten's Entropy Fix JCP(1983), 49, pp357-393: only for the nonlinear fields.
        //NOTE: It avoids vanishing wave speeds by making a parabolic fit near ws = 0.
        
        dws(0) = 0.2; 
        if ( ws(0) < dws(0) ) 
             ws(0) = 0.5 * ( ws(0)*ws(0)/dws(0)+dws(0) );
        dws(2) = 0.2; 
        if ( ws(2) < dws(2) ) 
             ws(3) = 0.5 * ( ws(2)*ws(2)/dws(2)+dws(2) );
        
        //Right Eigenvectors
        //Note: Two shear wave components are combined into one, so that tangent vectors
        //      are not required. And that's why there are only 4 vectors here.
        //      See "I do like CFD, VOL.1" about how tangent vectors are eliminated.
        
        //  Left-moving acoustic wave
        
        roeM(0,0) = 1.; 
        roeM(1,0) = velRho(0) - rho_a*nor_dim(0);
        roeM(2,0) = velRho(1) - rho_a*nor_dim(1);  
        roeM(3,0) = velRho(2) - rho_a*nor_dim(2);
        roeM(4,0) = rho_h - rho_a*vn_rho;
        
        // Entropy wave
           
        roeM(0,1) = 1.; 
        roeM(1,1) = velRho(0);
        roeM(2,1) = velRho(1);
        roeM(3,1) = velRho(2); 
        roeM(4,1) = 0.5*vel_sq_rho;
        
        // Right-moving acoustic wave
        
        roeM(0,2) = 1.; 
        roeM(1,2) = velRho(0) + rho_a*nor_dim(0);
        roeM(2,2) = velRho(1) + rho_a*nor_dim(1);  
        roeM(3,2) = velRho(2) + rho_a*nor_dim(2);
        roeM(4,2) = rho_h + rho_a*vn_rho;

        // Two shear wave components combined into one (wave strength incorporated).
          
        du = vel_R(0) - vel_L(0);
        dv = vel_R(1) - vel_L(1);
        dw = vel_R(2) - vel_L(2);

        roeM(0,3) = 0.; 
        roeM(1,3) = du - dvn*nor_dim(0);
        roeM(2,3) = dv - dvn*nor_dim(1);
        roeM(3,3) = dw - dvn*nor_dim(2);
        roeM(4,3) = velRho(0)*du + velRho(1)*dv + velRho(2)*dw - vn_rho*dvn;

        // Dissipation Term: |An|(UR-UL) = R|Lambda|L*dU = sum_k of [ ws(k) * R(:,k) * L*dU(k) ]
        
        for(int j = 0; j < var_dim; j++)
            diss(j) = ws(0)*LdU(0)*roeM(j,0) + ws(1)*LdU(1)*roeM(j,1) 
             + ws(2)*LdU(2)*roeM(j,2) + ws(3)*LdU(3)*roeM(j,3);

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < var_dim; j++)
            {
                f_com((i*var_dim + j)*num_pts + p) += 
                    -0.5*nor_dim(i)*diss(j);
            }
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




/*
 * Get Interaction flux using the local Roe solver 
 * solver, u1 is the left value and u2 the right value
 * Copied from Dr. Katate Masatsuka (info[at]cfdbooks.com),
 */
void getVectorRoeFlux(const double R, const double gamm, const int dim, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f_com)
{
    int var_dim = dim + 2;
    int num_pts = u1.Size()/var_dim;

    double Cv   = R/(gamm - 1);

    double rho_L, E_L, vel_sq_L, T_L, p_L, a_L, h_L, sqrtRho_L, vnl, mach_L;
    double rho_R, E_R, vel_sq_R, T_R, p_R, a_R, h_R, sqrtRho_R, vnr, mach_R;
    Vector vel_L(dim);
    Vector vel_R(dim);

    double sSqrtRho, vel_sq_rho, r_rho, rho_h, rho_a, vn_rho;
    Vector velRho(dim);

    double drho, dp, dvn, du, dv, dw;
    Vector LdU(4), ws(4), dws(4);

    DenseMatrix roeM(dim + 2, 4);
    Vector diss(var_dim); // dissipation 

    Vector nor_in(dim), nor_dim(dim);
    double nor_l2;

    Vector fl(dim*var_dim*num_pts), fr(dim*var_dim*num_pts);
    getInvFlux(dim, u1, fl);
    getInvFlux(dim, u2, fr);
    add(0.5, fl, fr, f_com);

    bnd_u_max = 0.0;
    mach_max  = 0.0;

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
        T_L        = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
        p_L        = rho_L*R*T_L;
        a_L        = sqrt(gamm * R * T_L);
        h_L        = (E_L + p_L)/rho_L;
        sqrtRho_L  = std::sqrt(rho_L);
        mach_L     = std::sqrt(vel_sq_L)/a_L;

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
        T_R       = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
        p_R       = rho_R*R*T_R;
        a_R       = sqrt(gamm * R * T_R);
        h_R       = (E_R + p_R)/rho_R;
        sqrtRho_R = std::sqrt(rho_R);
        mach_R    = std::sqrt(vel_sq_R)/a_R;

        bnd_u_max = max(bnd_u_max, sqrt(vel_sq_R) + a_R);
        bnd_u_max = max(bnd_u_max, sqrt(vel_sq_L) + a_L);
 
        mach_max = max(mach_max, sqrt(vel_sq_R)/a_R);
        mach_max = max(mach_max, sqrt(vel_sq_L)/a_L);

        for (int i = 0; i < dim; i++)
        {
            nor_in(i) = nor(i*num_pts + p);
        }
        nor_l2 = nor_in.Norml2();
        nor_dim.Set(1/nor_l2, nor_in);

        vnl   = 0.0, vnr = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vnl += vel_L[i]*nor_dim(i); 
            vnr += vel_R[i]*nor_dim(i); 
        }

        sSqrtRho = 1./(sqrtRho_L + sqrtRho_R); 
        vel_sq_rho = 0.0;;
        for (int i = 0; i < dim; i++)
        {
            velRho(i)   = (sqrtRho_L*vel_L(i) + sqrtRho_R*vel_R(i))*sSqrtRho;
            vel_sq_rho += velRho(i)*velRho(i);
        }
        r_rho = (sqrtRho_L*rho_L + sqrtRho_R*rho_R)*sSqrtRho;
        rho_h = (sqrtRho_L*h_L   + sqrtRho_R*h_R)*sSqrtRho;
        rho_a = std::sqrt((gamm - 1)*(rho_h - 0.5*vel_sq_rho));

        vn_rho   = 0.0;
        for (int i = 0; i < dim; i++)
        {
            vn_rho += velRho[i]*nor_dim(i); 
        }

       //Wave Strengths

        drho = rho_R - rho_L ;//Density difference
        dp   = p_R - p_L     ;//Pressure difference
        dvn  = vnr - vnl     ;//Normal velocity difference

        LdU(0) = (dp - r_rho*rho_a*dvn )/(2.*rho_a*rho_a); //Left-moving acoustic wave strength
        LdU(1) =  drho - dp/(rho_a*rho_a);                 //Entropy wave strength
        LdU(2) = (dp + r_rho*rho_a*dvn )/(2.*rho_a*rho_a); //Right-moving acoustic wave strength
        LdU(3) = r_rho;                                    //Shear wave strength 

        // Absolute values of the wave Speeds
        ws(0) = std::abs(vn_rho - rho_a) ;//Left-moving acoustic wave
        ws(1) = std::abs(vn_rho);         //Entropy wave
        ws(2) = std::abs(vn_rho + rho_a) ;//Right-moving acoustic wave
        ws(3) = std::abs(vn_rho) ;        //Shear waves

        //Harten's Entropy Fix JCP(1983), 49, pp357-393: only for the nonlinear fields.
        //NOTE: It avoids vanishing wave speeds by making a parabolic fit near ws = 0.
        
        dws(0) = 0.2; 
        if ( ws(0) < dws(0) ) 
             ws(0) = 0.5 * ( ws(0)*ws(0)/dws(0)+dws(0) );
        dws(2) = 0.2; 
        if ( ws(2) < dws(2) ) 
             ws(3) = 0.5 * ( ws(2)*ws(2)/dws(2)+dws(2) );
        
        //Right Eigenvectors
        //Note: Two shear wave components are combined into one, so that tangent vectors
        //      are not required. And that's why there are only 4 vectors here.
        //      See "I do like CFD, VOL.1" about how tangent vectors are eliminated.
        
        //  Left-moving acoustic wave
        
        roeM(0,0) = 1.; 
        roeM(1,0) = velRho(0) - rho_a*nor_dim(0);
        roeM(2,0) = velRho(1) - rho_a*nor_dim(1);  
        roeM(3,0) = velRho(2) - rho_a*nor_dim(2);
        roeM(4,0) = rho_h - rho_a*vn_rho;
        
        // Entropy wave
           
        roeM(0,1) = 1.; 
        roeM(1,1) = velRho(0);
        roeM(2,1) = velRho(1);
        roeM(3,1) = velRho(2); 
        roeM(4,1) = 0.5*vel_sq_rho;
        
        // Right-moving acoustic wave
        
        roeM(0,2) = 1.; 
        roeM(1,2) = velRho(0) + rho_a*nor_dim(0);
        roeM(2,2) = velRho(1) + rho_a*nor_dim(1);  
        roeM(3,2) = velRho(2) + rho_a*nor_dim(2);
        roeM(4,2) = rho_h + rho_a*vn_rho;

        // Two shear wave components combined into one (wave strength incorporated).
          
        du = vel_R(0) - vel_L(0);
        dv = vel_R(1) - vel_L(1);
        dw = vel_R(2) - vel_L(2);

        roeM(0,3) = 0.; 
        roeM(1,3) = du - dvn*nor_dim(0);
        roeM(2,3) = dv - dvn*nor_dim(1);
        roeM(3,3) = dw - dvn*nor_dim(2);
        roeM(4,3) = velRho(0)*du + velRho(1)*dv + velRho(2)*dw - vn_rho*dvn;

        // Dissipation Term: |An|(UR-UL) = R|Lambda|L*dU = sum_k of [ ws(k) * R(:,k) * L*dU(k) ]
        
        for(int j = 0; j < var_dim; j++)
            diss(j) = ws(0)*LdU(0)*roeM(j,0) + ws(1)*LdU(1)*roeM(j,1) 
             + ws(2)*LdU(2)*roeM(j,2) + ws(3)*LdU(3)*roeM(j,3);

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < var_dim; j++)
            {
                f_com((i*var_dim + j)*num_pts + p) += 
                    -0.5*nor_dim(i)*diss(j);
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



