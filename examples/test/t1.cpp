#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.031;
const double R_gas = 1;
const double   Pr  = 0.72;

//Run parameters
const char *mesh_file   = "per_5.mesh";
const int    order      = 1;
const double t_final    = 0.0200;
const double cfl        = 0.20;
const double dt_const   = 0.0100;
const int    vis_steps  = 50 ;
const int    ref_levels = 3;
const int    problem    = 0;

// Velocity coefficient
void init_function(const Vector &x, Vector &v);

// Characteristic boundary condition specification
void char_bnd_cnd(const Vector &x, Vector &v);
// Wall boundary condition specification
void wall_bnd_cnd(const Vector &x, Vector &v);
void wall_adi_bnd_cnd(const Vector &x, Vector &v);

double getUMax(int dim, const Vector &u);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getVisFlux(int dim, const Vector &u, const Vector &aux_grad, Vector &f);

void getAuxGrad(int dim, const SparseMatrix &K_x, const SparseMatrix &K_y, const SparseMatrix &M, 
        const Vector &u, const Vector &b_aux_x, const Vector &b_aux_y, Vector &aux_grad);
void getAuxVar(int dim, const Vector &u, Vector &aux_sol);

void getFields(const GridFunction &u_sol, Vector &rho, Vector &u1, Vector &u2, Vector &E);

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form is M du/dt = K u + b, where M and K are the mass
    and operator matrices, and b describes the face correction terms. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
   SparseMatrix &M, &K_inv_x, &K_inv_y;
   const Vector &b;
   DSmoother M_prec;
   CGSolver M_solver;

public:
   FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, const Vector &_b);

   void GetSize() ;

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

class CNS 
{
private:
    Mesh *mesh;
    Array<int> eleOrder;
    VarL2_FiniteElementCollection *vfec;
    FiniteElementSpace *fes, *fes_vec, *fes_op;
   
    BilinearForm *m, *k_inv_x, *k_inv_y;

    DataCollection *dc;
    ODESolver *ode_solver; 

    int dim;

    GridFunction u_sol, f_inv;   
    LinearForm *b;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, t, u_max;
    int ti;
public:
   CNS();

   void Step();
   void Update();

   ~CNS(); 
};





int main(int argc, char *argv[])
{
   int precision = 8;
   cout.precision(precision);

   CNS run;

   return 0;
}

CNS::CNS() 
{
   // Read the mesh from the given mesh file
   mesh = new Mesh(mesh_file, 1, 1);
           dim = mesh->Dimension();
   int var_dim = dim + 2;
   int aux_dim = dim + 1; //Auxiliary variables for the viscous terms

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }

   double kappa_min, kappa_max;
   mesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   int ne = mesh->GetNE();
   eleOrder.SetSize(ne);

   for (int i = 0; i < ne; i++) eleOrder[i] = order; // Initializing with uniform order

   //    Define the discontinuous DG finite element space 
   //    Var_L2 allows variable order polynomials 
   vfec = new VarL2_FiniteElementCollection(mesh, eleOrder);
   fes = new FiniteElementSpace(mesh, vfec, var_dim);

   cout << "Number of unknowns: " << fes->GetVSize() << endl;

   VectorFunctionCoefficient u0(var_dim, init_function);
   u_sol.SetSpace(fes);
   u_sol.ProjectCoefficient(u0);

   fes_vec = new FiniteElementSpace(mesh, vfec, dim*var_dim);
   f_inv.SetSpace(fes_vec);
   getInvFlux(dim, u_sol, f_inv);

   ///////////////////////////////////////////////////////////
   // Setup bilinear form for x derivative and the mass matrix
   Vector dir(dim);
   dir(0) = 1.0; dir(1) = 0.0;
   VectorConstantCoefficient x_dir(dir);
   dir(0) = 0.0; dir(1) = 1.0;
   VectorConstantCoefficient y_dir(dir);

   fes_op = new FiniteElementSpace(mesh, vfec);
   m = new BilinearForm(fes_op);
   m->AddDomainIntegrator(new MassIntegrator);
   k_inv_x = new BilinearForm(fes_op);
   k_inv_x->AddDomainIntegrator(new ConvectionIntegrator(x_dir, -1.0));
   k_inv_y = new BilinearForm(fes_op);
   k_inv_y->AddDomainIntegrator(new ConvectionIntegrator(y_dir, -1.0));

   m->Assemble();
   m->Finalize();
   int skip_zeros = 1;
   k_inv_x->Assemble(skip_zeros);
   k_inv_x->Finalize(skip_zeros);
   k_inv_y->Assemble(skip_zeros);
   k_inv_y->Finalize(skip_zeros);


   SparseMatrix &K_inv_x   = k_inv_x->SpMat();
   SparseMatrix &K_inv_y   = k_inv_y->SpMat();
   SparseMatrix &M         = m->SpMat();
   /////////////////////////////////////////////////////////////

   VectorGridFunctionCoefficient u_vec(&u_sol);
   VectorGridFunctionCoefficient f_vec(&f_inv);

   /////////////////////////////////////////////////////////////
   // Linear form
   b = new LinearForm(fes);
   b->AddFaceIntegrator(
      new DGEulerIntegrator(u_vec, f_vec, var_dim, -1.0));
   b->Assemble();
   ///////////////////////////////////////////////////////////

   dc = new VisItDataCollection("Euler", mesh);
   dc->SetPrecision(8);
 
   FiniteElementSpace fes_fields(mesh, vfec);
   GridFunction rho(&fes_fields);
   GridFunction u1(&fes_fields);
   GridFunction u2(&fes_fields);
   GridFunction E(&fes_fields);

   dc->RegisterField("rho", &rho);
   dc->RegisterField("u1", &u1);
   dc->RegisterField("u2", &u2);
   dc->RegisterField("E", &E);

   getFields(u_sol, rho, u1, u2, E);

   dc->SetCycle(0);
   dc->SetTime(0.0);
   dc->Save();

   FE_Evolution *adv = new FE_Evolution(M, K_inv_x, K_inv_y, *b);

//   ode_solver = new ForwardEulerSolver; 
   ode_solver = new RK3SSPSolver; 

   u_max = getUMax(dim, u_sol);
//   dt = cfl*((h_min/(2.0*order + 1))/u_max); 
   dt = dt_const; 

   t = 0.0; ti = 0;

   adv->SetTime(t);
   ode_solver->Init(*adv);

   bool done = false;
   for (ti = 0; !done; )
   {
      Step();

      if (ti == 1)
      {
          Update();      

          delete adv;   
          FE_Evolution *adv = new FE_Evolution(M, K_inv_x, K_inv_y, *b);
          adv->SetTime(t);
          ode_solver->Init(*adv);
      }

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
//          getFields(u_sol, rho, u1, u2, E);
//
//          dc->SetCycle(ti);
//          dc->SetTime(t);
//          dc->Save();
      }
   }
   // Print all nodes in the finite element space 
   FiniteElementSpace fes_nodes(mesh, vfec, dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1 = i, sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
//       cout << nodes(sub1) << '\t' << nodes(sub2)  << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << b[0][sub1] << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub4) << "\t" << f_inv(sub4) << endl;      
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << u_sol(sub1) << "\t" << u_out(sub1) << endl;      
   }

   delete adv;

}


CNS::~CNS()
{
    delete b;
    delete ode_solver;
    delete dc;
    delete mesh;
}

void CNS::Step()
{
      b->Assemble();

      double dt_real = min(dt, t_final - t);
      ode_solver->Step(u_sol, t, dt_real);
      ti++;

      cout << "time step: " << ti << ", dt: " << dt_real << ", time: " << t << ", max_speed " << u_max << endl;

      u_max = getUMax(dim, u_sol);
//      dt = cfl*((h_min/(2.0*order + 1))/u_max); 
      dt = dt_const; 

      getInvFlux(dim, u_sol, f_inv); // To update f_vec
}

void CNS::Update()
{
   int var_dim = dim + 2; 
   int ne = mesh->GetNE();
   Array<int> eleOrder_temp(ne);

   for (int i = 0; i < ne; i++) eleOrder_temp[i] = eleOrder[i]; 

   VarL2_FiniteElementCollection vfec_temp(mesh, eleOrder_temp);
   FiniteElementSpace fes_temp(mesh, &vfec_temp, var_dim);

   GridFunction u_temp(&fes_temp);
   u_temp.ProjectGridFunction(u_sol); // Create a temp variable to get the previous space solution

   for (int i = 0; i < ne; i++) eleOrder[i] = order + 1; 

   vfec->Update(eleOrder);
   fes->Update();
   fes_vec->Update();
   fes_op->Update();

   u_sol.Update();
   u_sol.GetValuesFrom(u_temp);   // Update solution to new space using temp variable

   f_inv.Update();
   getInvFlux(dim, u_sol, f_inv); // To update f_vec

   m->Update();
   m->Assemble();
   m->Finalize();

   int skip_zeros = 1;
   k_inv_x->Update();
   k_inv_x->Assemble(skip_zeros);
   k_inv_x->Finalize(skip_zeros);
   k_inv_y->Update();
   k_inv_y->Assemble(skip_zeros);
   k_inv_y->Finalize(skip_zeros);

   b->Update();
   b->Assemble();

}


// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(SparseMatrix &_M, SparseMatrix &_K_inv_x, SparseMatrix &_K_inv_y, const Vector &_b)
   : TimeDependentOperator(_b.Size()), M(_M), K_inv_x(_K_inv_x), K_inv_y(_K_inv_y), b(_b)
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
    int dim = x.Size()/K_inv_x.Size() - 2;
    int var_dim = dim + 2;

    y.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector y_temp;
    y_temp.SetSize(x.Size()); // Needed since by default ode_init will set it as size of K

    Vector f(dim*x.Size());
    getInvFlux(dim, x, f);

    int offset  = K_inv_x.Size();
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
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i], f_x_m);
    }
    y += y_temp;

    for(int i = var_dim + 0; i < 2*var_dim; i++)
    {
        f.GetSubVector(offsets[i], f_sol);
        K_inv_y.Mult(f_sol, f_x);
        M_solver.Mult(f_x, f_x_m);
        y_temp.SetSubVector(offsets[i - var_dim], f_x_m);
    }
    y += y_temp;

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
    
        for(int i = 0; i < dim; i++) vel[i] = u[1 + i]/rho;

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


// Get gradient of auxilliary variable 
void getAuxGrad(int dim, const SparseMatrix &K_x, const SparseMatrix &K_y, const SparseMatrix &M, 
        const Vector &u, const Vector &b_aux_x, const Vector &b_aux_y, Vector &aux_grad)
{
    CGSolver M_solver;

    M_solver.SetOperator(M);
    M_solver.iterative_mode = false;

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
        aux_x *= -1.0; // K_vis = -d/dx 
        b_aux_x.GetSubVector(offsets[i], b_sub);
        add(b_sub, aux_x, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[          i], aux_x);

        K_y.Mult(aux_var, aux_x);
        aux_x *= -1.0; // K_vis = -d/dx 
        b_aux_y.GetSubVector(offsets[i], b_sub);
        add(b_sub, aux_x, aux_x);
        M_solver.Mult(aux_x, aux_x);
        aux_grad.SetSubVector(offsets[aux_dim + i], aux_x);
    }
    for(int i = 0; i < offset; i++)
    {
//        cout << i << "\t" << aux_sol(2*offset + i) << "\t" << aux_grad(3*offset + i) << endl;    
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
    u.GetSubVector(offsets[0], rho   );
    u.GetSubVector(offsets[3],      E);

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
//        cout << i << "\t" << f(2*offset + i) << endl;
    }
}




//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p, T, M;
       double R, beta, omega, f, du, dv, dT;

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

//       cout << x(0) << "\t" << x(1) << "\t" << beta << "\t" << p << endl;
    
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
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

   if (dim == 2)
   {
       v(0) = 0.0;  // x velocity   
       v(1) = 0.0;  // y velocity 
       v(2) = 0.6;  // Temperature 
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
}


void char_bnd_cnd(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double rho, u1, u2, p;
       rho = 1;
       u1 = 3.10; u2 = 0.0;
       p   =172.2;
    
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }
}





void getFields(const GridFunction &u_sol, Vector &rho, Vector &u1, Vector &u2, Vector &E)
{

    int vDim    = u_sol.VectorDim();
    int dofs    = u_sol.Size()/vDim;

    for (int i = 0; i < dofs; i++)
    {
        rho[i] = u_sol[         i];        
        u1 [i] = u_sol[  dofs + i]/rho[i];        
        u2 [i] = u_sol[2*dofs + i]/rho[i];        
        E  [i] = u_sol[3*dofs + i];        
    }
}
