#include "mfem.hpp"
#include "cns.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

//Constants
const double gamm  = 1.4;
const double   mu  = 0.       ; 
const double R_gas = 1.       ; 
const double   Pr  = 0.72;

//Run parameters
const int    problem         =  1;

const char *mesh_file        =  "per_5.mesh";
const int    order           =  2;
const int    ref_levels      =  3;
const double t_final         = 30.00001 ;

//Time marching parameters
const bool   time_adapt      =  false;
const double cfl             =  0.4  ;
const double dt_const        =  0.25 ;
const int    ode_solver_type =  3; // 1. Forward Euler 2. TVD SSP 3 Stage

//Implicit Time marching parameters
const double rtol            = 2E-8 ; // Linear solver tolerance
const int    max_newt_it     = 25   ; // Maximum Newton iterations
const double newt_tol        = 4E-16; // Newton solver  tolerance 

//Restart parameters
const bool restart           =  false;
const int  restart_freq      = 20000; // Create restart file after every 1000 time steps
const int  restart_cycle     =108000; // File number used for restart


//Freestream conditions
const double M_inf           =    0.5 ; 
const double p_inf           =   100.; 
const double rho_inf         =    1.0; 


////////////////////////////////////////////////////////////////////////


// Velocity coefficient
void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);

void getEulerJacobian(const int dim, const Vector &u, DenseMatrix &J1, DenseMatrix &J2);

double getAbsMin(Vector &v);

void postProcess(const double gamm, const double R_gas, 
                 GridFunction &u_sol,
                 int cycle, double time);

class CNS 
{
private:
    int num_procs, myid;

    ParMesh *pmesh ;

    ParFiniteElementSpace  *fes;
    ParFiniteElementSpace  *fes_flux;
   
    ParGridFunction *u_sol, *f_inv;   
    ParGridFunction *k_s          ;   
    
    ParGridFunction *k_t, *y_t, *u_t; 

    HypreParMatrix *M;

    int dim;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, dt_real, t;
    int ti;

public:
   CNS();

   void newton_solve(const double a21, const double a22, const Vector &e_rhs0, Vector &e_rhs) ;

   void solve(ParGridFunction &u_sol, Vector &rhs) ;

   ~CNS(); 
};

// Get D.J where D is the DG derivative matrix and J is Euler Jacobian
class DGMassIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape;

   DenseMatrix mmat;

public:
   DGMassIntegrator()
      { }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};


void DGMassIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd      = el.GetDof();
   int dim     = el.GetDim();
   int var_dim = dim + 2;

   double w;

   shape.SetSize(nd);

   elmat.SetSize(nd*var_dim);
   mmat .SetSize(nd        );

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   mmat  = 0.0;
   elmat = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcShape(ip, shape);

      Trans.SetIntPoint(&ip);

      w = Trans.Weight() * ip.weight;
      
      AddMult_a_VVt(w, shape, mmat);

   }

   // Full element matrix is just a blockwise stacking of the small matrix
   for (int i = 0; i < var_dim; i++)
       for (int j = 0; j < nd; j++)
           for (int k = 0; k < nd; k++)
           {
                elmat.Elem(i*nd + j, i*nd + k) = mmat.Elem(j, k);        
           }

}


// Get D.J where D is the DG derivative matrix and J is Euler Jacobian
class DeriJacobianIntegrator : public BilinearFormIntegrator
{
private:
   DenseMatrix dshape, adjJ, Q_ir;
   Vector shape, vec2, BdFidxT;
   VectorCoefficient &Q;

public:
   DeriJacobianIntegrator(VectorCoefficient &q)
      : Q(q) { }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};


void DeriJacobianIntegrator::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &elmat)
{
   int nd      = el.GetDof();
   int dim     = el.GetDim();
   int var_dim = dim + 2;

   DenseMatrix dshape, adjJ, Q_ir, small1_mat, small2_mat, d1mat, d2mat;
   Vector shape, vec2, BdFidxT;

   // Derivative variables
   small1_mat.SetSize(nd);
   small2_mat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);
   
   d1mat.SetSize(nd*var_dim);
   d2mat.SetSize(nd*var_dim);
  
   // Jacobian variables
   Vector u_ip;
   DenseMatrix J1, J2, j1, j2;

   j1.SetSize(nd*var_dim);
   j2.SetSize(nd*var_dim);

   elmat.SetSize(nd*var_dim);

   Vector vec1;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q_ir.SetSize(dim, nd);

   small1_mat = 0.0;
   small2_mat = 0.0;
   j1      = 0.0;
   j2      = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      // Get derivative
      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      
      // X-direction
      Q_ir = 0.0;

      Q_ir.SetRow(0, 1.0);

      Q_ir.GetColumnReference(i, vec1);
      vec1 *= ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, small1_mat);


      // Y-direction
      Q_ir = 0.0;

      Q_ir.SetRow(1, 1.0);

      Q_ir.GetColumnReference(i, vec1);
      vec1 *= ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, small2_mat);

      // Get Jacobian
      Q.Eval(u_ip, Trans, ip);
      getEulerJacobian(dim, u_ip, J1, J2);
   
      for (int j = 0; j < var_dim; j++)
      {
          for (int k = 0; k < var_dim; k++)
          {
              j1.Elem(i + j*nd, i + k*nd) = J1.Elem(j, k);
              j2.Elem(i + j*nd, i + k*nd) = J2.Elem(j, k);
          }
      }

   }

   d1mat     = 0.0;
   d2mat     = 0.0;
   // Full element matrix is just a blockwise stacking of the small matrix
   for (int i = 0; i < var_dim; i++)
       for (int j = 0; j < nd; j++)
           for (int k = 0; k < nd; k++)
           {
                d1mat.Elem(i*nd + j, i*nd + k) = small1_mat.Elem(j, k);        
                d2mat.Elem(i*nd + j, i*nd + k) = small2_mat.Elem(j, k); 
           }

   // Multiply derivative matrix with Euler Jacobian
   Mult(d1mat, j1, elmat);
   AddMult(d2mat, j2, elmat);

}


// Get P^T.W.J.P where P is the projection matrix and J is Euler Jacobian
class FaceJacobianIntegrator : public BilinearFormIntegrator
{
private:
   Vector shape1, shape2;
   VectorCoefficient &Q;

   GridFunction &uS;
   const FiniteElementSpace &fes;

   Vector nor;

   DenseMatrix j1, j2;

public:
   FaceJacobianIntegrator(VectorCoefficient &q, GridFunction &u_sol, const FiniteElementSpace &fes_)
      : Q(q), uS(u_sol), fes(fes_) { }
   virtual void AssembleFaceMatrix(const FiniteElement &, const FiniteElement &,
                                      FaceElementTransformations &,
                                      DenseMatrix &);
};


void FaceJacobianIntegrator::AssembleFaceMatrix(
        const FiniteElement &el1, const FiniteElement &el2,
        FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int nd1     = el1.GetDof();
   int nd2     = el2.GetDof();
   int dim     = el1.GetDim();
   int var_dim = dim + 2;

   Vector nor_dim;

   shape1.SetSize(nd1);
   shape2.SetSize(nd2);

   nor.SetSize(dim);
   nor_dim.SetSize(dim);
        
   double nor_l2 ;

   elmat.SetSize( (nd1 + nd2)*var_dim);
   
   const IntegrationRule *ir ;
   int order = 2*std::max(el1.GetOrder(), el2.GetOrder());
   ir = &IntRules.Get(Trans.FaceGeom, order);

   int numFacePts = ir->GetNPoints() ;

   DenseMatrix P_L(numFacePts*var_dim, nd1*var_dim  );
   DenseMatrix P_R(numFacePts*var_dim, nd2*var_dim  );

   DenseMatrix J_L(numFacePts*var_dim, numFacePts*var_dim);
   DenseMatrix J_R(numFacePts*var_dim, numFacePts*var_dim);
   
   DenseMatrix tempJ(numFacePts*var_dim, numFacePts*var_dim);

   DenseMatrix diss_L(numFacePts*var_dim, numFacePts*var_dim);
   DenseMatrix diss_R(numFacePts*var_dim, numFacePts*var_dim);

   DenseMatrix JP_L(numFacePts*var_dim, nd1*var_dim );
   DenseMatrix JP_R(numFacePts*var_dim, nd2*var_dim );
 
   DenseMatrix JD_L(numFacePts*var_dim, nd1*var_dim );
   DenseMatrix JD_R(numFacePts*var_dim, nd2*var_dim );
  
   DenseMatrix wts(numFacePts*var_dim, numFacePts*var_dim);

   DenseMatrix temp11(numFacePts*var_dim, nd1*var_dim );
   DenseMatrix temp12(numFacePts*var_dim, nd1*var_dim );
   DenseMatrix temp21(numFacePts*var_dim, nd2*var_dim );
   DenseMatrix temp22(numFacePts*var_dim, nd2*var_dim );

   DenseMatrix fu_L(nd1*var_dim, nd1*var_dim);
   DenseMatrix l1(nd1*var_dim, nd1*var_dim);
   DenseMatrix l2(nd1*var_dim, nd1*var_dim);
   DenseMatrix fu_R(nd2*var_dim, nd2*var_dim);
   DenseMatrix r1(nd2*var_dim, nd2*var_dim);
   DenseMatrix r2(nd2*var_dim, nd2*var_dim);

   diss_L = 0.;
   diss_R = 0.;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
        const IntegrationPoint &ip = ir->IntPoint(p);
        IntegrationPoint eip1, eip2;
        Trans.Loc1.Transform(ip, eip1);
        Trans.Loc2.Transform(ip, eip2);
   
        Trans.Face->SetIntPoint(&ip);
        Trans.Elem1->SetIntPoint(&eip1);
        Trans.Elem2->SetIntPoint(&eip2);
 
        el1.CalcShape(eip1, shape1);
        el2.CalcShape(eip2, shape2);

        int offset;
        for(int j=0; j < var_dim; j++)
        {
            offset = nd1;
        
            for(int k=0; k < offset; k++)
            {
                P_L(j*numFacePts + p, j*offset + k) = shape1[k];                  
            }
            
            offset = nd2;
        
            for(int k=0; k < offset; k++)
            {
                P_R(j*numFacePts + p, j*offset + k) = shape2[k];                  
            }

        }
        CalcOrtho(Trans.Face->Jacobian(), nor);
        nor_l2 = nor.Norml2();
        nor_dim.Set(1/nor_l2, nor);

        Vector u_l, u_r;      
        Q.Eval(u_l, *Trans.Elem1, eip1);
        Q.Eval(u_r, *Trans.Elem2, eip2);
   
        getEulerJacobian(dim, u_l, j1, j2);

        double u_max = 2.4;

        for(int j=0; j < var_dim; j++)
            for(int k=0; k < var_dim; k++)
            {
                J_L.Elem(p + j*numFacePts, p + k*numFacePts) = j1.Elem(j, k)*nor(0) + j2.Elem(j, k)*nor(1);
            }
 
        for(int j=0; j < var_dim; j++)
            {
                diss_L.Elem(p + j*numFacePts, p + j*numFacePts)+=   u_max*nor(0)*nor_dim(0) + u_max*nor(1)*nor_dim(1);
            }

        getEulerJacobian(dim, u_r, j1, j2);
        
        for(int j=0; j < var_dim; j++)
            for(int k=0; k < var_dim; k++)
            {
                J_R.Elem(p + j*numFacePts, p + k*numFacePts) = j1.Elem(j, k)*nor(0) + j2.Elem(j, k)*nor(1);
            }

        for(int j=0; j < var_dim; j++)
            {
                diss_R.Elem(p + j*numFacePts, p + j*numFacePts)+=    u_max*nor(0)*nor_dim(0) + u_max*nor(1)*nor_dim(1);
            }

        for(int j=0; j < var_dim; j++)
        {
            wts(j*numFacePts + p, j*numFacePts + p) = -0.5*ip.weight;
        }


   }// p loop 

   /*
    * Here we implement the Lax Friedrichs flux, we want to construct a matrix like this
    * | l1 l2 |
    * | r1 r2 |
    * so that l1.u1 + l2.u2 gives the correct rhs for all left hand cells and
    * r1.u1 + r2.u2 gives the correct rhs for all right hand cells
    * To preserve this left hand, right hand cells we needed to multipl nor_dim for
    * Rusanov dissipation since it is (f_L + f_R) -alpha(u_R - u_L)
    * and left and right cells depend on direction of normal.
    */

   Add( 1.0, J_L, -1.0, diss_L, tempJ);
   J_L = tempJ;
   Mult(J_L, P_L, JP_L);
   Mult(wts, JP_L, temp11);
   Add(1.0, J_L,  2.0, diss_L, tempJ);
   J_L = tempJ;
   Mult(J_L, P_L, JP_L);
   Mult(wts, JP_L, temp12);
 
   Add(-1.0, J_R,   1.0, diss_R, tempJ);
   J_R = tempJ;
   Mult(J_R, P_R, JP_R);
   Mult(wts, JP_R, temp21);
   Add(1.0, J_R,  -2.0, diss_R, tempJ);
   J_R = tempJ;
   Mult(J_R, P_R, JP_R);
   Mult(wts, JP_R, temp22);

   P_L.Transpose();
   P_R.Transpose();

   elmat = 0.0;

   // Here we assume the common flux is 0.5(f_L + f_R)
   Mult(P_L, temp11, fu_L);
   Mult(P_R, temp12, fu_R);

   Mult(P_L, temp11, l1  );
   Mult(P_R, temp12, r1  );

   Mult(P_L, temp21, l2  );
   Mult(P_R, temp22, r2  );

   for (int j = 0; j < nd1*var_dim; j++)
    for (int k = 0; k < nd1*var_dim; k++)
       {
            elmat.Elem(j, k) =   l1(j, k);        
       }

   for (int j = 0; j < nd2*var_dim; j++)
    for (int k = 0; k < nd1*var_dim; k++)
       {
            elmat.Elem(nd1*var_dim + j, k) =   r1(j, k);        
       }

   Mult(P_L, temp21, fu_L);
   Mult(P_R, temp22, fu_R);

   for (int j = 0; j < nd1*var_dim; j++)
    for (int k = 0; k < nd2*var_dim; k++)
       {
            elmat.Elem(j, nd1*var_dim + k) =    l2(j, k);        
       }
   
   for (int j = 0; j < nd2*var_dim; j++)
    for (int k = 0; k < nd2*var_dim; k++)
       {
            elmat.Elem(nd1*var_dim + j, nd1*var_dim + k) =    r2(j, k);        
       }



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

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh->UniformRefinement();
   }
 
   DG_FECollection fec(order, dim);
   
   pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;

   double kappa_min, kappa_max;
   pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);

   fes = new ParFiniteElementSpace(pmesh, &fec, var_dim, Ordering::byVDIM);
   fes_flux = new ParFiniteElementSpace(pmesh, &fec, dim*var_dim, Ordering::byVDIM);

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   VectorFunctionCoefficient u0(var_dim, init_function);
   
   u_sol = new ParGridFunction(fes);
   f_inv = new ParGridFunction(fes_flux); 

   double r_t; int r_ti;
   if (restart == false)
       u_sol->ProjectCoefficient(u0);
   else
       doRestart(restart_cycle, *pmesh, *u_sol, r_t, r_ti);

   getInvFlux(dim, *u_sol, *f_inv);
 
   ParBilinearForm *m = new ParBilinearForm(fes);
   m->AddDomainIntegrator( new DGMassIntegrator );
   m->Assemble(1); 
   m->Finalize(1); 
   
   M = m  ->ParallelAssemble();

   HypreSmoother M_prec;
   CGSolver M_solver;

   M_prec.SetType(HypreSmoother::Jacobi); 
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(*M);
   
   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(10);
   M_solver.SetPrintLevel(0);

   delete m;

  
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
       postProcess(gamm, R_gas, *u_sol, ti, t);
   }

   Vector e_rhs0(u_sol->Size()); // Stage value for Newton iterations
   Vector e_rhs1(u_sol->Size()); // Stage value for Newton iterations
   Vector e_rhs (u_sol->Size()); // Stage value for Newton iterations
   Vector m_rhs (u_sol->Size()); // Stage value for Newton iterations

   // Initialize parameters
   
   k_s       = new ParGridFunction(fes);
   *k_s      = *u_sol;
   
   if (time_adapt == false)
   {
       dt = dt_const; 
   }

   bool done = false;
   for (ti = ti_in; !done; )
   {
       dt_real = min(dt, t_final - t);

//       double a11      = 1.   -     (1./std::sqrt(2.)); 
//       double a21      =-1.   +     (   std::sqrt(2.)); 
//       double a22      = 1.   -     (1./std::sqrt(2.)); 
 
       double a11      = 0.5  +     (.5/std::sqrt(3.)); 
       double a21      =      -     (1./std::sqrt(3.)); 
       double a22      = 0.5  +     (.5/std::sqrt(3.)); 
      
       double b1       = 0.5; 
       double b2       = 0.5; 

       *k_s   = *u_sol;
       e_rhs0 = 0.0;

       newton_solve(  0,   1, e_rhs0, e_rhs1);       
       e_rhs0 = e_rhs1;
   
       newton_solve(a21, a22, e_rhs0, e_rhs1);       

       {
           e_rhs0 *= -b1*dt_real;
           e_rhs1 *= -b2*dt_real;

           add(e_rhs0, e_rhs1, e_rhs);
           
           M_solver.Mult(e_rhs, m_rhs);
       }

       add(*u_sol, m_rhs, *k_s);
       *u_sol = *k_s;

       cout << u_sol->Min() << "\t" << u_sol->Max() << "\t" << u_sol->Sum() << endl;

       t += dt_real;
       ti++;

       if (myid == 0)
       {
            cout << setprecision(6) << "time step: " << ti << ", dt: " << dt_real << ", time: " << 
            t <<  endl;

       }

       postProcess(gamm, R_gas, *u_sol, ti, t);

       done = (t >= t_final - 1e-8*dt);

   }

   delete M;


}

/*
 * Function for explicit stepping
 */
void CNS::solve(ParGridFunction &u_sol, Vector &rhs) 
{
    VectorGridFunctionCoefficient u_vec(&u_sol);

    rhs.SetSize(u_sol.Size()); // Stage value for Newton iterations

    {
        ParBilinearForm *KJx = new ParBilinearForm(fes);
        KJx->AddDomainIntegrator( new DeriJacobianIntegrator(u_vec) );
        KJx->AddInteriorFaceIntegrator( new FaceJacobianIntegrator(u_vec, u_sol, *fes) );
        KJx->Assemble(1); 
        KJx->Finalize(1); 
    
        HypreParMatrix *A = KJx->ParallelAssemble();
        
        delete KJx;
     
        A->Mult(u_sol, rhs);
        rhs *= -1;
    
        delete A;

    }

}



void CNS::newton_solve(const double a21, const double a22, const Vector &e_rhs0, Vector &e_rhs) 
{
    VectorGridFunctionCoefficient u_vec(k_s);

    Vector temp1(u_sol->Size()); // Stage value for Newton iterations
    Vector temp2(u_sol->Size()); // Stage value for Newton iterations
    Vector   rhs(u_sol->Size()); // Stage value for Newton iterations
    Vector    du(u_sol->Size()); // Stage value for Newton iterations

    int it;
    double min_du = 1E15;
    // Newton iteration
    for (it = 0; it < max_newt_it; it++)
    {
        ParBilinearForm *KJx = new ParBilinearForm(fes);
        KJx->AddDomainIntegrator( new DeriJacobianIntegrator(u_vec) );
        KJx->AddInteriorFaceIntegrator( new FaceJacobianIntegrator(u_vec, *u_sol, *fes) );
        KJx->Assemble(1); 
        KJx->Finalize(1); 
    
        HypreParMatrix *A = KJx->ParallelAssemble();
        
        delete KJx;
     
        (*M) *= 1./(a22*dt);   // M = M/dt
    
        subtract(*k_s, *u_sol, temp1);
        M->Mult(temp1, temp2);
    
        A->Mult(*k_s, temp1);
        e_rhs = temp1;
    
        add(temp2, temp1, rhs); // (M/dt)*(u_n - u_0) + F(u) 
        add(rhs, (a21/a22), e_rhs0, temp1); 
        rhs = temp1;
    
        rhs *= -1;
    
        HypreParMatrix *C = Add(1., *M,  1, *A);
    
        HypreSmoother jac(*C, 0); // Jacobi smoother

        GMRESSolver gmres(C->GetComm());
        gmres.SetKDim(50);
        IterativeSolver &solver = gmres;
        solver.SetRelTol(rtol);
        solver.SetMaxIter(500);
        solver.SetPrintLevel(0);
        solver.SetOperator(*C);
        solver.SetPreconditioner(jac);

        solver.Mult(rhs, du);
    
        add(*k_s, du, temp1);
    
        *k_s = temp1;
        
        (*M) *= a22*dt;   // Reset M 
    
        delete A;
        C->~HypreParMatrix(); // This seems to be the only way to delete it

        min_du = std::min(min_du, getAbsMin(du));
        if (min_du < newt_tol)
            break;

    }
    cout << "Newton iterations: " << it << " Min of du: "<< getAbsMin(du) << endl; 

}



CNS::~CNS()
{
    delete pmesh;
    delete u_sol, f_inv;
    delete fes;

    MPI_Finalize();
}

double getAbsMin(Vector &v)
{
    int len = v.Size();

    double min = std::abs(v[0]);

    for(int i = 1; i < len; i++)
        min = std::min(std::abs(v[i]), min);

    return min;

}




// Inviscid flux 
void getInvFlux(int dim, const Vector &u, Vector &f)
{
    int var_dim = dim + 2;
    int offset  = u.Size()/var_dim;

    Array<int> offsets[var_dim];
    for(int i = 0; i < var_dim; i++)
    {
        offsets[i].SetSize(offset);
    }

    for(int j = 0; j < var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets[j][i] = var_dim*i + j ; // Redo offsets according to Ordering::ByVDIM
        }
    }
    Vector rho, rho_e;
    u.GetSubVector(offsets[0],           rho   );
    u.GetSubVector(offsets[var_dim - 1],  rho_e);
    
    Vector rho_vel[dim];
    for(int i = 0; i < dim; i++) u.GetSubVector(offsets[1 + i], rho_vel[i]);

    Vector temp_f[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
        temp_f[i].SetSize(offset);
    for(int i = 0; i < offset; i++)
    {
        double vel[dim];        
        for(int j = 0; j < dim; j++) vel[j]   = rho_vel[j](i)/rho(i);

        double vel_sq = 0.0;
        for(int j = 0; j < dim; j++) vel_sq += pow(vel[j], 2);

        double pres    = (rho_e(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);

        for(int j = 0; j < dim; j++)
        {
            temp_f[j*var_dim][i]       = rho_vel[j][i]; //rho*u
    
            for (int k = 0; k < dim ; k++)
            {
                temp_f[j*var_dim + 1 + k][i]     = rho_vel[j](i)*vel[k]; //rho*u*u + p    
            }
            temp_f[j*var_dim + 1 + j][i]        += pres; 
    
            temp_f[j*var_dim + var_dim - 1][i]   = (rho_e(i) + pres)*vel[j] ;//(E+p)*u
        }

    }

    Array<int> offsets_f[dim*var_dim];
    for(int i = 0; i < dim*var_dim; i++)
    {
        offsets_f[i].SetSize(offset);
    }

    for(int j = 0; j < dim*var_dim; j++)
    {
        for(int i = 0; i < offset; i++)
        {
            offsets_f[j][i] = dim*var_dim*i + j ; // Redo offsets according to Ordering::ByVDIM
        }
    }

    for (int i = 0; i < dim*var_dim; i++)    
    {
        f.SetSubVector(offsets_f[i], temp_f[i]  );
    }


}





//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   if (dim == 2)
   {
       double u1, u2, rho, p;
       if (problem == 0) // Smooth periodic density. Exact solution to Euler 
       {
           rho =  1.0 + 0.2*sin(M_PI*(x(0) + x(1)));
           p   =  1.0; 
           u1  =  1.0;
           u2  =  1.0;
       }
       else if (problem == 1)
       {
           double beta, r, omega1, omega2; 
           
           beta = 5;
           
           r      = std::sqrt( (std::pow(x(0), 2) + std::pow(x(1), 2)) );
           omega1 = std::exp(1 - r*r);
           omega2 = std::exp( 0.5*(1 - r*r) );
        
           rho    = std::pow( ( 1 - std::pow(beta, 2)*(gamm - 1)*omega1/(8.*gamm*pow(M_PI, 2)) ), (1./(gamm - 1)) );
           u1     = M_inf - beta*x(1)*omega2/(2.*M_PI);
           u2     =         beta*x(0)*omega2/(2.*M_PI);
        
           p      = std::pow(rho, gamm);
       }
        
       double v_sq = pow(u1, 2) + pow(u2, 2);
    
       v(0) = rho;                     //rho
       v(1) = rho * u1;                //rho * u
       v(2) = rho * u2;                //rho * v
       v(3) = p/(gamm - 1) + 0.5*rho*v_sq;
   }

}




/*
 * Get the Jacobian of the Euler equations
 */
void getEulerJacobian(const int dim, const Vector &u, DenseMatrix &J1, DenseMatrix &J2)
{
    int var_dim = dim + 2;

    double vel[dim];

    double rho  = u[0];

    double v_sq = 0;
    for(int i = 0; i < dim; i++)
    {
        vel[i] = u[1 + i]/rho;

        v_sq   = v_sq + vel[i]*vel[i];
    }

    double rho_e = u[dim + 1];
        
    double pres  = (rho_e - 0.5*rho*v_sq)*(gamm - 1);
    double H     = (rho_e + pres)/rho;

    J1.SetSize(var_dim);
    J2.SetSize(var_dim);

    J1 = 0.0; J2 = 0.0;

    if (dim == 2)
    {
        // X
        // Row 1
        J1.Elem(0, 1) = 1.0;
        
        // Row 2
        J1.Elem(1, 0) = -vel[0]*vel[0] + 0.5*(gamm - 1)*v_sq;
        J1.Elem(1, 1) =  (3 - gamm)*vel[0];
        J1.Elem(1, 2) = -(gamm - 1)*vel[1];
        J1.Elem(1, 3) =  gamm - 1;
 
        // Row 3
        J1.Elem(2, 0) = -vel[0]*vel[1]; 
        J1.Elem(2, 1) =  vel[1]; 
        J1.Elem(2, 2) =  vel[0]; 
        J1.Elem(2, 3) =  0; 
 
        // Row 4
        J1.Elem(3, 0) = (0.5*(gamm - 1)*v_sq - H)*vel[0]; 
        J1.Elem(3, 1) =  H - (gamm - 1)*vel[0]*vel[0]; 
        J1.Elem(3, 2) = -(gamm - 1)*vel[0]*vel[1] ; 
        J1.Elem(3, 3) =  gamm * vel[0]; 

        // Y
        // Row 1
        J2.Elem(0, 2) = 1.0;
        
        // Row 2
        J2.Elem(1, 0) = -vel[0]*vel[1]; 
        J2.Elem(1, 1) =  vel[1]; 
        J2.Elem(1, 2) =  vel[0]; 
        J2.Elem(1, 3) =  0.0;
 
        // Row 3
        J2.Elem(2, 0) = -vel[1]*vel[1] + 0.5*(gamm - 1)*v_sq;
        J2.Elem(2, 1) = -(gamm - 1)*vel[0]; 
        J2.Elem(2, 2) =  (3 - gamm)*vel[1]; 
        J2.Elem(2, 3) =  (gamm - 1); 
 
        // Row 4
        J2.Elem(3, 0) =  (0.5*(gamm - 1)*v_sq - H)*vel[1]; 
        J2.Elem(3, 1) = -(gamm - 1)*vel[0]*vel[1] ; 
        J2.Elem(3, 2) =  H - (gamm - 1)*vel[1]*vel[1]; 
        J2.Elem(3, 3) =  gamm * vel[1]; 
     
    }

//    Vector f(var_dim);
//    J.Mult(u, f);
//
//    Vector f_test(var_dim);
//    getInvFlux(dim, u, f_test);
//
//    f.Print();
//    f_test.Print();

}


void postProcess(const double gamm, const double R_gas, 
                 GridFunction &u_sol,
                 int cycle, double time)
{
   FiniteElementSpace &fes = *(u_sol.FESpace());
   Mesh &mesh              = *(fes.GetMesh());

   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   DG_FECollection fec(order , dim);
   FiniteElementSpace fes_field(&mesh, &fec);

   VisItDataCollection dc("CNS", &mesh);
   dc.SetPrecision(8);
 
   GridFunction rho(&fes_field);
   GridFunction p(&fes_field);
   GridFunction u(&fes_field);
   GridFunction v(&fes_field);

   dc.RegisterField("rho", &rho);
   dc.RegisterField("p", &p);
   dc.RegisterField("u", &u);
   dc.RegisterField("v", &v);

   int offset  = u_sol.Size()/var_dim;

   Array<int> offsets[var_dim];
   for(int i = 0; i < var_dim; i++)
   {
       offsets[i].SetSize(offset);
   }

   for(int j = 0; j < var_dim; j++)
   {
       for(int i = 0; i < offset; i++)
       {
           offsets[j][i] = var_dim*i + j ; // Redo offsets according to Ordering::ByVDIM
       }
   }
   Vector rho_e;
   u_sol.GetSubVector(offsets[0],           rho   );
   u_sol.GetSubVector(offsets[var_dim - 1],  rho_e);
   
   Vector rho_vel[dim];
   for(int i = 0; i < dim; i++) u_sol.GetSubVector(offsets[1 + i], rho_vel[i]);

   for(int i = 0; i < offset; i++)
   {
       double vel[dim];        
       u[i] =  rho_vel[0](i)/rho(i);
       v[i] =  rho_vel[1](i)/rho(i);

       double vel_sq = 0.0;
       for(int j = 0; j < dim; j++) vel_sq += pow(vel[j], 2);

       p[i]    = (rho_e(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);

   }

   dc.SetCycle(cycle);
   dc.SetTime(time);
   dc.Save();
  
}




