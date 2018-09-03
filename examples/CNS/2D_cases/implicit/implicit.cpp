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
const char *mesh_file        =  "per_5.mesh";
const int    order           =  2;
const int    ref_levels      =  0;


//Freestream conditions
const double M_inf           =    0.5 ; 
const double p_inf           =   100.; 
const double rho_inf         =    1.0; 


////////////////////////////////////////////////////////////////////////


// Velocity coefficient
void init_function(const Vector &x, Vector &v);

void getInvFlux(int dim, const Vector &u, Vector &f);
void getEulerJacobian(const int dim, const Vector &u, DenseMatrix &J);


class CNS 
{
private:
    int num_procs, myid;

    ParMesh *pmesh ;

    ParFiniteElementSpace  *fes;
   
    ParGridFunction *u_sol, *f_inv;   

    int dim;

    double h_min, h_max;  // Minimum, maximum element size
    double dt, dt_real, t;
    int ti;

public:
   CNS();

   ~CNS(); 
};


class ElemEulerJacobian : public BilinearFormIntegrator
{
private:
   DenseMatrix Q_ir;
   Vector shape;
   VectorCoefficient &Q;
   double alpha;

public:
   ElemEulerJacobian(VectorCoefficient &q)
      : Q(q) { }
   virtual void AssembleElementMatrix(const FiniteElement &,
                                      ElementTransformation &,
                                      DenseMatrix &);
};



void ElemEulerJacobian::AssembleElementMatrix(
   const FiniteElement &el, ElementTransformation &Trans, DenseMatrix &jmat)
{
   int nd  = el.GetDof();
   int dim = el.GetDim();

   int var_dim = dim + 2;

   Vector u_ip;
   DenseMatrix J;

   jmat.SetSize(nd*var_dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   jmat = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Q.Eval(u_ip, Trans, ip);
      getEulerJacobian(dim, u_ip, J);
   
      for (int j = 0; j < var_dim; j++)
      {
          for (int k = 0; k < var_dim; k++)
          {
              jmat.Elem(i + j*nd, i + k*nd) = J.Elem(j, k);
          }
      }

   }
//   elmat.Print();

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

   DenseMatrix dshape, adjJ, Q_ir, small_mat, dmat;
   Vector shape, vec2, BdFidxT;

   // Derivative variables
   small_mat.SetSize(nd);
   dshape.SetSize(nd,dim);
   adjJ.SetSize(dim);
   shape.SetSize(nd);
   vec2.SetSize(dim);
   BdFidxT.SetSize(nd);
   
   dmat.SetSize(nd*var_dim);
  
   // Jacobian variables
   Vector u_ip;
   DenseMatrix J, jmat;

   jmat.SetSize(nd*var_dim);

   elmat.SetSize(nd*var_dim);

   Vector vec1;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = Trans.OrderGrad(&el) + Trans.Order() + el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   Q_ir.SetSize(dim, nd);
   Q_ir = 0.0;

   // X-direction
   Q_ir.SetRow(0, 1.0);

   small_mat = 0.0;
   jmat      = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      el.CalcDShape(ip, dshape);
      el.CalcShape(ip, shape);

      // Get derivative
      Trans.SetIntPoint(&ip);
      CalcAdjugate(Trans.Jacobian(), adjJ);
      Q_ir.GetColumnReference(i, vec1);
      vec1 *= ip.weight;

      adjJ.Mult(vec1, vec2);
      dshape.Mult(vec2, BdFidxT);

      AddMultVWt(shape, BdFidxT, small_mat);
     
      // Get Jacobian
      Q.Eval(u_ip, Trans, ip);
      getEulerJacobian(dim, u_ip, J);
   
      for (int j = 0; j < var_dim; j++)
      {
          for (int k = 0; k < var_dim; k++)
          {
              jmat.Elem(i + j*nd, i + k*nd) = J.Elem(j, k);
          }
      }

   }

   dmat     = 0.0;
   // Full element matrix is just a blockwise stacking of the small matrix
   for (int i = 0; i < var_dim; i++)
       for (int j = 0; j < nd; j++)
           for (int k = 0; k < nd; k++)
           {
                dmat.Elem(i*nd + j, i*nd + k) = small_mat.Elem(j, k);        
           }

   // Multiply derivative matrix with Euler Jacobian
   Mult(dmat, jmat, elmat);

//   jmat.Print();
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

   HYPRE_Int global_vSize = fes->GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of unknowns: " << global_vSize << endl;
   }

   VectorFunctionCoefficient u0(var_dim, init_function);
   
   u_sol = new ParGridFunction(fes);
   f_inv = new ParGridFunction(fes); // We take only the x direction flux

   u_sol->ProjectCoefficient(u0);
   getInvFlux(dim, *u_sol, *f_inv);

   VectorGridFunctionCoefficient u_vec(u_sol);

   ParBilinearForm *fJ = new ParBilinearForm(fes);
   fJ->AddDomainIntegrator( new ElemEulerJacobian(u_vec) );
   fJ->Assemble(); 
   fJ->Finalize(); 

   Vector xdir(dim), ydir(dim), zdir(dim); 
   xdir = 0.0; xdir(0) = 1.0;
   VectorConstantCoefficient x_dir(xdir);

   ParBilinearForm *KJx = new ParBilinearForm(fes);
   KJx->AddDomainIntegrator( new DeriJacobianIntegrator(u_vec) );
   KJx->Assemble(1); 
   KJx->Finalize(1); 

   Vector f_test(f_inv->Size());

//   fJ->SpMat().Print();
//   fJ->SpMat().Mult(*u_sol, f_test);

//   Kx->SpMat().Print();
//   KJx->SpMat().Mult(*u_sol, f_test);

   delete fJ;

   // Print all nodes in the finite element space 
   ParFiniteElementSpace fes_nodes(pmesh, &fec, dim);
   ParGridFunction nodes(&fes_nodes);
   pmesh->GetNodes(nodes);

   int offset = nodes.Size()/dim;

   for (int i = 0; i < nodes.Size()/dim; i++)
   {
       int offset = nodes.Size()/dim;
       int sub1  = i,  sub2 = offset + i, sub3 = 2*offset + i, sub4 = 3*offset + i;
       int vsub1 = var_dim*i + 0, vsub2 = var_dim*i + 1, vsub3 = var_dim*i + 2, vsub4 = var_dim*i + 3;
//       cout << nodes(sub1) << '\t' << nodes(sub2) << '\t' << (*u_sol)[vsub2] << "\t" 
//            << (*f_inv)[vsub3] << "\t" << f_test[vsub3] << endl;      
   }


}


CNS::~CNS()
{
    delete pmesh;
    delete u_sol, f_inv;
    delete fes;

    MPI_Finalize();
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

    Vector temp_f[var_dim];
    for(int i = 0; i < var_dim; i++)
        temp_f[i].SetSize(offset);
    for(int i = 0; i < offset; i++)
    {
        double vel[dim];        
        for(int j = 0; j < dim; j++) vel[j]   = rho_vel[j](i)/rho(i);

        double vel_sq = 0.0;
        for(int j = 0; j < dim; j++) vel_sq += pow(vel[j], 2);

        double pres    = (rho_e(i) - 0.5*rho(i)*vel_sq)*(gamm - 1);

        int j = 0;
        temp_f[0][i]       = rho_vel[j][i]; //rho*u

        for (int k = 0; k < dim ; k++)
        {
            temp_f[1 + k][i]     = rho_vel[j](i)*vel[k]; //rho*u*u + p    
        }
        temp_f[1 + j][i]        += pres; 

        temp_f[var_dim - 1][i]   = (rho_e(i) + pres)*vel[j] ;//(E+p)*u

    }

    for (int i = 0; i < var_dim; i++)    
        f.SetSubVector(offsets[i], temp_f[i]  );
    
//    for(int i = 0; i < offset; i++) 
//        cout << u[var_dim*i + 3] << "\t" << temp_f[0][i] << "\t" << f[var_dim*i + 3]<< endl; 

}





//  Initialize variables coefficient
void init_function(const Vector &x, Vector &v)
{
   //Space dimensions 
   int dim = x.Size();

   {
       double u1, u2, rho, p;
       double beta, r, omega1, omega2; 
       
       beta = 5;
       
       r      = std::sqrt( (std::pow(x(0), 2) + std::pow(x(1), 2)) );
       omega1 = std::exp(1 - r*r);
       omega2 = std::exp( 0.5*(1 - r*r) );
    
       rho    = std::pow( ( 1 - std::pow(beta, 2)*(gamm - 1)*omega1/(8.*gamm*pow(M_PI, 2)) ), (1./(gamm - 1)) );
       u1     = M_inf - beta*x(1)*omega2/(2.*M_PI);
       u2     =         beta*x(0)*omega2/(2.*M_PI);
    
       p      = std::pow(rho, gamm);
    
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
void getEulerJacobian(const int dim, const Vector &u, DenseMatrix &J)
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

    J.SetSize(var_dim);

    J = 0.0;

    if (dim == 2)
    {
        // Row 1
        J.Elem(0, 1) = 1.0;
        
        // Row 2
        J.Elem(1, 0) = -vel[0]*vel[0] + 0.5*(gamm - 1)*v_sq;
        J.Elem(1, 1) =  (3 - gamm)*vel[0];
        J.Elem(1, 2) = -(gamm - 1)*vel[1];
        J.Elem(1, 3) =  gamm - 1;
 
        // Row 3
        J.Elem(2, 0) = -vel[0]*vel[1]; 
        J.Elem(2, 1) =  vel[1]; 
        J.Elem(2, 2) =  vel[0]; 
        J.Elem(2, 3) =  0; 
 
        // Row 4
        J.Elem(3, 0) = (0.5*(gamm - 1)*v_sq - H)*vel[0]; 
        J.Elem(3, 1) =  H - (gamm - 1)*vel[0]*vel[0]; 
        J.Elem(3, 2) = -(gamm - 1)*vel[0]*vel[1] ; 
        J.Elem(3, 3) =  gamm * vel[0]; 
      
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





















