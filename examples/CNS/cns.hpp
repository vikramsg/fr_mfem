#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

void writeRestart(Mesh &mesh, GridFunction &u_sol, int cycle, double time);
void doRestart(const int cycle, ParMesh &pmesh, ParGridFunction &u_sol,
        double &time, int &time_step);


void postProcess(Mesh &mesh, int order, const double gamm, const double R_gas, 
        GridFunction &u_sol, GridFunction &aux_grad,
        int cycle, double time);
void getMoreFields(const double gamm, const double R_gas, 
                const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &M,
                Vector &p, Vector &T, Vector &E, Vector &u, Vector &v, Vector &w,
                Vector &vort, Vector &q);



void doRestart(const int cycle, ParMesh &pmesh, ParGridFunction &u_sol,
        double &time, int &time_step)
{
   MPI_Comm comm = pmesh.GetComm();
           
   int dim = pmesh.Dimension();

   VisItDataCollection dc("CNS_restart", &pmesh);

   dc.Load(cycle);

   time      = dc.GetTime();
   time_step = dc.GetTimeStep();

   u_sol     = *(dc.GetParField("u_cns"));  
}



void writeRestart(Mesh &mesh, GridFunction &u_sol, int cycle, double time)
{
   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   VisItDataCollection dc("CNS_restart", &mesh);
   dc.SetPrecision(16);
 
   dc.RegisterField("u_cns", &u_sol);

   dc.SetCycle(cycle);
   dc.SetTime(time);
   dc.Save();
  
}



void postProcess(Mesh &mesh, int order, const double gamm, const double R_gas, 
        GridFunction &u_sol, GridFunction &aux_grad,
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

   fes_post.Update();
   u_post.Update();
   aux_grad_post.Update();

   VisItDataCollection dc("CNS", &mesh);
   dc.SetPrecision(8);
 
   FiniteElementSpace fes_fields(&mesh, &fec);
   GridFunction rho(&fes_fields);
   GridFunction M(&fes_fields);
   GridFunction p(&fes_fields);
   GridFunction T(&fes_fields);
   GridFunction E(&fes_fields);
   GridFunction u(&fes_fields);
   GridFunction v(&fes_fields);
   GridFunction w(&fes_fields);
   GridFunction q(&fes_fields);
   GridFunction vort(&fes_fields);

   dc.RegisterField("rho", &rho);
   dc.RegisterField("M", &M);
   dc.RegisterField("p", &p);
   dc.RegisterField("T", &T);
   dc.RegisterField("E", &E);
   dc.RegisterField("u", &u);
   dc.RegisterField("v", &v);
   dc.RegisterField("w", &w);
   dc.RegisterField("q", &q);
   dc.RegisterField("vort", &vort);

//   getFields(u_post, aux_grad_post, rho, M, p, vort, q);
   getMoreFields(gamm, R_gas, u_post, aux_grad_post, rho, M, p, T, E, u, v, w, vort, q);

   dc.SetCycle(cycle);
   dc.SetTime(time);
   dc.Save();
  
}



void getMoreFields(const double gamm, const double R_gas,
                const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &M,
                Vector &p, Vector &T, Vector &E, Vector &u, Vector &v, Vector &w,
                Vector &vort, Vector &q)
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
        u[i] = vel[0];
        v[i] = vel[1];
        if (dim == 3)
            w[i] = vel[2];

        E[i]  = u_sol[(vDim - 1)*dofs + i];        

        double v_sq = 0.0;    
        for (int j = 0; j < dim; j++)
        {
            v_sq += pow(vel[j], 2); 
        }

        p[i]     = (E[i] - 0.5*rho[i]*v_sq)*(gamm - 1);
        T[i]     =  p[i]/(R_gas*rho[i]);

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

