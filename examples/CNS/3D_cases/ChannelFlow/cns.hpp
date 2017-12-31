#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Function definitions

// Restart Functions
void writeRestart(Mesh &mesh, GridFunction &u_sol, int cycle, double time);
void doRestart(const int cycle, ParMesh &pmesh, ParGridFunction &u_sol,
        double &time, int &time_step);

// Visualization functions
void postProcess(Mesh &mesh, int order, const double gamm, const double R_gas, 
        GridFunction &u_sol, GridFunction &aux_grad,
        int cycle, double time);
void getMoreFields(const double gamm, const double R_gas, 
                const GridFunction &u_sol, const Vector &aux_grad, Vector &rho, Vector &M,
                Vector &p, Vector &T, Vector &E, Vector &u, Vector &v, Vector &w,
                Vector &vort, Vector &q);

// Analysis functions
void ComputeGlobPeriodicMean(MPI_Comm &comm, const vector< vector<int> > &ids, const vector<double> &loc_u_mean, 
        vector<double> &glob_u_mean);
void ComputePeriodicMean(int dim, const GridFunction &uD, const vector< vector<int> > &ids, vector<double> &u_mean);
void GetPeriodicIds(const FiniteElementSpace &fes, const vector<double> &y_unique, vector< vector<int> > &ids);
void GetUniqueY(const FiniteElementSpace &fes, vector<double>  &y_uni);


// Functions
void doRestart(const int cycle, ParMesh &pmesh, ParGridFunction &u_sol,
        double &time, int &time_step)
{
   MPI_Comm comm = pmesh.GetComm();
           
   int dim = pmesh.Dimension();

   VisItDataCollection dc("Restart", &pmesh);

   dc.Load(cycle);

   time      = dc.GetTime();
   time_step = dc.GetTimeStep();

   GridFunction *u_temp = dc.GetField("u_cns");

   for(int i = 0; i < u_temp->Size(); i++) 
       u_sol[i] = (*u_temp)[i];

}



void writeRestart(Mesh &mesh, GridFunction &u_sol, int cycle, double time)
{
   int dim     = mesh.Dimension();
   int var_dim = dim + 2;

   VisItDataCollection dc("Restart", &mesh);
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

   DG_FECollection fec(order + 1 , dim);
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


void ComputeGlobPeriodicMean(MPI_Comm &comm, const vector< vector<int> > &ids, const vector<double> &loc_u_mean, 
        vector<double> &glob_u_mean)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    vector<double> glob_u;

    int vert_nodes = ids.size();
    for(int i = 0; i < vert_nodes; i++)
    {
        int loc_row_size = ids.at(i).size();            
        double loc_row_u = loc_u_mean.at(i)*loc_row_size;

        int glob_row_size ;            
        double glob_row_u ;

        MPI_Allreduce(&loc_row_u,    &glob_row_u,    1, MPI_DOUBLE, MPI_SUM, comm); // Get global u_max across processors
        MPI_Allreduce(&loc_row_size, &glob_row_size, 1, MPI_INT,    MPI_SUM, comm); // Get global u_max across processors

        glob_u.push_back(glob_row_u/double(glob_row_size));

    }
    glob_u_mean = glob_u;
}


void ComputePeriodicMean(int dim, const GridFunction &uD, const vector< vector<int> > &ids, vector<double> &u_mean)
{
   vector<double> u_m;

   int var_dim = dim + 2;
   int offset  = uD.Size()/var_dim;

   int vert_nodes = ids.size();
   for(int i = 0; i < vert_nodes; i++)
   {
       vector<int> row_ids = ids.at(i);          
       int row_nodes = row_ids.size();
       double u = 0.0;
       for(int j = 0; j < row_nodes; j++)
       {
           int sub = row_ids.at(j);       
           u      += uD[offset + sub];
       }

       if (row_nodes > 0)
           u = u/double(row_nodes);
       else
           u = 0;

       u_m.push_back(u);
   }

   u_mean = u_m;

}



void GetPeriodicIds(const FiniteElementSpace &fes, const vector<double> &y_unique, vector< vector<int> > &ids)
{
   double eps = 1E-11;

   Mesh *mesh = fes.GetMesh();
   int   dim  = mesh->Dimension();
   int meshNE = mesh->GetNE();

   FiniteElementSpace fes_nodes(mesh, fes.FEColl(), dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   int nodeSize        = nodes.Size()/dim ;

   std::vector<double> x, y;

   for(int i = 0; i < nodeSize; i++)
   {
       x.push_back(nodes(i));
       y.push_back(nodes(nodeSize + i));
   }

   vector< vector<int> > l_ids;

   // Now we create a 2D vector which has the node numbers of each unique y
   for(int j = 0; j < y_unique.size(); j++)
   {
       vector<int>  row_ids;
       for(int i = 0; i < nodeSize; i++)
       {
           if (abs(y.at(i) - y_unique.at(j)) < eps)
               row_ids.push_back(i);
       }
       l_ids.push_back(row_ids);
   }
   ids = l_ids;
}



/*
 * We'll assume a structured grid in terms of shape for now to make it easier
 * This is for the Turbulent Channel flow where we average over
 * the streamwise and spanwise directions
 * The algorithm is sensitive to Tolerance levels
 * We simply get all unique y values and then get their subscripts 
 */
void GetUniqueY(const FiniteElementSpace &fes, vector<double> &y_uni)
{
   double eps = 1E-11;

   Mesh *mesh = fes.GetMesh();
   int   dim  = mesh->Dimension();
   int meshNE = mesh->GetNE();

   FiniteElementSpace fes_nodes(mesh, fes.FEColl(), dim);
   GridFunction nodes(&fes_nodes);
   mesh->GetNodes(nodes);

   int nodeSize        = nodes.Size()/dim ;

   std::vector<double> x, y, y_orig;

   for(int i = 0; i < nodeSize; i++)
   {
       x.push_back(nodes(i));
       y.push_back(nodes(nodeSize + i));
       y_orig.push_back(nodes(nodeSize + i));
   }

   double xMin = *(std::min_element(x.begin(), x.end()));
   std::sort(y.begin(), y.end(), std::less<double>());
   
   std::vector<double> y_unique;
   y_unique.push_back(y[0]);

   int yCoun = 0; //Count the number of co-ordinates in y direction on which we'll average
   for(int i = 0; i < nodeSize; i++)
   {
       double last_y = y_unique.back(); // Last unique y added
       if (abs(y[i] - last_y) > eps)
           y_unique.push_back(y[i]);

       if (abs(x[i] - xMin) < eps)
           yCoun++;
   }


   MFEM_ASSERT(y_unique.size() == yCoun, "Tolerance levels not enough for unique y") ;

   y_uni = y_unique;
}

