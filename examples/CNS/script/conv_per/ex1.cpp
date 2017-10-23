//                                MFEM Example 9
//              For converting GMSH to MFEM mesh 

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;



int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   problem = 0;
//   const char *mesh_file = "../data/periodic-square.mesh";
//   const char *mesh_file = "per_sq_mfem.mesh";
//   const char *mesh_file = "periodic-cube.mesh";
   const char *mesh_file = "per_cube.msh";
//   const char *mesh_file = "cu_mfem.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = 0.01;
   bool visualization = true;
   bool visit = true;
   bool binary = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   ofstream out_mesh;
   out_mesh.open("test_mfem.mesh");
   mesh->Print(out_mesh);
   mesh->SetCurvature(1, 1);
   mesh->Print(out_mesh);
   out_mesh.close();

   delete mesh;

   return 0;
}

