// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#include "mfem.hpp"
#include "periodic.hpp"

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
//    const char *mesh_file        =  "channel.msh";
//    const char *mesh_file        =  "channel_small.msh";
    const char *mesh_file        =  "channel_395.msh";
    Mesh *mesh                   =  new Mesh(mesh_file, 1, 1);

    int dim  = mesh->Dimension();
    int sdim = mesh->SpaceDimension();

    vector<Vector> trans_vecs(2);
    for(int i = 0; i < 2; i++)
    {
        trans_vecs[i].SetSize(sdim);    
    }
    trans_vecs[0][0] = 2*M_PI; trans_vecs[0][1] = 0.0; trans_vecs[0][2] = 0.0;
    trans_vecs[1][0] = 0.0;    trans_vecs[1][1] = 0.0; trans_vecs[1][2] = M_PI; 


    Mesh *nmesh                  = MakePeriodicMesh(mesh, trans_vecs);

    const char *out_file         = "channel_395.mesh";
    ofstream out_mesh;
    out_mesh.precision(16);
    out_mesh.open(out_file);
    nmesh->Print(out_mesh);
    out_mesh.close();

    delete mesh, nmesh;

    return 0;
}

