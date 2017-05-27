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

#include "../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "fem.hpp"

namespace mfem
{

void ParLinearForm::Update(ParFiniteElementSpace *pf)
{
   if (pf) { pfes = pf; }

   LinearForm::Update(pfes);
}

void ParLinearForm::Update(ParFiniteElementSpace *pf, Vector &v, int v_offset)
{
   pfes = pf;
   LinearForm::Update(pf,v,v_offset);
}

void ParLinearForm::ParallelAssemble(Vector &tv)
{
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, tv);
}

HypreParVector *ParLinearForm::ParallelAssemble()
{
   HypreParVector *tv = pfes->NewTrueDofVector();
   pfes->Dof_TrueDof_Matrix()->MultTranspose(*this, *tv);
   return tv;
}

void ParLinearForm::AssembleSharedFaces()
{
   ParMesh *pmesh = pfes->GetParMesh();
   FaceElementTransformations *T;
   Array<int> vdofs1, vdofs2, vdofs_all;
   Vector elemvect;
   
   int nfaces = pmesh->GetNSharedFaces();
   for (int i = 0; i < nfaces; i++)
   {
      T = pmesh->GetSharedFaceTransformations(i);
      pfes->GetElementVDofs(T->Elem1No, vdofs1);
      pfes->GetFaceNbrElementVDofs(T->Elem2No, vdofs2);
      vdofs1.Copy(vdofs_all);

//      for (int j = 0; j < vdofs2.Size(); j++) FIXME: don't know what height is
//      {
//         vdofs2[j] += height;
//      }
      vdofs_all.Append(vdofs2);

//      for (int k = 0; k < vdofs2.Size(); k++) std::cout << k << '\t' << vdofs2[k] << std::endl;

      for (int k = 0; k < ilfi.Size(); k++)
      {
          ilfi[k] -> AssembleRHSElementVect (*pfes->GetFE(T -> Elem1No),
                                             *pfes->GetFaceNbrFE(T -> Elem2No),
                                              *T, elemvect);

          AddElementVector (vdofs_all, elemvect);
       }
    }
}


void ParLinearForm::Assemble()
{
   LinearForm::Assemble();

   if (ilfi.Size() > 0)
   {
      AssembleSharedFaces();
   }
}


}

#endif
