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

#ifdef MFEM_USE_OCCA
#  ifndef MFEM_OSOLVERS
#  define MFEM_OSOLVERS

#include "solvers.hpp"
#include "ovector.hpp"

namespace mfem {
  inline void CG(const Operator &A, const OccaVector &b, OccaVector &x,
                 int print_iter = 0, int max_num_iter = 1000,
                 double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
  {
    TCG<OccaVector>(A, b, x,
                    print_iter, max_num_iter,
                    RTOLERANCE, ATOLERANCE);
  }

  inline void PCG(const Operator &A, Solver &B, const OccaVector &b, OccaVector &x,
                  int print_iter = 0, int max_num_iter = 1000,
                  double RTOLERANCE = 1e-12, double ATOLERANCE = 1e-24)
  {
    TPCG<OccaVector>(A, B, b, x,
                     print_iter, max_num_iter,
                     RTOLERANCE, ATOLERANCE);
  }
}

#  endif
#endif
