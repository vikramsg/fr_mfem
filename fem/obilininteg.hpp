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
#  ifndef MFEM_OCCABILININTEG
#  define MFEM_OCCABILININTEG

#include "obilinearform.hpp"

namespace mfem {
  class OccaDofQuadMaps {
  private:
    // Reuse dof-quad maps
    static std::map<occa::hash_t, OccaDofQuadMaps> AllDofQuadMaps;
    occa::hash_t hash;

  public:
    // Local stiffness matrices (B and B^T operators)
    occa::array<double> dofToQuad, dofToQuadD; // B
    occa::array<double> quadToDof, quadToDofD; // B^T

    OccaDofQuadMaps();
    OccaDofQuadMaps(const OccaDofQuadMaps &maps);
    OccaDofQuadMaps& operator = (const OccaDofQuadMaps &maps);

    static OccaDofQuadMaps& Get(occa::device device,
                                const H1_TensorBasisElement &el,
                                const IntegrationRule &ir);
  };

  //---[ Base Integrator ]--------------
  class OccaIntegrator {
  protected:
    occa::device device;

    OccaBilinearForm &bilinearForm;
    BilinearFormIntegrator *integrator;
    occa::properties props;
    OccaIntegratorType itype;

  public:
    OccaIntegrator(OccaBilinearForm &bilinearForm_);

    virtual ~OccaIntegrator();

    OccaIntegrator* CreateInstance(BilinearFormIntegrator &integrator_,
                                   const occa::properties &props_,
                                   const OccaIntegratorType itype_);

    virtual OccaIntegrator* CreateInstance() = 0;

    virtual void Setup();
    virtual void Assemble() = 0;
    virtual void Mult(OccaVector &x) = 0;

    occa::kernel GetAssembleKernel(const occa::properties &props);
    occa::kernel GetMultKernel(const occa::properties &props);

    occa::kernel GetKernel(const std::string &kernelName,
                           const occa::properties &props);
  };
  //====================================

  //---[ Diffusion Integrator ]---------
  class OccaDiffusionIntegrator : public OccaIntegrator {
  private:
    OccaDofQuadMaps maps;

    occa::kernel assembleKernel, multKernel;

    occa::array<double> coefficients;
    occa::array<double> jacobian, assembledOperator;

    bool hasConstantCoefficient;

  public:
    OccaDiffusionIntegrator(OccaBilinearForm &bilinearForm_);
    virtual ~OccaDiffusionIntegrator();

    virtual OccaIntegrator* CreateInstance();

    virtual void Setup();

    virtual void Assemble();
    virtual void Mult(OccaVector &x);
  };
  //====================================
}

#  endif
#endif