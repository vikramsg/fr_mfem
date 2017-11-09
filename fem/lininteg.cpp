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


#include <cmath>
#include "fem.hpp"

namespace mfem
{

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

void LinearFormIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("LinearFormIntegrator::AssembleRHSElementVect(...)");
}

void DomainLFIntegrator::AssembleRHSElementVect(const FiniteElement &el,
                                                ElementTransformation &Tr,
                                                Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);       // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // ir = &IntRules.Get(el.GetGeomType(),
      //                    oa * el.GetOrder() + ob + Tr.OrderW());
      ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);        // vector of size dof
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = Tr.Weight() * Q.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight * val, shape, elvect);
   }
}

void BoundaryNormalLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector nor(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      elvect.Add(ip.weight*(Qvec*nor), shape);
   }
}

void BoundaryTangentialLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();
   Vector tangent(dim), Qvec;

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   if (dim != 2)
   {
      mfem_error("These methods make sense only in 2D problems.");
   }

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = oa * el.GetOrder() + ob;  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      const DenseMatrix &Jac = Tr.Jacobian();
      tangent(0) =  Jac(0,0);
      tangent(1) = Jac(1,0);

      Q.Eval(Qvec, Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, ip.weight*(Qvec*tangent), shape, elvect);
   }
}

void VectorDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   double val,cf;

   shape.SetSize(dof);       // vector of size dof

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      val = Tr.Weight();

      el.CalcShape(ip, shape);
      Q.Eval (Qvec, Tr, ip);

      for (int k = 0; k < vdim; k++)
      {
         cf = val * Qvec(k);

         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += ip.weight * cf * shape(s);
         }
      }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Q.Eval(vec, Tr, ip);
      Tr.SetIntPoint (&ip);
      vec *= Tr.Weight() * ip.weight;
      el.CalcShape(ip, shape);
      for (int k = 0; k < vdim; k++)
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
   }
}

void VectorBoundaryLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int vdim = Q.GetVDim();
   int dof  = el.GetDof();

   shape.SetSize(dof);
   vec.SetSize(vdim);

   elvect.SetSize(dof * vdim);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = el.GetOrder() + 1;
      ir = &IntRules.Get(Tr.FaceGeom, intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      Tr.Face->SetIntPoint(&ip);
      Q.Eval(vec, *Tr.Face, ip);
      vec *= Tr.Face->Weight() * ip.weight;
      el.CalcShape(eip, shape);
      for (int k = 0; k < vdim; k++)
      {
         for (int s = 0; s < dof; s++)
         {
            elvect(dof*k+s) += vec(k) * shape(s);
         }
      }
   }
}


void VectorFEDomainLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   int spaceDim = Tr.GetSpaceDim();

   vshape.SetSize(dof,spaceDim);
   vec.SetSize(spaceDim);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // int intorder = 2*el.GetOrder() - 1; // ok for O(h^{k+1}) conv. in L2
      int intorder = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      el.CalcVShape(Tr, vshape);

      QF.Eval (vec, Tr, ip);
      vec *= ip.weight * Tr.Weight();

      vshape.AddMult (vec, elvect);
   }
}


void VectorBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dim = el.GetDim()+1;
   int dof = el.GetDof();

   shape.SetSize (dof);
   nor.SetSize (dim);
   elvect.SetSize (dim*dof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      ir = &IntRules.Get(el.GetGeomType(), el.GetOrder() + 1);
   }

   elvect = 0.0;
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Tr.SetIntPoint (&ip);
      CalcOrtho(Tr.Jacobian(), nor);
      el.CalcShape (ip, shape);
      nor *= Sign * ip.weight * F -> Eval (Tr, ip);
      for (int j = 0; j < dof; j++)
         for (int k = 0; k < dim; k++)
         {
            elvect(dof*k+j) += nor(k) * shape(j);
         }
   }
}


void VectorFEBoundaryFluxLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();

   shape.SetSize(dof);
   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint (&ip);
      double val = ip.weight*F.Eval(Tr, ip);

      el.CalcShape(ip, shape);

      add(elvect, val, shape, elvect);
   }
}


void VectorFEBoundaryTangentLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   int dof = el.GetDof();
   DenseMatrix vshape(dof, 2);
   Vector f_loc(3);
   Vector f_hat(2);

   elvect.SetSize(dof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int intorder = 2*el.GetOrder();  // <----------
      ir = &IntRules.Get(el.GetGeomType(), intorder);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetIntPoint(&ip);
      f.Eval(f_loc, Tr, ip);
      Tr.Jacobian().MultTranspose(f_loc, f_hat);
      el.CalcVShape(ip, vshape);

      Swap<double>(f_hat(0), f_hat(1));
      f_hat(0) = -f_hat(0);
      f_hat *= ip.weight;
      vshape.AddMult(f_hat, elvect);
   }
}


void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("BoundaryFlowIntegrator::AssembleRHSElementVect\n"
              "  is not implemented as boundary integrator!\n"
              "  Use LinearForm::AddBdrFaceIntegrator instead of\n"
              "  LinearForm::AddBoundaryIntegrator.");
}

void BoundaryFlowIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof, order;
   double un, w, vu_data[3], nor_data[3];

   dim  = el.GetDim();
   ndof = el.GetDof();
   Vector vu(vu_data, dim), nor(nor_data, dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // Assuming order(u)==order(mesh)
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   shape.SetSize(ndof);
   elvect.SetSize(ndof);
   elvect = 0.0;

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      el.CalcShape(eip, shape);

      Tr.Face->SetIntPoint(&ip);

      u->Eval(vu, *Tr.Elem1, eip);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      un = vu * nor;
      w = 0.5*alpha*un - beta*fabs(un);
      w *= ip.weight*f->Eval(*Tr.Elem1, eip);
      elvect.Add(w, shape);
   }
}


void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, ndof;
   bool kappa_is_nonzero = (kappa != 0.);
   double w;

   dim = el.GetDim();
   ndof = el.GetDof();

   nor.SetSize(dim);
   nh.SetSize(dim);
   ni.SetSize(dim);
   adjJ.SetSize(dim);
   if (MQ)
   {
      mq.SetSize(dim);
   }

   shape.SetSize(ndof);
   dshape.SetSize(ndof, dim);
   dshape_dn.SetSize(ndof);

   elvect.SetSize(ndof);
   elvect = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      // a simple choice for the integration order; is this OK?
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;

      Tr.Loc1.Transform(ip, eip);
      Tr.Face->SetIntPoint(&ip);
      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);
      Tr.Elem1->SetIntPoint(&eip);
      // compute uD through the face transformation
      w = ip.weight * uD->Eval(*Tr.Face, ip) / Tr.Elem1->Weight();
      if (!MQ)
      {
         if (Q)
         {
            w *= Q->Eval(*Tr.Elem1, eip);
         }
         ni.Set(w, nor);
      }
      else
      {
         nh.Set(w, nor);
         MQ->Eval(mq, *Tr.Elem1, eip);
         mq.MultTranspose(nh, ni);
      }
      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      adjJ.Mult(ni, nh);

      dshape.Mult(nh, dshape_dn);
      elvect.Add(sigma, dshape_dn);

      if (kappa_is_nonzero)
      {
         elvect.Add(kappa*(ni*nor), shape);
      }
   }
}


void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGElasticityDirichletLFIntegrator::AssembleRHSElementVect");
}

void DGElasticityDirichletLFIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   MFEM_ASSERT(Tr.Elem2No < 0, "interior boundary is not supported");

#ifdef MFEM_THREAD_SAFE
   Vector shape;
   DenseMatrix dshape;
   DenseMatrix adjJ;
   DenseMatrix dshape_ps;
   Vector nor;
   Vector dshape_dn;
   Vector dshape_du;
   Vector u_dir;
#endif

   const int dim = el.GetDim();
   const int ndofs = el.GetDof();
   const int nvdofs = dim*ndofs;

   elvect.SetSize(nvdofs);
   elvect = 0.0;

   adjJ.SetSize(dim);
   shape.SetSize(ndofs);
   dshape.SetSize(ndofs, dim);
   dshape_ps.SetSize(ndofs, dim);
   nor.SetSize(dim);
   dshape_dn.SetSize(ndofs);
   dshape_du.SetSize(ndofs);
   u_dir.SetSize(dim);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      const int order = 2*el.GetOrder(); // <-----
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int pi = 0; pi < ir->GetNPoints(); ++pi)
   {
      const IntegrationPoint &ip = ir->IntPoint(pi);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);
      Tr.Face->SetIntPoint(&ip);
      Tr.Elem1->SetIntPoint(&eip);

      // Evaluate the Dirichlet b.c. using the face transformation.
      uD.Eval(u_dir, *Tr.Face, ip);

      el.CalcShape(eip, shape);
      el.CalcDShape(eip, dshape);

      CalcAdjugate(Tr.Elem1->Jacobian(), adjJ);
      Mult(dshape, adjJ, dshape_ps);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      double wL, wM, jcoef;
      {
         const double w = ip.weight / Tr.Elem1->Weight();
         wL = w * lambda->Eval(*Tr.Elem1, eip);
         wM = w * mu->Eval(*Tr.Elem1, eip);
         jcoef = kappa * (wL + 2.0*wM) * (nor*nor);
         dshape_ps.Mult(nor, dshape_dn);
         dshape_ps.Mult(u_dir, dshape_du);
      }

      // alpha < uD, (lambda div(v) I + mu (grad(v) + grad(v)^T)) . n > +
      //   + kappa < h^{-1} (lambda + 2 mu) uD, v >

      // i = idof + ndofs * im
      // v_phi(i,d) = delta(im,d) phi(idof)
      // div(v_phi(i)) = dphi(idof,im)
      // (grad(v_phi(i)))(k,l) = delta(im,k) dphi(idof,l)
      //
      // term 1:
      //   alpha < uD, lambda div(v_phi(i)) n >
      //   alpha lambda div(v_phi(i)) (uD.n) =
      //   alpha lambda dphi(idof,im) (uD.n) --> quadrature -->
      //   ip.weight/det(J1) alpha lambda (uD.nor) dshape_ps(idof,im) =
      //   alpha * wL * (u_dir*nor) * dshape_ps(idof,im)
      // term 2:
      //   < alpha uD, mu grad(v_phi(i)).n > =
      //   alpha mu uD^T grad(v_phi(i)) n =
      //   alpha mu uD(k) delta(im,k) dphi(idof,l) n(l) =
      //   alpha mu uD(im) dphi(idof,l) n(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu uD(im) dshape_ps(idof,l) nor(l) =
      //   alpha * wM * u_dir(im) * dshape_dn(idof)
      // term 3:
      //   < alpha uD, mu (grad(v_phi(i)))^T n > =
      //   alpha mu n^T grad(v_phi(i)) uD =
      //   alpha mu n(k) delta(im,k) dphi(idof,l) uD(l) =
      //   alpha mu n(im) dphi(idof,l) uD(l) --> quadrature -->
      //   ip.weight/det(J1) alpha mu nor(im) dshape_ps(idof,l) uD(l) =
      //   alpha * wM * nor(im) * dshape_du(idof)
      // term j:
      //   < kappa h^{-1} (lambda + 2 mu) uD, v_phi(i) > =
      //   kappa/h (lambda + 2 mu) uD(k) v_phi(i,k) =
      //   kappa/h (lambda + 2 mu) uD(k) delta(im,k) phi(idof) =
      //   kappa/h (lambda + 2 mu) uD(im) phi(idof) --> quadrature -->
      //      [ 1/h = |nor|/det(J1) ]
      //   ip.weight/det(J1) |nor|^2 kappa (lambda + 2 mu) uD(im) phi(idof) =
      //   jcoef * u_dir(im) * shape(idof)

      wM *= alpha;
      const double t1 = alpha * wL * (u_dir*nor);
      for (int im = 0, i = 0; im < dim; ++im)
      {
         const double t2 = wM * u_dir(im);
         const double t3 = wM * nor(im);
         const double tj = jcoef * u_dir(im);
         for (int idof = 0; idof < ndofs; ++idof, ++i)
         {
            elvect(i) += (t1*dshape_ps(idof,im) + t2*dshape_dn(idof) +
                          t3*dshape_du(idof) + tj*shape(idof));
         }
      }
   }
}


void EulerIntegrator::getEulerFlux(const double R, const double gamm, const Vector &u, Vector &f)
{
    int var_dim = u.Size();
    int dim     = var_dim - 2;

    double rho  = u(0);

    Vector vel(dim);
    for (int i = 0; i < dim; i++)
    {
        vel(i) = u(1 + i)/rho;    
    }

    double v_sq = 0.0; 

    for (int i = 0; i < dim; i++)
    {
        v_sq += pow(vel(i), 2) ;
    }

    double p    = (u(var_dim - 1) - 0.5*rho*v_sq)*(gamm - 1);
    
    for (int i = 0; i < dim; i++)
    {
        f(i*var_dim + 0) = u(1 + i);
        for (int j = 0; j < dim; j++)
        {
            f(i*var_dim + 1 + j) = u(1 + i)*vel(j);
        }
        f(i*var_dim + 1 + i)    += p;

        f(i*var_dim + var_dim - 1) = (u(var_dim - 1) + p)*vel(i);
    }
}

/*
 * Get Interaction flux using the local Lax Friedrichs Riemann  
 * solver, u1 is the left value and u2 the right value
 */
void EulerIntegrator::getLFFlux(const double R, const double gamm, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f)
{
    int var_dim = u1.Size();
    int dim     = var_dim - 2;

    double Cv   = R/(gamm - 1);

    double rho_L = u1(0);
    
    Vector vel_L(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_L(i) = u1(1 + i)/rho_L;    
    }
    double E_L   = u1(var_dim - 1);

    double vel_sq_L = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_L += pow(vel_L(i), 2) ;
    }
    double T_L   = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
    double a_L   = sqrt(gamm * R * T_L);

    double rho_R = u2(0);
    Vector vel_R(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_R(i) = u2(1 + i)/rho_R;    
    }
    double E_R   = u2(var_dim - 1);

    double vel_sq_R = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_R += pow(vel_R(i), 2) ;
    }
    double T_R   = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
    double a_R   = sqrt(gamm * R * T_R);

    Vector nor_dim(dim);
    double nor_l2 = nor.Norml2();
    nor_dim.Set(1/nor_l2, nor);

    double vnl   = 0.0, vnr = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vnl += vel_L[i]*nor_dim(i); 
        vnr += vel_R[i]*nor_dim(i); 
    }

    double u_max = std::max(a_L + std::abs(vnl), a_R + std::abs(vnr));

    Vector fl(dim*var_dim), fr(dim*var_dim);
    getEulerFlux(R, gamm, u1, fl);
    getEulerFlux(R, gamm, u2, fr);
    add(0.5, fl, fr, f); //Common flux without dissipation

    Vector f_diss(dim*var_dim); //Rusanov dissipation 
    // Left and right depends on normal direction
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < var_dim; j++)
        {
            f_diss(i*var_dim + j) = -0.5*u_max*nor_dim(i)*(u2(j) - u1(j)); 
        }
    }
    add(f, f_diss, f);
}


/*
 * Get Interaction flux using the HLL 
 * solver, u1 is the left value and u2 the right value
 */
void EulerIntegrator::getHLLFlux(const double R, const double gamm, const Vector &u1, const Vector &u2, 
                                const Vector &nor, Vector &f)
{
    int var_dim = u1.Size();
    int dim     = var_dim - 2;

    double Cv   = R/(gamm - 1);

    double rho_L = u1(0);
    
    Vector vel_L(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_L(i) = u1(1 + i)/rho_L;    
    }
    double E_L   = u1(var_dim - 1);

    double vel_sq_L = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_L += pow(vel_L(i), 2) ;
    }
    double T_L   = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
    double a_L   = sqrt(gamm * R * T_L);
    double p_L   = rho_L*R*T_L; 
    double H_L   = E_L + p_L; 

    double rho_R = u2(0);
    Vector vel_R(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_R(i) = u2(1 + i)/rho_R;    
    }
    double E_R   = u2(var_dim - 1);

    double vel_sq_R = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_R += pow(vel_R(i), 2) ;
    }
    double T_R   = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
    double a_R   = sqrt(gamm * R * T_R);
    double p_R   = rho_R*R*T_R; 
    double H_R   = E_R + p_R; 

    Vector nor_dim(dim);
    double nor_l2 = nor.Norml2();
    nor_dim.Set(1/nor_l2, nor);

    double vnl   = 0.0, vnr = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vnl += vel_L[i]*nor_dim(i); 
        vnr += vel_R[i]*nor_dim(i); 
    }
    
    double rho_half = (rho_L + rho_R)/2.0; 
    double a_half   = (a_L + a_R)/2.0; 

    double v_half   = (vnl + vnr)/2.0 - (p_R - p_L)/(2*rho_half*a_half); 
    double p_half   = (p_L + p_R)/2.0 - (vnr - vnl)*rho_half*a_half/2.0; 

    p_half = std::max(0.0, p_half);

    double q_L, q_R;

    if (p_half <= p_L)
    {
        q_L = 1.0   ;
    }
    else
    {
        q_L = sqrt(1 + ((gamm + 1)/(2*gamm))*(p_half/p_L - 1)  );
    }
    if (p_half <= p_R)
    {
        q_R = 1.0; 
    }
    else
    {
        q_R = sqrt(1 + ((gamm + 1)/(2*gamm))*(p_half/p_R - 1)  );
    }

    double s_L, s_R, s_half;

    s_L = vnl - a_L*q_L;
    s_R = vnr + a_R*q_R;

    s_half = v_half;

    Vector fl(dim*var_dim), fr(dim*var_dim);
    getEulerFlux(R, gamm, u1, fl);
    getEulerFlux(R, gamm, u2, fr);

    if (0 <= s_L)
    {
        f = fl;
    }
    else if ((s_L < 0 ) && (0 < s_R))
    {
        fl *= s_R;    
        fr *= -s_L;    
        add(fl, fr, f);
        f  *= 1/(s_R - s_L);
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < var_dim; j++)
            {
                f(i*var_dim + j) += (s_L*s_R)*(u2(j) - u1(j))/(s_R - s_L); 
            }
        }
    }
    else
    {
        f = fr;
    }

}




/*
 * Get Interaction flux using a simple central flux 
 * solver, u1 is the left value and u2 the right value
 */
void EulerIntegrator::getConvectiveFlux(const double R, const double gamm, const Vector &u1, const Vector &u2, 
                                        const Vector &nor, Vector &f)
{
    int var_dim = u1.Size();
    int dim     = var_dim - 2;

    double Cv   = R/(gamm - 1);

    double rho_L = u1(0);
    
    Vector vel_L(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_L(i) = u1(1 + i)/rho_L;    
    }
    double E_L   = u1(var_dim - 1);

    double vel_sq_L = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_L += pow(vel_L(i), 2) ;
    }
    double T_L   = (E_L - 0.5*rho_L*vel_sq_L)/(rho_L*Cv);
    double a_L   = sqrt(gamm * R * T_L);

    double rho_R = u2(0);
    Vector vel_R(dim);
    for (int i = 0; i < dim; i++)
    {
        vel_R(i) = u2(1 + i)/rho_R;    
    }
    double E_R   = u2(var_dim - 1);

    double vel_sq_R = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vel_sq_R += pow(vel_R(i), 2) ;
    }
    double T_R   = (E_R - 0.5*rho_R*vel_sq_R)/(rho_R*Cv);
    double a_R   = sqrt(gamm * R * T_R);

    Vector nor_dim(dim);
    double nor_l2 = nor.Norml2();
    nor_dim.Set(1/nor_l2, nor);

    double vnl   = 0.0, vnr = 0.0;
    for (int i = 0; i < dim; i++)
    {
        vnl += vel_L[i]*nor_dim(i); 
        vnr += vel_R[i]*nor_dim(i); 
    }

    double u_max = std::max(a_L + std::abs(vnl), a_R + std::abs(vnr));

    Vector fl(dim*var_dim), fr(dim*var_dim);
    getEulerFlux(R, gamm, u1, fl);
    getEulerFlux(R, gamm, u2, fr);
    add(0.5, fl, fr, f); //Common flux without dissipation
}




void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}

void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("DGEEulerIntegrator::AssembleRHSElementVect");
}



void DGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, ndof1, ndof2;

   double un, a, b, w;

   Vector shape1, shape2;

   dim = el1.GetDim();
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(vDim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(vDim), u2_dir(vDim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      Vector f_dir(dim*vDim);

      getLFFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs

      Vector f1_dir(dim*vDim), f2_dir(dim*vDim);
      fD.Eval(f1_dir, *Trans.Elem1, eip1); // Get discontinuous flux at face
      fD.Eval(f2_dir, *Trans.Elem2, eip2);


      Vector face_f(vDim), face_f1(vDim), face_f2(vDim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; face_f2 = 0.0;
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < vDim; j++)
          {
              face_f1(j) += f1_dir(i*vDim + j)*nor(i);
              face_f2(j) += f2_dir(i*vDim + j)*nor(i);
              face_f (j) += f_dir (i*vDim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)              += face_f1(j)*w*shape1(i); 
          }
      }

      subtract(face_f, face_f2, face_f2); //fcomm - f2
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(vDim*ndof1 + j*ndof2 + i) -= face_f2(j)*w*shape2(i); 
          }
      }

   }// for ir loop
//      std::cout << u_L << '\t' << u_max << '\t' << face_f(0) << std::endl;
//      std::cout << Trans.Elem1No << '\t' << Trans.Elem2No << '\t' << nor_dim(0) << '\t' << nor_dim(1) << std::endl;
//      std::cout << p << '\t' << nor_dim(0) << '\t' << nor_dim(1) << std::endl;
//      std::cout << p << '\t' << f_dir(0) << '\t' << f_dir(1) << '\t' << f_dir(2) << '\t' << f_dir(3) << std::endl;

//      std::cout << p << '\t' << f_diss(0) << '\t' << f_diss(1) << '\t' << f_diss(2) << '\t' << f_diss(3) << std::endl;
//      std::cout << p << '\t' << f_dir(0) << '\t' << f_dir(1) << '\t' << f_dir(2) << '\t' << f_dir(3) << std::endl;
//
//      std::cout << p << '\t' << u1_dir(0) << '\t' << u1_dir(1) << '\t' << u1_dir(2) << '\t' << u1_dir(3) << std::endl;


//   std::cout << elvect(0) << '\t' << elvect(1) << '\t' << elvect(2) << '\t' << elvect(3) << std::endl;
//   std::cout << elvect(0) << '\t' << elvect(4) << '\t' << elvect(8) << '\t' << elvect(12) << std::endl;

}


void mDGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("mDGEulerIntegrator::AssembleRHSElementVect");
}

void mDGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("mDGEEulerIntegrator::AssembleRHSElementVect");
}



void mDGEulerIntegrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, ndof1, ndof2;
   double un, a, b, w;

   dim  = el1.GetDim();
   vDim = dim + 2; // CNS variable dimension
   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   
   vu.SetSize(dim); nor.SetSize(dim);

   elvect.SetSize(vDim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   double tol = std::numeric_limits<double>::epsilon();

   Poly_1D poly1, poly2;
   int p1_basis = el1.GetOrder(), p2_basis = el2.GetOrder(), type = 0;
   
   int Np1 = p1_basis + 1, Np2 = p2_basis + 1;
   Poly_1D::Basis &basis1(poly1.OpenBasis(p1_basis, type));
   Poly_1D::Basis &basis2(poly2.OpenBasis(p2_basis, type));

   const double *pts1 = poly1.GetPoints(p1_basis, type);
   P.SetSize(Np1);
   van1.SetSize(Np1);
   for (int i = 0; i < Np1; i++)
   {
       poly1.CalcLegendreBasis(p1_basis, pts1[i], P);
       van1.SetRow(i, P);
   
       for (int j = 0; j < Np1; j++)
       {
           double p_gamma = 2.0/(2.0*j + 1);
           van1(i, j)   = van1(i, j)/sqrt(p_gamma);
       }
   }
   const double *pts2 = poly2.GetPoints(p2_basis, type);
   P.SetSize(Np2);
   van2.SetSize(Np2);
   for (int i = 0; i < Np2; i++)
   {
       poly2.CalcLegendreBasis(p2_basis, pts2[i], P);
       van2.SetRow(i, P);
   
       for (int j = 0; j < Np2; j++)
       {
           double p_gamma = 2.0/(2.0*j + 1);
           van2(i, j)   = van2(i, j)/sqrt(p_gamma);
       }
   }

   van1.Transpose(); van1.Invert();
   van2.Transpose(); van2.Invert();

   mod_lag1_R.SetSize(Np1); mod_lag1_L.SetSize(Np1);
   P.SetSize(Np1);
   poly1.CalcLegendreBasis(p1_basis, 0.0, P);
   for (int j = 0; j < Np1; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np1 - 1] *= m_fac;
   van1.Mult(P, mod_lag1_L);
   poly1.CalcLegendreBasis(p1_basis, 1.0, P);
   for (int j = 0; j < Np1; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np1 - 1] *= m_fac;
   van1.Mult(P, mod_lag1_R);

   mod_lag2_R.SetSize(Np2); mod_lag2_L.SetSize(Np2);
   P.SetSize(Np2);
   poly2.CalcLegendreBasis(p2_basis, 0.0, P);
   for (int j = 0; j < Np2; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np2 - 1] *= m_fac;
   van2.Mult(P, mod_lag2_L);
   poly2.CalcLegendreBasis(p2_basis, 1.0, P);
   for (int j = 0; j < Np2; j++)
   {
       double p_gamma = 2.0/(2.0*j + 1);
       P[j] = P[j]/sqrt(p_gamma);
   }
   P[Np2 - 1] *= m_fac;
   van2.Mult(P, mod_lag2_R);


   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }
   
   mod_shape1.SetSize(ndof1); mod_shape2.SetSize(ndof2);

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(vDim), u2_dir(vDim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      Vector f_dir(dim*vDim);

      getLFFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs

      Vector f1_dir(dim*vDim), f2_dir(dim*vDim);
      fD.Eval(f1_dir, *Trans.Elem1, eip1); // Get discontinuous flux at face
      fD.Eval(f2_dir, *Trans.Elem2, eip2);


      Vector face_f(vDim), face_f1(vDim), face_f2(vDim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; face_f2 = 0.0;
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < vDim; j++)
          {
              face_f1(j) += f1_dir(i*vDim + j)*nor(i);
              face_f2(j) += f2_dir(i*vDim + j)*nor(i);
              face_f (j) += f_dir (i*vDim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      shape_x.SetSize(Np1); shape_y.SetSize(Np1);
      basis1.Eval(eip1.x, shape_x);
      basis1.Eval(eip1.y, shape_y);

      if (std::abs(eip1.x) < tol)
      {
          shape_x = mod_lag1_L;              
      }
      else if ( 1 - std::abs(eip1.x) < tol)
      {
          shape_x = mod_lag1_R;              
      }
      if (std::abs(eip1.y) < tol)
      {
          shape_y = mod_lag1_L;              
      }
      else if ( 1 - std::abs(eip1.y) < tol)
      {
          shape_y = mod_lag1_R;              
      }

      if (dim == 2)
      {
          mod_shape1 = 0.0;
          for (int ot = 0, jt = 0; jt <= p1_basis; jt++)
              for (int it = 0; it <= p1_basis; it++)
              {
                  mod_shape1(ot++) = shape_x(it)*shape_y(jt);
              }
      }
      else if (dim == 3)
      {
          shape_z.SetSize(Np1); 
          basis1.Eval(eip1.z, shape_z);
          if (std::abs(eip1.z) < tol)
          {
              shape_z = mod_lag1_L;              
          }
          else if ( 1 - std::abs(eip1.z) < tol)
          {
              shape_z = mod_lag1_R;              
          }
          for (int ot = 0, kt = 0; kt <= p1_basis; kt++)
              for (int jt = 0; jt <= p1_basis; jt++)
                  for (int it = 0; it <= p1_basis; it++)
                  {
                      mod_shape1(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                  }
      }
//      std::cout << shape_x[0] << "\t" << shape_x[1] << std::endl;
      
      shape_x.SetSize(Np2); shape_y.SetSize(Np2);
      basis2.Eval(eip2.x, shape_x);
      basis2.Eval(eip2.y, shape_y);

      if (std::abs(eip2.x) < tol)
      {
          shape_x = mod_lag2_L;              
      }
      else if ( 1 - std::abs(eip2.x) < tol)
      {
          shape_x = mod_lag2_R;              
      }
      if (std::abs(eip2.y) < tol)
      {
          shape_y = mod_lag2_L;              
      }
      else if ( 1 - std::abs(eip2.y) < tol)
      {
          shape_y = mod_lag2_R;              
      }

      if (dim == 2)
      {
          mod_shape2 = 0.0;
          for (int ot = 0, jt = 0; jt <= p2_basis; jt++)
              for (int it = 0; it <= p2_basis; it++)
              {
                  mod_shape2(ot++) = shape_x(it)*shape_y(jt);
              }
      }
      else if (dim == 3)
      {
          shape_z.SetSize(Np2); 
          basis2.Eval(eip2.z, shape_z);
          if (std::abs(eip2.z) < tol)
          {
              shape_z = mod_lag2_L;              
          }
          else if ( 1 - std::abs(eip2.z) < tol)
          {
              shape_z = mod_lag2_R;              
          }
          for (int ot = 0, kt = 0; kt <= p2_basis; kt++)
              for (int jt = 0; jt <= p2_basis; jt++)
                  for (int it = 0; it <= p2_basis; it++)
                  {
                      mod_shape2(ot++) = shape_x(it)*shape_y(jt)*shape_z(kt);                          
                  }
      }

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)              += face_f1(j)*w*mod_shape1(i); 
          }
      }

      subtract(face_f, face_f2, face_f2); //fcomm - f2
      for (int j = 0; j < vDim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(vDim*ndof1 + j*ndof2 + i) -= face_f2(j)*w*mod_shape2(i); 
          }
      }

   }// for ir loop
}




void FaceInt::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}

void FaceInt::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("DGEEulerIntegrator::AssembleRHSElementVect");
}



void FaceInt::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, aux_dim, ndof1, ndof2;

   double un, a, b, w;

   Vector shape1, shape2;

   dim = el1.GetDim();
   aux_dim = dim + 1;

   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(aux_dim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(aux_dim), u2_dir(aux_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      Vector dir_(dim);
      dir.Eval(dir_, *Trans.Elem1, eip1);

      double un = dir_*nor;

      w = ip.weight * alpha * un; 

      Vector u_common(aux_dim);

      add(0.5, u1_dir, u2_dir, u_common);

      subtract(u_common, u1_dir, u1_dir); //f_comm - f1
      for (int j = 0; j < aux_dim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)                   += u1_dir(j)*w*shape1(i); 
          }
      }

      subtract(u_common, u2_dir, u2_dir); //fcomm - f2
      for (int j = 0; j < aux_dim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(aux_dim*ndof1 + j*ndof2 + i)  -= u2_dir(j)*w*shape2(i); 
          }
      }

   }// for ir loop

}

void DG_CNS_Aux_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}


void DG_CNS_Aux_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, aux_dim, ndof;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   aux_dim = dim + 1;

   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(aux_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(aux_dim), u2_bnd(aux_dim), u2_dir(aux_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Trans.Elem1, eip);

      Vector nor_dim(dim);
      double nor_l2 = nor.Norml2();
      nor_dim.Set(1/nor_l2, nor);

      Vector vel_L(dim);
      for (int j = 0; j < dim; j++) 
      {
          vel_L(j) = u1_dir(j);
      }

      u2_dir = u1_dir;

      Vector vel_R(dim);
      for (int j = 0; j < dim; j++) 
      {
          vel_R(j)  = 2*u2_bnd(j) - vel_L(j);
          u2_dir(j) = vel_R(j);
      }

      Vector dir_(dim);
      dir.Eval(dir_, *Trans.Elem1, eip);

      double un = dir_*nor;
      w = ip.weight * alpha * un; 

      Vector u_common(aux_dim);

      add(0.5, u1_dir, u2_dir, u_common);

      subtract(u_common, u1_dir, u1_dir); //f_comm - f1
      for (int j = 0; j < aux_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)            += u1_dir(j)*w*shape(i); 
          }
      }

   }// for ir loop

}

void DG_Euler_NoSlip_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}



void DG_Euler_NoSlip_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, ndof, aux_dim;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   aux_dim = dim + 1;
   var_dim = dim + 2;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
         
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(var_dim), u2_dir(var_dim);
      Vector u2_bnd(aux_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Trans.Elem1, eip);

      Vector vel_L(dim);    
      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);

      double p_R = p_L; // Extrapolate pressure
      Vector vel_R(dim);   
      v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = u2_bnd(j);      
//          vel_R(j) = 2*u2_bnd(j) - vel_L(j);      
          v_sq    += pow(vel_R(j), 2);
      }
//      double T_R   = u2_bnd(aux_dim - 1);
//      double rho_R = p_R/(R*T_R);
      double rho_R = rho_L;
      double E_R   = p_R/(gamm - 1) + 0.5*rho_R*v_sq;

      u2_dir(0) = rho_R;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_R*vel_R(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;

      Vector f_dir(dim*var_dim);
//      getLFFlux(u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs
      getConvectiveFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using central convective flux 

      Vector f1_dir(dim*var_dim);
      fD.Eval(f1_dir, *Trans.Elem1, eip); // Get discontinuous flux at face

      Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}

void DG_Euler_Slip_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}



void DG_Euler_Slip_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, ndof, aux_dim;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   aux_dim = dim + 1;
   var_dim = dim + 2;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
         
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(var_dim), u2_dir(var_dim);
      Vector u2_bnd(aux_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Trans.Elem1, eip);

      Vector nor_dim(dim);
      double nor_l2 = nor.Norml2();
      nor_dim.Set(1/nor_l2, nor);

      Vector vel_L(dim);    
      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);

      double vn  = vel_L*nor_dim;// Dot product of velocity and normal

      double p_R = p_L; // Extrapolate pressure
      Vector vel_R(dim);   
      v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = u2_bnd(j) + (vel_L(j) - 2*vn*nor_dim(j)); // Negate normal velocity     
          v_sq    += pow(vel_R(j), 2);
      }
      double rho_R = rho_L;
      double E_R   = p_R/(gamm - 1) + 0.5*rho_R*v_sq;

      u2_dir(0) = rho_R;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_R*vel_R(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;

      Vector f_dir(dim*var_dim);
//      getLFFlux(u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs
      getConvectiveFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using central convective flux 

      Vector f1_dir(dim*var_dim);
      fD.Eval(f1_dir, *Trans.Elem1, eip); // Get discontinuous flux at face

      Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}




void DG_Euler_Characteristic_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}



void DG_Euler_Characteristic_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, ndof, aux_dim;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   var_dim = dim + 2;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
         
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(var_dim), u2_dir(var_dim);
      Vector u2_bnd(var_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_dir, *Trans.Elem1, eip);

      Vector f_dir(dim*var_dim);
      getLFFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs

      Vector f1_dir(dim*var_dim);
      fD.Eval(f1_dir, *Trans.Elem1, eip); // Get discontinuous flux at face

      Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}




void DG_Euler_Subsonic_Pressure_Outflow_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DGEulerIntegrator::AssembleRHSElementVect");
}



void DG_Euler_Subsonic_Pressure_Outflow_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, ndof, aux_dim;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   var_dim = dim + 2;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
         
      order = Trans.Elem1->OrderW() + 2*el.GetOrder();
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Trans.Loc1.Transform(ip, eip);

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(var_dim), u2_dir(var_dim);
      Vector u2_bnd(var_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip);
      u_bnd.Eval(u2_dir, *Trans.Elem1, eip);

      Vector vel_L(dim);    
      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);

      Vector vel_R(dim);    
      double rho_R = u2_dir(0);
      double v2_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = u2_dir(1 + j)/rho_R;      
          v2_sq    += pow(vel_R(j), 2);
      }
      double p_R   = (gamm - 1)*(u2_dir(var_dim - 1) - 0.5*rho_R*v2_sq);
      double E_R   = (2*p_R - p_L)/(gamm - 1) + 0.5*rho_L*v_sq; // PNR BC AIAA 2014-2923

      u2_dir(0) = rho_L;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_L*vel_L(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;


      Vector f_dir(dim*var_dim);
      getLFFlux(R, gamm, u1_dir, u2_dir, nor, f_dir); // Get interaction flux at face using local Lax Friedrichs

      Vector f1_dir(dim*var_dim);
      fD.Eval(f1_dir, *Trans.Elem1, eip); // Get discontinuous flux at face

      Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}



void CNSIntegrator::getViscousCNSFlux(double R, double gamm, const Vector &u, const Vector &aux_grad, 
                                        const double mu, const double Pr, Vector &f)
{
    int var_dim = u.Size();
    int dim     = var_dim - 2;
    int aux_dim = dim + 1;

    double rho = u[0], E = u[var_dim - 1];

    double rho_vel[dim];
    for(int i = 0; i < dim; i++) rho_vel[i] = u[1 + i];

    double vel[dim];        
    for(int j = 0; j < dim; j++) vel[j]   = rho_vel[j]/rho;

    double vel_grad[dim][dim];
    for (int k = 0; k < dim; k++)
        for (int j = 0; j < dim; j++)
        {
            vel_grad[j][k]      =  aux_grad[k*(aux_dim) + j];
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
        int_en_grad[j] = (R/(gamm - 1))*aux_grad[j*(aux_dim) + (aux_dim - 1)] ; // Cv*T_x
    }

    for (int j = 0; j < dim ; j++)
    {
        f(j*var_dim)       = 0.0;

        for (int k = 0; k < dim ; k++)
        {
            f(j*var_dim + (k + 1))        = tau[j][k];
        }
        f(j*var_dim + (var_dim - 1))      =  (mu/Pr)*gamm*int_en_grad[j]; 
        for (int k = 0; k < dim ; k++)
        {
            f(j*var_dim + (var_dim - 1)) += vel[k]*tau[j][k]; 
        }
    }
}


void DG_CNS_Vis_Adiabatic_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DG_CNS_Vis_Ad_Integrator::AssembleRHSElementVect");
}


void DG_CNS_Vis_Adiabatic_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   int dim, var_dim, aux_dim, ndof;

   double un, a, b, w;

   Vector shape;

   dim = el.GetDim();
   var_dim = dim + 2;
   aux_dim = dim + 1;
   ndof = el.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof));
   elvect = 0.0;

   shape.SetSize(ndof);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
     
      order = Tr.Elem1->OrderW() + 2*el.GetOrder();
      
      if (el.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Tr.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip;
      Tr.Loc1.Transform(ip, eip);

      Tr.Face->SetIntPoint(&ip);
      Tr.Elem1->SetIntPoint(&eip);

      el.CalcShape(eip, shape);

      if (dim == 1)
      {
         nor(0) = 2*eip.x - 1.0;
      }
      else
      {
         CalcOrtho(Tr.Face->Jacobian(), nor);
      }

      Vector nor_dim(dim);
      double nor_l2 = nor.Norml2();
      nor_dim.Set(1/nor_l2, nor);

      Vector u1_dir(var_dim), u2_dir(var_dim);
      Vector u2_bnd(dim);
      uD.Eval(u1_dir, *Tr.Elem1, eip);
      u_bnd.Eval(u2_bnd, *Tr.Elem1, eip);

      Vector aux1_dir(dim*aux_dim), aux2_dir(dim*aux_dim);
      auxD.Eval(aux1_dir, *Tr.Elem1, eip);

//      ///////////
//      Vector f_dir(dim*var_dim);
//      getViscousCNSFlux(u1_dir, aux1_dir, mu, Pr, f_dir);
//
//      Vector f1_dir(dim*var_dim);
//      fD.Eval(f1_dir, *Tr.Elem1, eip);        // Get discontinuous flux at face
//      ////////////

      Vector vel_L(dim);    
      double rho_L = u1_dir(0);
      double v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_L(j) = u1_dir(1 + j)/rho_L;      
          v_sq    += pow(vel_L(j), 2);
      }
      double p_L = (gamm - 1)*(u1_dir(var_dim - 1) - 0.5*rho_L*v_sq);

      double p_R = p_L; // Extrapolate pressure
      Vector vel_R(dim);   
      v_sq  = 0.0;
      for (int j = 0; j < dim; j++)
      {
          vel_R(j) = u2_bnd(j);      
          v_sq    += pow(vel_R(j), 2);
      }
      double rho_R = rho_L;
      double E_R   = p_R/(gamm - 1) + 0.5*rho_R*v_sq;

      u2_dir(0) = rho_R;
      for (int j = 0; j < dim; j++)
      {
          u2_dir(1 + j)   = rho_R*vel_R(j)    ;
      }
      u2_dir(var_dim - 1) = E_R;

      aux2_dir = aux1_dir;

      Vector gradT(dim);
      for (int i = 0; i < dim; i++) gradT[i] = aux1_dir[i*aux_dim + aux_dim - 1];

      double T_dot_n = gradT*nor_dim;

      for (int i = 0; i < dim; i++)
      {
////           aux2_dir[i*aux_dim + aux_dim - 1] -= T_dot_n*nor_dim(i); // T_x = T_x - T_x.n
           aux2_dir[i*aux_dim + aux_dim - 1] = 0.0; // T_x = 0.0 
      }

      Vector fL_dir(dim*var_dim), fR_dir(dim*var_dim);
      Vector f_dir(dim*var_dim);
      getViscousCNSFlux(R, gamm, u1_dir, aux1_dir, mu, Pr, fL_dir);
      getViscousCNSFlux(R, gamm, u2_dir, aux2_dir, mu, Pr, f_dir);

//      for(int j = 0; j < var_dim; j++) std::cout << j << "\t" << fL_dir(var_dim + j) << "\t" << fR_dir(var_dim + j) << std::endl;

//      for(int j = 0; j < var_dim; j++) std::cout << j << "\t" << fL_dir(j) << "\t" << f_dir(j) << std::endl;

      Vector f1_dir(dim*var_dim);
      fD.Eval(f1_dir, *Tr.Elem1, eip);        // Get discontinuous flux at face

      Vector face_f(var_dim), face_f1(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; 
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof; i++)
          {
              elvect(j*ndof + i)              += face_f1(j)*w*shape(i); 
          }
      }

   }// for ir loop

}



void DG_Viscous_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DG_Viscous_Integrator::AssembleRHSElementVect");
}

void DG_Viscous_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("DG_Viscous_Integrator::AssembleRHSElementVect");
}




void DG_Viscous_Integrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, var_dim, aux_dim, ndof1, ndof2;

   double un, a, b, w;

   Vector shape1, shape2;

   dim      = el1.GetDim();
   var_dim  = dim + 2; 
   aux_dim  = dim + 2; 
   ndof1    = el1.GetDof();
   ndof2    = el2.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(var_dim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(var_dim), u2_dir(var_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      Vector aux1_dir(dim*aux_dim), aux2_dir(dim*aux_dim);
      auxD.Eval(aux1_dir, *Trans.Elem1, eip1);
      auxD.Eval(aux2_dir, *Trans.Elem2, eip2);

      Vector fL_dir(dim*var_dim), fR_dir(dim*var_dim);
      getViscousCNSFlux(R, gamm, u1_dir, aux1_dir, mu, Pr, fL_dir);
      getViscousCNSFlux(R, gamm, u2_dir, aux2_dir, mu, Pr, fR_dir);

//      Vector f_dir(dim*var_dim);
//      add(0.5, fL_dir, fR_dir, f_dir); //Common flux is taken as average

      Vector f1_dir(dim*var_dim), f2_dir(dim*var_dim);
      fD.Eval(f1_dir, *Trans.Elem1, eip1); // Get discontinuous flux at face
      fD.Eval(f2_dir, *Trans.Elem2, eip2);

      Vector f_dir(dim*var_dim);
      add(0.5, f1_dir, f2_dir, f_dir); //Common flux is taken as average


      Vector face_f(var_dim), face_f1(var_dim), face_f2(var_dim); //Face fluxes (dot product with normal)
      face_f = 0.0; face_f1 = 0.0; face_f2 = 0.0;
      for (int i = 0; i < dim; i++)
      {
          for (int j = 0; j < var_dim; j++)
          {
              face_f1(j) += f1_dir(i*var_dim + j)*nor(i);
              face_f2(j) += f2_dir(i*var_dim + j)*nor(i);
              face_f (j) += f_dir (i*var_dim + j)*nor(i);
          }
      }

      w = ip.weight * alpha; 

      subtract(face_f, face_f1, face_f1); //f_comm - f1
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)              += face_f1(j)*w*shape1(i); 
          }
      }

      subtract(face_f, face_f2, face_f2); //fcomm - f2
      for (int j = 0; j < var_dim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(var_dim*ndof1 + j*ndof2 + i) -= face_f2(j)*w*shape2(i); 
          }
      }

   }// for ir loop

}

void DG_Viscous_Aux_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, ElementTransformation &Tr, Vector &elvect)
{
   mfem_error("DG_Viscous_Aux_Integrator::AssembleRHSElementVect");
}

void DG_Viscous_Aux_Integrator::AssembleRHSElementVect(
   const FiniteElement &el, FaceElementTransformations &Tr, Vector &elvect)
{
   mfem_error("DG_Viscous_Aux_Integrator::AssembleRHSElementVect");
}



void DG_Viscous_Aux_Integrator::AssembleRHSElementVect(
   const FiniteElement &el1, const FiniteElement &el2, FaceElementTransformations &Trans, Vector &elvect)
{
   int dim, aux_dim, ndof1, ndof2;

   double un, a, b, w;

   Vector shape1, shape2;

   dim = el1.GetDim();
   aux_dim = dim + 1;

   ndof1 = el1.GetDof();
   ndof2 = el2.GetDof();
   
   Vector vu(dim), nor(dim);

   elvect.SetSize(aux_dim*(ndof1 + ndof2));
   elvect = 0.0;

   shape1.SetSize(ndof1);
   shape2.SetSize(ndof2);

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order;
      // Assuming order(u)==order(mesh)
      if (Trans.Elem2No >= 0)
         order = (std::min(Trans.Elem1->OrderW(), Trans.Elem2->OrderW()) +
                  2*std::max(el1.GetOrder(), el2.GetOrder()));
      else
      {
         order = Trans.Elem1->OrderW() + 2*el1.GetOrder();
      }
      if (el1.Space() == FunctionSpace::Pk)
      {
         order++;
      }
      ir = &IntRules.Get(Trans.FaceGeom, order);
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      IntegrationPoint eip1, eip2;
      Trans.Loc1.Transform(ip, eip1);
      if (ndof2)
      {
         Trans.Loc2.Transform(ip, eip2);
      }

      Trans.Face->SetIntPoint(&ip);
      Trans.Elem1->SetIntPoint(&eip1);
      Trans.Elem2->SetIntPoint(&eip2);

      el1.CalcShape(eip1, shape1);
      el2.CalcShape(eip2, shape2);

      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
         CalcOrtho(Trans.Face->Jacobian(), nor);
      }

      Vector u1_dir(aux_dim), u2_dir(aux_dim);
      uD.Eval(u1_dir, *Trans.Elem1, eip1);
      uD.Eval(u2_dir, *Trans.Elem2, eip2);

      Vector dir_(dim);
      dir.Eval(dir_, *Trans.Elem1, eip1);

      double un = dir_*nor;

      w = ip.weight * alpha * un; 

      Vector u_common(aux_dim);

      add(0.5, u1_dir, u2_dir, u_common);

      subtract(u_common, u1_dir, u1_dir); //f_comm - f1
      for (int j = 0; j < aux_dim; j++)
      {
          for (int i = 0; i < ndof1; i++)
          {
              elvect(j*ndof1 + i)                   += u1_dir(j)*w*shape1(i); 
          }
      }

      subtract(u_common, u2_dir, u2_dir); //fcomm - f2
      for (int j = 0; j < aux_dim; j++)
      {
          for (int i = 0; i < ndof2; i++)
          {
              elvect(aux_dim*ndof1 + j*ndof2 + i)  -= u2_dir(j)*w*shape2(i); 
          }
      }

   }// for ir loop

}



}
