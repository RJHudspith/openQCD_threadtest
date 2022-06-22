
/*******************************************************************************
*
* File force3.c
*
* Copyright (C) 2012-2018, 2020, 2022 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Rational function forces.
*
*   qflt setpf3(int *irat,int ipf,int isw,int isp,int icom,int *status)
*     Generates a pseudo-fermion field phi with probability proportional
*     to exp(-Spf) and returns the action Spf+Sdet-(phi,phi) if isw=1 or
*     Spf-(phi,phi) if isw!=1 (see the notes).
*
*   void rotpf3(int *irat,int ipf,int isp,double c1,double c2,int *status)
*     Generates a pseudo-fermion field eta with probability proportional
*     to exp(-Spf) and replaces phi by c1*phi+c2*eta (see the notes).
*
*   void force3(int *irat,int ipf,int isw,int isp,double c,int *status)
*     Computes the force deriving from the action Spf+Sdet if isw=1 or
*     Spf if isw!=1 (see the notes). The calculated force is multiplied
*     by c and added to the molecular-dynamics force field.
*
*   qflt action3(int *irat,int ipf,int isw,int isp,int icom,int *status)
*     Returns the action Spf+Sdet-(phi,phi) if isw=1 or Spf-(phi,phi) if
*     isw!=1 (see the notes).
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* Simulations including the charm and/or the strange quark are based on
* a version of the RHMC algorithm. See the notes "Charm and strange quark
* in openQCD simulations" (file doc/rhmc.pdf).
*
* The pseudo-fermion action Spf is given by
*
*   Spf=(phi,P_{k,l}*phi),
*
* where P_{k,l} is the fraction of a Zolotarev rational function, which
* is defined by the parameters:
*
*   irat[0]       Index of the Zolotarev rational function in the
*                 parameter data base.
*
*   irat[1]       Lower end k of the selected coefficient range.
*
*   irat[2]       Upper end l of the selected coefficient range.
*
* See ratfcts/ratfcts.c for further explanations. The inclusion of the
* "small quark determinant" amounts to adding the action
*
*   Sdet=-ln{det(1e+Doo)}+constant
*
* to the molecular-dynamics Hamilton function, where 1e is the projector
* to the quark fields that vanish on the odd lattice sites and Doo the
* odd-odd component of the Dirac operator (the constant is adjusted so
* as to reduce the significance losses when the action differences are
* computed at the end of the molecular-dynamics trajectories).
*
* The other parameters of the programs in this module are:
*
*   ipf           Index of the pseudo-fermion field phi in the
*                 structure returned by mdflds() [mdflds.c].
*
*   isp           Index of the solver parameter set describing the
*                 solver to be used for the solution of the Dirac
*                 equation.
*
*   icom          The action returned by the programs setpf3() and
*                 action3() is summed over all MPI processes if the
*                 bit (icom&0x1) is set. Otherwise the local part of
*                 the action is returned.
*
*   status        Status values returned by the solvers used for the
*                 solution of the Dirac equation. These are the total
*                 (MSCG) or average (SAP_GCR, DFL_SAP_GCR) numbers of
*                 Krylov vectors generated by the solver.
*
* The supported solvers are MSCG, SAP_GCR and DFL_SAP_GCR. In all cases
* the status array must be of the standard length (see utils/futils.c).
*
* The bare quark mass m0 is the one last set by sw_parms() [flags/lat_parms.c]
* and it is taken for granted that the solver parameters have been set by
* set_solver_parms() [flags/solver_parms.c].
*
* Some debugging information is printed to stdout on MPI process 0 if the
* macro FORCE_DBG is defined at compilation time.
*
* The required workspaces of double-precision spinor fields are
*
*                   MSCG        SAP_GCR       DFL_SAP_GCR
*   setpf3()        np+1           2               2
*   rotpf3()        np+1           2               2
*   force3()        np+1           3               3
*   action3()       np+1           2               2
*
* where np is the number of poles of P_{k,l}. These figures do not include
* the workspace required by the solvers.
*
*******************************************************************************/

#define FORCE3_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "sw_term.h"
#include "sflds.h"
#include "dirac.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "forces.h"
#include "global.h"

static int nps=0;
static int ifail0[2],ifail1[2],*stat0,*stat1=NULL;
static double *rs;


static void init_stat(int *status)
{
   if (stat1==NULL)
   {
      stat0=alloc_std_status();
      stat1=alloc_std_status();
   }

   ifail0[0]=0;
   ifail0[1]=0;
   ifail1[0]=0;
   ifail1[1]=0;
   reset_std_status(stat0);
   reset_std_status(stat1);
   reset_std_status(status);
}

#if (defined FORCE_DBG)

static char *program[4]={"action3","force3","rotpf3","setpf3"};


static void solver_info(int ipgm,int np,double *mu,solver_parms_t sp)
{
   int k;

   if (sp.solver==MSCG)
      message("[%s]: MSCG solver, np = %d, istop = %d, mu = %.2e",
              program[ipgm],np,sp.istop,mu[0]);
   else if (sp.solver==SAP_GCR)
      message("[%s]: SAP_GCR solver, np = %d, istop = %d, mu = %.2e",
              program[ipgm],np,sp.istop,mu[0]);
   else if (sp.solver==DFL_SAP_GCR)
      message("[%s]: DFL_SAP_GCR solver, np = %d, istop = %d, mu = %.2e",
              program[ipgm],np,sp.istop,mu[0]);

   for (k=1;k<np;k++)
      message(", %.2e",mu[k]);

   message("\n");
}


static void check_flds0(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   sw_term(ODD_PTS);
   Dwhat_dble(mu,psi,rho);
   mulg5_dble(VOLUME_TRD/2,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD/2,3,rho);
      rpsi/=unorm_dble(VOLUME_TRD/2,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD/2,3,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

   release_wsd();
}


static void check_flds1(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi,spinor_dble *chi)
{
   double rchi,rpsi;
   qflt qnrm;
   spinor_dble *rho,**wsd;

   wsd=reserve_wsd(1);
   rho=wsd[0];

   if (sp.solver==MSCG)
   {
      scale_dble(VOLUME_TRD/2,2,-1.0,psi+(VOLUME/2));
      scale_dble(VOLUME_TRD/2,2,-1.0,chi+(VOLUME/2));
   }

   sw_term(NO_PTS);
   Dw_dble(-mu,chi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,rho,psi,-1.0);

   if (sp.istop)
   {
      rchi=unorm_dble(VOLUME_TRD,3,rho);
      rchi/=unorm_dble(VOLUME_TRD/2,3,psi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,rho);
      rchi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,psi);
      rchi/=qnrm.q[0];
      rchi=sqrt(rchi);
   }

   Dw_dble(mu,psi,rho);
   mulg5_dble(VOLUME_TRD,2,rho);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,rho,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD,3,rho);
      rpsi/=unorm_dble(VOLUME_TRD/2,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,rho);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   if (sp.solver==MSCG)
   {
      scale_dble(VOLUME_TRD/2,2,-1.0,psi+(VOLUME/2));
      scale_dble(VOLUME_TRD/2,2,-1.0,chi+(VOLUME/2));
      sw_term(ODD_PTS);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);
   message("[%s]: Residue of chi = %.1e (should be <= %.1e)\n",
           program[ipgm],rchi,sp.res);

   release_wsd();
}


static void check_flds2(int ipgm,double mu,solver_parms_t sp,
                        spinor_dble *phi,spinor_dble *psi)
{
   double rpsi;
   qflt qnrm;
   spinor_dble *rho,*chi,**wsd;

   wsd=reserve_wsd(2);
   rho=wsd[0];
   chi=wsd[1];

   sw_term(ODD_PTS);
   Dwhat_dble(-mu,psi,rho);
   mulg5_dble(VOLUME_TRD/2,2,rho);
   Dwhat_dble(mu,rho,chi);
   mulg5_dble(VOLUME_TRD/2,2,chi);
   mulr_spinor_add_dble(VOLUME_TRD/2,2,chi,phi,-1.0);

   if (sp.istop)
   {
      rpsi=unorm_dble(VOLUME_TRD/2,3,chi);
      rpsi/=unorm_dble(VOLUME_TRD/2,3,phi);
   }
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD/2,3,chi);
      rpsi=qnrm.q[0];
      qnrm=norm_square_dble(VOLUME_TRD/2,3,phi);
      rpsi/=qnrm.q[0];
      rpsi=sqrt(rpsi);
   }

   message("[%s]: Residue of psi = %.1e (should be <= %.1e)\n",
           program[ipgm],rpsi,sp.res);

   release_wsd();
}

#endif

static void set_res(int np,double res)
{
   int k;

   if (np>nps)
   {
      if (nps>0)
         free(rs);

      rs=malloc(np*sizeof(*rs));
      error(rs==NULL,1,"set_res [force3.c]",
            "Unable to allocate auxiliary array");
      nps=np;
   }

   for (k=0;k<np;k++)
      rs[k]=res;
}


qflt setpf3(int *irat,int ipf,int isw,int isp,int icom,int *status)
{
   int np,k;
   double *nu,*rnu,*qsm[1];
   qflt act0,act1;
   complex_dble z;
   spinor_dble *phi,*chi,**rsd;
   spinor_dble *psi,**wsd;
   mdflds_t *mdfs;
   ratfct_t rf;
   tm_parms_t tm;
   solver_parms_t sp;
   sap_parms_t sap;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   chi=NULL;

   rf=ratfct(irat);
   np=rf.np;
   nu=rf.nu;
   rnu=rf.rnu;

   if (isw==1)
      act0=small_det(0);
   else
   {
      act0.q[0]=0.0;
      act0.q[1]=0.0;
   }

   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(3,np,nu,sp);
#endif

   if (sp.solver==MSCG)
   {
      rsd=reserve_wsd(np+1);
      chi=rsd[np];

      random_sd(VOLUME_TRD/2,2,chi,1.0);
      bnd_sd2zero(EVEN_PTS,chi);
      assign_sd2sd(VOLUME_TRD/2,2,chi,phi);
      set_res(np,sp.res);

      tmcgm(sp.nmx,sp.istop,rs,np,nu,chi,rsd,ifail0,stat0);
      acc_std_status("tmcgm",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgm",ifail0,stat0);
         error_root(1,1,"setpf3 [force3.c]","MSCG solver failed "
                    "(irat=%d,%d,%d, parameter set no %d)",
                    irat[0],irat[1],irat[2],isp);
      }

      wsd=reserve_wsd(1);
      psi=wsd[0];
      set_sd2zero(VOLUME_TRD/2,2,chi);

      for (k=0;k<np;k++)
      {
         Dwhat_dble(-nu[k],rsd[k],psi);
         mulg5_dble(VOLUME_TRD/2,2,psi);
         mulr_spinor_add_dble(VOLUME_TRD/2,2,chi,psi,rnu[k]);

         act1=norm_square_dble(VOLUME_TRD/2,2,psi);
         scl_qflt(-2.0*nu[k]*rnu[k],act1.q);
         add_qflt(act0.q,act1.q,act0.q);

#if (defined FORCE_DBG)
         check_flds0(3,nu[k],sp,phi,psi);
#endif
      }

      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      wsd=reserve_wsd(2);
      psi=wsd[0];
      chi=wsd[1];

      random_sd(VOLUME_TRD/2,2,psi,1.0);
      bnd_sd2zero(EVEN_PTS,psi);
      assign_sd2sd(VOLUME_TRD/2,2,psi,phi);
      set_sd2zero(VOLUME_TRD/2,2,chi);

      for (k=0;k<np;k++)
      {
         assign_sd2sd(VOLUME_TRD/2,2,phi,psi);
         mulg5_dble(VOLUME_TRD/2,2,psi);
         set_sd2zero(VOLUME_TRD/2,2,psi+(VOLUME/2));

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,nu[k],psi,psi,
                    ifail0,stat0);
            acc_std_status("sap_gcr",ifail0,stat0,0,status);

            if (ifail0[0]<0)
            {
               print_status("sap_gcr",ifail0,stat0);
               error_root(1,1,"setpf3 [force3.c]","SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,nu[k],psi,psi,
                         ifail0,stat0);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               error_root(1,1,"setpf3 [force3.c]","DFL_SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }

         mulr_spinor_add_dble(VOLUME_TRD/2,2,chi,psi,rnu[k]);

         act1=norm_square_dble(VOLUME_TRD/2,2,psi);
         scl_qflt(-2.0*nu[k]*rnu[k],act1.q);
         add_qflt(act0.q,act1.q,act0.q);

#if (defined FORCE_DBG)
         check_flds0(3,nu[k],sp,phi,psi);
#endif
      }

      avg_std_status(np,status);
   }
   else
      error(1,1,"setpf3 [force3.c]","Unknown solver");

   z.re=0.0;
   z.im=1.0;
   mulc_spinor_add_dble(VOLUME_TRD/2,2,phi,chi,z);

   act1=norm_square_dble(VOLUME_TRD/2,2,chi);
   act1.q[0]=-act1.q[0];
   act1.q[1]=-act1.q[1];
   add_qflt(act0.q,act1.q,act0.q);

   release_wsd();

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=act0.q;
      global_qsum(1,qsm,qsm);
   }

   return act0;
}


void rotpf3(int *irat,int ipf,int isp,double c1,double c2,int *status)
{
   int np,k;
   double *nu,*rnu;
   complex_dble z;
   spinor_dble *phi,*eta,**wsd;
   spinor_dble *psi,**rsd;
   mdflds_t *mdfs;
   ratfct_t rf;
   tm_parms_t tm;
   solver_parms_t sp;
   sap_parms_t sap;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];
   eta=wsd[0];
   random_sd(VOLUME_TRD/2,2,eta,1.0);
   bnd_sd2zero(EVEN_PTS,eta);
   combine_spinor_dble(VOLUME_TRD/2,2,phi,eta,c1,c2);

   rf=ratfct(irat);
   np=rf.np;
   nu=rf.nu;
   rnu=rf.rnu;

   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(3,np,nu,sp);
#endif

   if (sp.solver==MSCG)
   {
      rsd=reserve_wsd(np);
      set_res(np,sp.res);
      tmcgm(sp.nmx,sp.istop,rs,np,nu,eta,rsd,ifail0,stat0);
      acc_std_status("tmcgm",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgm",ifail0,stat0);
         error_root(1,1,"rotpf3 [force3.c]","MSCG solver failed "
                    "(irat=%d,%d,%d, parameter set no %d)",
                    irat[0],irat[1],irat[2],isp);
      }

#if (defined FORCE_DBG)
      for (k=0;k<np;k++)
         check_flds2(2,nu[k],sp,eta,rsd[k]);
#endif

      for (k=0;k<np;k++)
      {
         Dwhat_dble(-nu[k],rsd[k],eta);
         mulg5_dble(VOLUME_TRD/2,2,eta);
         z.re=0.0;
         z.im=c2*rnu[k];
         mulc_spinor_add_dble(VOLUME_TRD/2,2,phi,eta,z);
      }

      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      rsd=reserve_wsd(1);
      psi=rsd[0];
      mulg5_dble(VOLUME_TRD/2,2,eta);
      set_sd2zero(VOLUME_TRD/2,2,eta+(VOLUME/2));

      for (k=0;k<np;k++)
      {
         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,nu[k],eta,psi,
                    ifail0,stat0);
            acc_std_status("sap_gcr",ifail0,stat0,0,status);

            if (ifail0[0]<0)
            {
               print_status("sap_gcr",ifail0,stat0);
               error_root(1,1,"rotpf3 [force3.c]","SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,nu[k],eta,psi,
                         ifail0,stat0);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               error_root(1,1,"rotpf3 [force3.c]","DFL_SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }

         z.re=0.0;
         z.im=c2*rnu[k];
         mulc_spinor_add_dble(VOLUME_TRD/2,2,phi,psi,z);

#if (defined FORCE_DBG)
         check_flds0(2,nu[k],sp,eta,psi);
#endif
      }

      avg_std_status(np,status);
      release_wsd();
   }
   else
      error(1,1,"rotpf3 [force3.c]","Unknown solver");

   release_wsd();
}


void force3(int *irat,int ipf,int isw,int isp,double c,int *status)
{
   int np,k;
   double *mu,*rmu;
   spinor_dble *phi,*chi,**wsd;
   mdflds_t *mdfs;
   ratfct_t rf;
   tm_parms_t tm;
   solver_parms_t sp;
   sap_parms_t sap;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   rf=ratfct(irat);
   np=rf.np;
   mu=rf.mu;
   rmu=rf.rmu;

   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(1,np,mu,sp);
#endif

   set_xt2zero();
   set_xv2zero();

   if (isw==1)
   {
      ifail0[0]=add_det2xt(1.0,ODD_PTS);
      error_root(ifail0[0]!=0,1,"force3 [force3.c]",
                 "Inversion of the SW term was not safe");
   }

   if (sp.solver==MSCG)
   {
      wsd=reserve_wsd(np+1);
      chi=wsd[np];

      set_res(np,sp.res);
      assign_sd2sd(VOLUME_TRD/2,2,phi,chi);

      tmcgm(sp.nmx,sp.istop,rs,np,mu,chi,wsd,ifail0,stat0);
      acc_std_status("tmcgm",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcgm",ifail0,stat0);
         error_root(1,1,"force3 [force3.c]","MSCG solver failed "
                    "(irat=%d,%d,%d, parameter set no %d)",
                    irat[0],irat[1],irat[2],isp);
      }

      for (k=0;k<np;k++)
      {
         Dwoe_dble(wsd[k],wsd[k]);
         Dwoo_dble(0.0,wsd[k],wsd[k]);

         Dwhat_dble(-mu[k],wsd[k],chi);
         mulg5_dble(VOLUME_TRD/2,2,chi);
         Dwoe_dble(chi,chi);
         Dwoo_dble(0.0,chi,chi);

         add_prod2xt(rmu[k],wsd[k],chi);
         add_prod2xv(-rmu[k],wsd[k],chi);

#if (defined FORCE_DBG)
         check_flds1(1,mu[k],sp,phi,chi,wsd[k]);
#endif
      }

      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      wsd=reserve_wsd(3);
      chi=wsd[2];
      assign_sd2sd(VOLUME_TRD/2,2,phi,chi);
      mulg5_dble(VOLUME_TRD/2,2,chi);
      set_sd2zero(VOLUME_TRD/2,2,chi+(VOLUME/2));

      for (k=0;k<np;k++)
      {
         if (sp.solver==SAP_GCR)
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],chi,wsd[0],
                    ifail0,stat0);
         else
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],chi,wsd[0],
                         ifail0,stat0);

         assign_sd2sd(VOLUME_TRD/2,2,wsd[0],wsd[1]);
         mulg5_dble(VOLUME_TRD/2,2,wsd[1]);
         set_sd2zero(VOLUME_TRD/2,2,wsd[1]+(VOLUME/2));

         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,-mu[k],wsd[1],
                    wsd[1],ifail1,stat1);
            acc_std_status("sap_gcr",ifail0,stat0,0,status);
            acc_std_status("sap_gcr",ifail1,stat1,1,status);

            if ((ifail0[0]<0)||(ifail1[0]<0))
            {
               print_status("sap_gcr",ifail0,stat0);
               print_status("sap_gcr",ifail1,stat1);
               error_root(1,1,"force3 [force3.c]","SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,-mu[k],wsd[1],
                         wsd[1],ifail1,stat1);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);
            acc_std_status("dfl_sap_gcr2",ifail1,stat1,1,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0)||(ifail1[0]<-2)||(ifail1[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               print_status("dfl_sap_gcr2",ifail1,stat1);
               error_root(1,1,"force3 [force3.c]","DFL_SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }

         add_prod2xt(rmu[k],wsd[1],wsd[0]);
         add_prod2xv(rmu[k],wsd[1],wsd[0]);

#if (defined FORCE_DBG)
         check_flds1(1,mu[k],sp,phi,wsd[0],wsd[1]);
#endif
      }

      avg_std_status(np,status);
      release_wsd();
   }
   else
      error(1,1,"force3 [force3.c]","Unknown solver");

   sw_frc(c);
   hop_frc(c);
}


qflt action3(int *irat,int ipf,int isw,int isp,int icom,int *status)
{
   int np,k;
   double *mu,*rmu,*qsm[1];
   qflt act0,act1;
   spinor_dble *phi,*chi,*psi,**wsd;
   mdflds_t *mdfs;
   ratfct_t rf;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   init_stat(status);
   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   mdfs=mdflds();
   phi=(*mdfs).pf[ipf];

   rf=ratfct(irat);
   np=rf.np;
   mu=rf.mu;
   rmu=rf.rmu;

   if (isw==1)
      act0=small_det(0);
   else
   {
      act0.q[0]=0.0;
      act0.q[1]=0.0;
   }

   sp=solver_parms(isp);

#if (defined FORCE_DBG)
   solver_info(0,np,mu,sp);
#endif

   if (sp.solver==MSCG)
   {
      wsd=reserve_wsd(np+1);
      chi=wsd[np];

      set_res(np,sp.res);
      assign_sd2sd(VOLUME_TRD/2,2,phi,chi);

      tmcgm(sp.nmx,sp.istop,rs,np,mu,chi,wsd,ifail0,stat0);
      acc_std_status("tmcgm",ifail0,stat0,0,status);

      if (ifail0[0]<0)
      {
         print_status("tmcg",ifail0,stat0);
         error_root(1,1,"action3 [force3.c]","MSCG solver failed "
                    "(irat=%d,%d,%d, parameter set no %d)",
                    irat[0],irat[1],irat[2],isp);
      }

#if (defined FORCE_DBG)
      for (k=0;k<np;k++)
         check_flds2(0,mu[k],sp,chi,wsd[k]);
#endif

      for (k=0;k<np;k++)
      {
         Dwhat_dble(-mu[k],wsd[k],chi);
         act1=norm_square_dble(VOLUME_TRD/2,2,chi);
         scl_qflt(rmu[k],act1.q);
         add_qflt(act0.q,act1.q,act0.q);
      }

      release_wsd();
   }
   else if ((sp.solver==SAP_GCR)||(sp.solver==DFL_SAP_GCR))
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      wsd=reserve_wsd(2);
      chi=wsd[0];
      psi=wsd[1];

      assign_sd2sd(VOLUME_TRD/2,2,phi,chi);
      mulg5_dble(VOLUME_TRD/2,2,chi);
      set_sd2zero(VOLUME_TRD/2,2,chi+(VOLUME/2));

      for (k=0;k<np;k++)
      {
         if (sp.solver==SAP_GCR)
         {
            sap_gcr(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],chi,psi,
                    ifail0,stat0);
            acc_std_status("sap_gcr",ifail0,stat0,0,status);

            if (ifail0[0]<0)
            {
               print_status("sap_gcr",ifail0,stat0);
               error_root(1,1,"action3 [force3.c]","SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }
         else
         {
            dfl_sap_gcr2(sp.nkv,sp.nmx,sp.istop,sp.res,mu[k],chi,psi,
                         ifail0,stat0);
            acc_std_status("dfl_sap_gcr2",ifail0,stat0,0,status);

            if ((ifail0[0]<-2)||(ifail0[1]<0))
            {
               print_status("dfl_sap_gcr2",ifail0,stat0);
               error_root(1,1,"action3 [force3.c]","DFL_SAP_GCR solver failed "
                          "(irat=%d,%d,%d, parameter set no %d)",
                          irat[0],irat[1],irat[2],isp);
            }
         }

         act1=norm_square_dble(VOLUME_TRD/2,2,psi);
         scl_qflt(rmu[k],act1.q);
         add_qflt(act0.q,act1.q,act0.q);

#if (defined FORCE_DBG)
         check_flds0(0,mu[k],sp,chi,psi);
#endif
      }

      avg_std_status(np,status);
      release_wsd();
   }
   else
      error(1,1,"action3 [force3.c]","Unknown solver");

   if ((NPROC>1)&&(icom&0x1))
   {
      qsm[0]=act0.q;
      global_qsum(1,qsm,qsm);
   }

   return act0;
}
