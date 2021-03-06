
/*******************************************************************************
*
* File dfl_sap_gcr.c
*
* Copyright (C) 2007-2013, 2018, 2020, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* SAP+GCR solver for the Dirac equation with multilevel local deflation.
*
*   double dfl_sap_gcr(int nkv,int nmx,int istop,double res,double mu,
*                      spinor_dble *eta,spinor_dble *psi,
*                      int *ifail,int *status)
*     Obtains an approximate solution psi of the Dirac equation for given
*     source eta using the deflated SAP-preconditioned GCR algorithm.
*
*   double dfl_sap_gcr2(int nkv,int nmx,int istop,double res,double mu,
*                       spinor_dble *eta,spinor_dble *psi,
*                       int *ifail,int *status)
*     Calls dfl_sap_gcr(nkv,..,status) and, if ifail=-1 or -2, regenerates
*     the deflation subspace and calls dfl_sap_gcr() a second time.
*
* Depending on whether the twisted-mass flag is set or not, the program
* solves the equation
*
*   (Dw+i*mu*gamma_5*1e)*psi=eta  or  (Dw+i*mu*gamma_5)*psi=eta,
*
* respectively, where 1e is 1 on the even and 0 on the odd lattice sites.
* The twisted-mass flag is retrieved from the parameter data base (see
* flags/lat_parms.c).
*
* The program dfl_sap_gcr() is based on the flexible GCR algorithm (see
* linsolv/fgcr.c). Before the solver is launched, the following parameter-
* setting programs must have been called:
*
*  set_lat_parms()        SW improvement coefficient.
*
*  set_bc_parms()         Boundary conditions and associated improvement
*                         coefficients.
*
*  set_sw_parms()         Bare quark mass.
*
*  set_sap_parms()        Parameters of the SAP preconditioner.
*
*  set_dfl_parms()        Parameters of the deflation subspace.
*
*  set_dfl_pro_parms()    Parameters used for the deflation projection.
*
*  set_dfl_gen_parms()    Subspace generation parameters.
*
* See doc/parms.pdf and the relevant files in the modules/flags directory
* for further explanations. The deflation subspace must have been properly
* initialized by the program dfl_subspace().
*
* All other parameters are passed through the argument list:
*
*  nkv       Maximal number of Krylov vectors generated before the GCR
*            algorithm is restarted.
*
*  nmx       Maximal total number of Krylov vectors that may be generated.
*
*  istop     Stopping criterion (0: L_2 norm based, 1: uniform norm based).
*
*  res       Desired maximal relative residue of the calculated solution.
*
*  mu        Value of the twisted mass in the Dirac equation.
*
*  eta       Source field. eta is unchanged on exit unless psi=eta (which
*            is permissible).
*
*  psi       Calculated approximate solution of the Dirac equation.
*
* In the case of the program dfl_sap_gcr(), the array ifail must have 1 or
* more elements. On exit
*
*  ifail[0]=0        The program completed successfully.
*
*  ifail[0]=-1       The solver did not converge.
*
*  ifail[0]=-2       The inversion of the diagonal parts of the little
*                    Dirac operator was not safe.
*
*  ifail[0]=-3       The inversion of the SW term on the odd sites of
*                    the lattice was not safe.
*
* The array status must have at least 3 elements. On exit
*
*  status[0]         Number of Krylov vectors generated by the fgcr()
*                    solver.
*
*  status[1]         Average number of Krylov vectors generated by the
*                    fgcr4vd() algorithm employed by the solver for the
*                    little Dirac equation (see ltl_gcr.c).
*
*  status[2]         Average number of Krylov vectors generated by the
*                    gcr4v() algorithm used as preconditioner in the
*                    solver for the little Dirac equation.
*
* In the case of the program dfl_sap_gcr2(), ifail and status must have at
* least 2 and 6 elements, respectively, the first half of them being filled
* by dfl_sap_gcr() when it is first called. If ifail[0]=0 or ifail[0]=-3 is
* returned, the unused array elements are set to zero and no further action
* is performed.
*
* If ifail[0]=-1 or ifail[0]=-2, the program calls dfl_modes() with output
* written to ifail[1] and status[4] and status[5]. The program terminates at
* this point if ifail[1]=-2. Otherwise dfl_sap_gcr() is called with output
* written to ifail[1] and status[3],..,status[5].
*
* The fields eta and psi must be such that the Dirac operator can act on
* them (see main/README.global). Moreover, the source eta is assumed to
* respect the chosen boundary conditions (see doc/dirac.pdf).
*
* The program dfl_sap_gcr() returns the norm of the residue of the calculated
* approximate solution psi if ifail[0]>=-1. Otherwise the program returns the
* norm of eta and sets psi to zero if psi!=eta.
*
* The program dfl_sap_gcr2() behaves in the same way depending on ifail[0]
* and ifail[1] (if dfl_sap_gcr() was called a second time).
*
* The SAP_BLOCKS blocks grid is automatically allocated if it is not already
* allocated, while the SW term is recalculated when needed and the gauge and
* SW fields are copied to the SAP block grid if they are not in the proper
* condition. Similarly, the little Dirac operator is updated when needed.
*
* The program dfl_sap_gcr2() can be used in place of dfl_sap_gcr() if
* some protection against the rare cases, where the little Dirac operator
* turns out to be accidentally ill-conditioned, is desired.
*
* The required workspaces are
*
*  spinor              2*nkv+2
*  spinor_dble         2
*  complex             2*dpr.nmx_gcr+3
*  complex_dble        2*dpr.nkv+4
*
* where dpr.nmx_gcr and dpr.nkv are parameters of the solver for the little
* Dirac equation (see utils/wspace.c and flags/dfl_parms.c).
*
* Some debugging output is printed to stdout on process 0 if FGCR_DBG is
* defined at compilation time.
*
* The programs in this module assumed to be called by the OpenMP master thread
* on all MPI processes simultaneously.
*
*******************************************************************************/

#define DFL_SAP_GCR_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "flags.h"
#include "block.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "linsolv.h"
#include "sap.h"
#include "vflds.h"
#include "little.h"
#include "dfl.h"
#include "global.h"

static int *ltl_status,init=0;
static float mus;
static double mud;
static sap_parms_t spr;
static dfl_pro_parms_t dpr;


static void init_once(void)
{
   int nb,isw;
   dfl_parms_t dfl;

   dfl=dfl_parms();
   error_root(dfl.Ns==0,1,"init_once [dfl_sap_gcr.c]",
              "Deflation parameters are not set");

   spr=sap_parms();
   error_root(spr.ncy==0,1,"init_once [dfl_sap_gcr.c]",
              "SAP parameters are not set");

   dpr=dfl_pro_parms();
   error_root(dpr.nkv==0,1,"init_once [dfl_sap_gcr.c]",
              "Deflation projection parameters are not set");

   blk_list(SAP_BLOCKS,&nb,&isw);
   error_root(nb==0,1,"init_once [dfl_sap_gcr.c]",
              "SAP block grid is not allocated");

   blk_list(DFL_BLOCKS,&nb,&isw);
   error_root(nb==0,1,"init_once [dfl_sap_gcr.c]",
              "DFL block grid is not allocated");

   init=1;
}


static void Dop(spinor_dble *sd,spinor_dble *rd)
{
   Dw_dble(mud,sd,rd);
}


static void Mop(int k,spinor *rho,spinor *phi,spinor *chi)
{
   int n,ifail,status[2];
   complex_dble **wvd;
   spinor **ws;

   wvd=reserve_wvd(1);
   ws=reserve_ws(1);

   dfl_s2vd(rho,wvd[0]);
   ltl_gcr(mud,wvd[0],wvd[0],&ifail,status);
   dfl_vd2s(wvd[0],ws[0]);

   Dw(mus,ws[0],chi);
   diff_s2s(VOLUME_TRD,2,rho,chi);
   set_s2zero(VOLUME_TRD,2,phi);

   for (n=0;n<spr.ncy;n++)
      sap(mus,spr.isolv,spr.nmr,phi,chi);

   diff_s2s(VOLUME_TRD,2,rho,chi);
   add_s2s(VOLUME_TRD,2,ws[0],phi);

   ltl_status[0]+=status[0];
   ltl_status[1]+=status[1];

   release_ws();
   release_wvd();
}


static double dfl_sap_gcr0(int nkv,int nmx,int istop,double res,double mu,
                           spinor_dble *eta,spinor_dble *psi,
                           int *ifail,int *status)
{
   int swde,swdo,swu,swe,swo;
   double rho,rho0;
   qflt qnrm;
   spinor **ws;
   spinor_dble **wsd;

   if (query_grid_flags(SAP_BLOCKS,UBGR_MATCH_UD)!=1)
      assign_ud2ubgr(SAP_BLOCKS);

   if (query_flags(SWD_UP2DATE)!=1)
      sw_term(NO_PTS);

   swde=query_flags(SWD_E_INVERTED);
   swdo=query_flags(SWD_O_INVERTED);

   swu=query_grid_flags(SAP_BLOCKS,SW_UP2DATE);
   swe=query_grid_flags(SAP_BLOCKS,SW_E_INVERTED);
   swo=query_grid_flags(SAP_BLOCKS,SW_O_INVERTED);
   spr=sap_parms();

   if (spr.isolv==0)
   {
      if ((swde==1)||(swdo==1))
         sw_term(NO_PTS);

      if ((swu!=1)||(swe==1)||(swo==1))
         assign_swd2swbgr(SAP_BLOCKS,NO_PTS);
   }
   else if (spr.isolv==1)
   {
      if ((swde!=1)&&(swdo==1))
      {
         if ((swu!=1)||(swe==1)||(swo!=1))
            assign_swd2swbgr(SAP_BLOCKS,NO_PTS);

         sw_term(NO_PTS);
      }
      else
      {
         if ((swde==1)||(swdo==1))
            sw_term(NO_PTS);

         if ((swu!=1)||(swe==1)||(swo!=1))
            ifail[0]=assign_swd2swbgr(SAP_BLOCKS,ODD_PTS);
      }
   }

   if (query_flags(U_MATCH_UD)!=1)
      assign_ud2u();

   if ((query_flags(SW_UP2DATE)!=1)||
       (query_flags(SW_E_INVERTED)==1)||(query_flags(SW_O_INVERTED)==1))
      assign_swd2sw();

   if (istop)
      rho=unorm_dble(VOLUME_TRD,3,eta);
   else
   {
      qnrm=norm_square_dble(VOLUME_TRD,3,eta);
      rho=sqrt(qnrm.q[0]);
   }

   if (ifail[0])
      ifail[0]=-3;
   else if (set_Awhat(mu))
      ifail[0]=-2;
   else if (rho!=0.0)
   {
      ws=reserve_ws(2*nkv+1);
      wsd=reserve_wsd(1);

      ltl_status=status+1;
      mus=(float)(mu);
      mud=mu;
      rho0=rho;
      scale_dble(VOLUME_TRD,2,1.0/rho0,eta);

      rho=fgcr(VOLUME_TRD,Dop,Mop,ws,wsd,nkv,nmx,istop,res,
               eta,psi,status);

      scale_dble(VOLUME_TRD,2,rho0,psi);
      rho*=rho0;

      if (status[0]<0)
      {
         ifail[0]=-1;
         status[0]=nmx;
         scale_dble(VOLUME_TRD,2,rho0,eta);
      }

      if (status[0]>0)
      {
         status[1]=(status[1]+(status[0]/2))/status[0];
         status[2]=(status[2]+(status[0]/2))/status[0];
      }

      release_wsd();
      release_ws();
   }
   else
      set_sd2zero(VOLUME_TRD,2,psi);

   return rho;
}


double dfl_sap_gcr(int nkv,int nmx,int istop,double res,double mu,
                   spinor_dble *eta,spinor_dble *psi,int *ifail,int *status)
{
   int l;
   double rho;
   spinor_dble *rsd,**wsd;

   if (init==0)
      init_once();

   ifail[0]=0;

   for (l=0;l<3;l++)
      status[l]=0;

   wsd=reserve_wsd(1);
   rsd=wsd[0];
   assign_sd2sd(VOLUME_TRD,2,eta,rsd);

   rho=dfl_sap_gcr0(nkv,nmx,istop,res,mu,rsd,psi,ifail,status);

   if ((psi!=eta)&&(ifail[0]<=-2))
      set_sd2zero(VOLUME_TRD,2,psi);

   release_wsd();

   return rho;
}


double dfl_sap_gcr2(int nkv,int nmx,int istop,double res,double mu,
                    spinor_dble *eta,spinor_dble *psi,int *ifail,int *status)
{
   int l;
   double rho;
   spinor_dble *rsd,**wsd;

   if (init==0)
      init_once();

   ifail[0]=0;
   ifail[1]=0;

   for (l=0;l<6;l++)
      status[l]=0;

   wsd=reserve_wsd(1);
   rsd=wsd[0];
   assign_sd2sd(VOLUME_TRD,2,eta,rsd);

   rho=dfl_sap_gcr0(nkv,nmx,istop,res,mu,rsd,psi,ifail,status);

   if ((psi!=eta)&&(ifail[0]<-2))
      set_sd2zero(VOLUME_TRD,2,psi);

   if ((ifail[0]==-1)||(ifail[0]==-2))
   {
      dfl_modes(ifail+1,status+4);

      if (ifail[1]==0)
      {
         rho=dfl_sap_gcr0(nkv,nmx,istop,res,mu,rsd,psi,ifail+1,status+3);

         if ((psi!=eta)&&(ifail[1]<=-2))
            set_sd2zero(VOLUME_TRD,2,psi);
      }
   }

   release_wsd();

   return rho;
}
