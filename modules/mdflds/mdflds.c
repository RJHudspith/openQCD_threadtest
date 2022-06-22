
/*******************************************************************************
*
* File mdflds.c
*
* Copyright (C) 2011-2018, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and initialization of the MD auxiliary fields.
*
*   mdflds_t *mdflds(void)
*     Returns the pointer to a mdflds_t structure containing the force,
*     momentum and pseudo-fermion fields. The fields are automatically
*     allocated if needed.
*
*   void set_frc2zero(void)
*     Sets all force variables, including those on the boundary, to zero.
*
*   void bnd_frc2zero(void)
*     Sets the force variables at the boundary of the local lattice to
*     zero.
*
*   void bnd_mom2zero(void)
*     Sets the components of the momentum field on the static links to
*     zero (see the notes).
*
*   void random_mom(void)
*     Sets the elements X of the momentum field on the active links to
*     random values with distribution proportional to exp(tr{X^2}). On
*     the static links the field is set to zero (see the notes).
*
*   void rotate_mom(double c1,double c2)
*     Replaces the momentum field mom by c1*mom+c2*eta, where eta is a
*     random field as generated by random_mom(). On the static links the
*     momentum field is set to zero (see the notes).
*
*   qflt momentum_action(int icom)
*     Returns the action of the momentum field. The action is summed
*     over all MPI processes if (and only if) icom%2=1.
*
* The quadruple-precision type qflt is defined in su3.h. See doc/qsum.pdf
* for further explanations.
*
* The arrays *.mom and *.frc in the structure returned by mdflds() are the
* molecular-dynamics momentum and force fields. Their elements are ordered
* in the same way as the link variables (see main/README.global). Moreover,
* the force field includes space for 7*(BNDRY/4) additional links as do the
* gauge fields (see lattice/README.uidx). The subsets of static and active
* links depend on the chosen boundary conditions. Only the field variables
* on the active links are updated in the simulations.
*
* The number npf of pseudo-fermion fields is retrieved from the parameter
* data base (see flags/hmc_parms.c and flags/smd_parms.c). Only those fields
* are allocated that actually occur in the chosen pseudo-fermion actions.
* Depending on the actions, the associated pseudo-fermion fields reside on
* all or only the even sites of the lattice and the field arrays thus have
* either VOLUME or VOLUME/2 elements.
*
* The elements of the structure returned by mdflds() are:
*
*  npf                Number of pseudo-fermion fields.
*
*  eo                 Array of flags eo[ipf], ipf=0,..,npf-1, indicating
*                     whether the pseudo-fermion field number ipf resides
*                     on all (eo[ipf]=0) or the even (eo[ipf]=1) sites.
*
*  mom                Pointer to the momentum field.
*
*  frc                Pointer to the force field.
*
*  pf                 Array of pointers pf[ipf], ipf=0,..,npf-1, to the
*                     pseudo-fermion fields (pf[ipf]=NULL if the field
*                     with index ipf is unused).
*
* The programs in this module assume that the simulation parameters have
* been entered to the parameter data base and that the geometry arrays are
* set. They perform global operations and are assumed to be called by the
* OpenMP master thread on all MPI processes simultaneously.
*
*******************************************************************************/

#define MDFLDS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "mdflds.h"
#include "global.h"

static int (*ofs_lks)[2],(*ofs_odd_pts)[2];
static int (*ofs_frc)[2],(*ofs_bnd)[2];
static mdflds_t *mdfs=NULL;


static void set_ofs(void)
{
   int k,nlks,npts,nfrc;
   int *a,*b;

   error(ipt==NULL,1,"set_ofs [mdflds.c]",
         "The geometry arrays are not set");

   ofs_lks=malloc(4*NTHREAD*sizeof(*ofs_lks));
   a=malloc(2*NTHREAD*sizeof(*a));
   error((ofs_lks==NULL)||(a==NULL),1,"set_ofs [mdflds.c]",
         "Unable to allocate offset arrays");

   ofs_odd_pts=ofs_lks+NTHREAD;
   ofs_frc=ofs_odd_pts+NTHREAD;
   ofs_bnd=ofs_frc+NTHREAD;
   b=a+NTHREAD;

   (void)(bnd_lks(&nlks));
   divide_range(nlks,NTHREAD,a,b);

   for (k=0;k<NTHREAD;k++)
   {
      ofs_lks[k][0]=a[k];
      ofs_lks[k][1]=b[k]-a[k];
   }

   (void)(bnd_pts(&npts));
   divide_range(npts/2,NTHREAD,a,b);

   for (k=0;k<NTHREAD;k++)
   {
      ofs_odd_pts[k][0]=a[k]+npts/2;
      ofs_odd_pts[k][1]=b[k]-a[k];
   }

   nfrc=4*VOLUME+7*(BNDRY/4);
   divide_range(nfrc,NTHREAD,a,b);

   for (k=0;k<NTHREAD;k++)
   {
      ofs_frc[k][0]=a[k];
      ofs_frc[k][1]=b[k]-a[k];
   }

   nfrc=7*(BNDRY/4);
   divide_range(nfrc,NTHREAD,a,b);

   for (k=0;k<NTHREAD;k++)
   {
      ofs_bnd[k][0]=a[k]+4*VOLUME;
      ofs_bnd[k][1]=b[k]-a[k];
   }

   free(a);
}


static void alloc_mdflds(void)
{
   int npf,nact,*iact,*eo;
   int ipf,vol,i;
   su3_alg_dble *mom;
   spinor_dble **pf,*phi;
   hmc_parms_t hmc;
   smd_parms_t smd;
   action_parms_t ap;

   set_ofs();

   mdfs=malloc(sizeof(*mdfs));
   mom=amalloc((8*VOLUME+7*(BNDRY/4))*sizeof(*mom),ALIGN);
   error((mdfs==NULL)||(mom==NULL),1,"alloc_mdflds [mdflds.c]",
         "Unable to allocate momentum and force fields");

   (*mdfs).mom=mom;
   (*mdfs).frc=mom+4*VOLUME;

   set_alg2zero(4*VOLUME_TRD,2,mom);
   set_frc2zero();

   hmc=hmc_parms();
   smd=smd_parms();

   if (hmc.nlv)
   {
      npf=hmc.npf;
      nact=hmc.nact;
      iact=hmc.iact;
   }
   else if (smd.nlv)
   {
      npf=smd.npf;
      nact=smd.nact;
      iact=smd.iact;
   }
   else
   {
      npf=0;
      nact=0;
      iact=NULL;
   }

   if (npf>0)
   {
      eo=malloc(npf*sizeof(*eo));
      pf=malloc(npf*sizeof(*pf));
      error((eo==NULL)||(pf==NULL),1,"alloc_mdflds [mdflds.c]",
            "Unable to allocate pseudo-fermion fields");

      for (ipf=0;ipf<npf;ipf++)
      {
         eo[ipf]=0;
         pf[ipf]=NULL;
      }

      for (i=0;i<nact;i++)
      {
         ap=action_parms(iact[i]);

         if (ap.action!=ACG)
         {
            ipf=ap.ipf;

            if ((ipf>=0)&&(ipf<npf))
            {
               if (pf[ipf]==NULL)
               {
                  if ((ap.action!=ACF_TM1)&&(ap.action!=ACF_TM2))
                  {
                     eo[ipf]=1;
                     phi=amalloc((VOLUME/2)*sizeof(*phi),ALIGN);
                     vol=VOLUME_TRD/2;
                  }
                  else
                  {
                     phi=amalloc(VOLUME*sizeof(*phi),ALIGN);
                     vol=VOLUME_TRD;
                  }

                  error(phi==NULL,1,"alloc_mdflds [mdflds.c]",
                        "Unable to allocate pseudo-fermion field");
                  pf[ipf]=phi;
                  set_sd2zero(vol,2,phi);
               }
            }
            else
               error(1,1,"alloc_mdflds [mdflds.c]",
                     "Pseudo-fermion index is out of range");
         }
      }

      (*mdfs).npf=npf;
      (*mdfs).eo=eo;
      (*mdfs).pf=pf;
   }
   else
   {
      (*mdfs).npf=0;
      (*mdfs).eo=NULL;
      (*mdfs).pf=NULL;
   }
}


mdflds_t *mdflds(void)
{
   if (mdfs==NULL)
      alloc_mdflds();

   return mdfs;
}


void set_frc2zero(void)
{
   int k;
   su3_alg_dble *frc;

   if (mdfs==NULL)
      alloc_mdflds();
   else
   {
      frc=(*mdfs).frc;

#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();

         set_alg2zero(ofs_frc[k][1],0,frc+ofs_frc[k][0]);
      }
   }
}


void bnd_frc2zero(void)
{
   int k;
   su3_alg_dble *frc;

   if (mdfs==NULL)
      alloc_mdflds();
   else
   {
      frc=(*mdfs).frc;

#pragma omp parallel private(k)
      {
         k=omp_get_thread_num();

         set_alg2zero(ofs_bnd[k][1],0,frc+ofs_bnd[k][0]);
      }
   }
}


void bnd_mom2zero(void)
{
   int bc,nlks,*lks,npts,*pts;
   int k,ifc,*lk,*lm,*pt,*pm;
   su3_alg_dble md0,*mom,*m;

   bc=bc_type();

   if ((bc==0)||(bc==1))
   {
      if (mdfs==NULL)
         alloc_mdflds();

      mom=(*mdfs).mom;
      set_alg2zero(1,0,&md0);

      if (bc==0)
      {
         lks=bnd_lks(&nlks);

         if (nlks>0)
         {
#pragma omp parallel private(k,lk,lm)
            {
               k=omp_get_thread_num();

               lk=lks+ofs_lks[k][0];
               lm=lk+ofs_lks[k][1];

               for (;lk<lm;lk++)
                  mom[*lk]=md0;
            }
         }
      }
      else if (bc==1)
      {
         pts=bnd_pts(&npts);

         if (npts>0)
         {
#pragma omp parallel private(k,ifc,pt,pm,m)
            {
               k=omp_get_thread_num();

               pt=pts+ofs_odd_pts[k][0];
               pm=pt+ofs_odd_pts[k][1];

               for (;pt<pm;pt++)
               {
                  m=mom+8*(pt[0]-(VOLUME/2));

                  for (ifc=2;ifc<8;ifc++)
                     m[ifc]=md0;
               }
            }
         }
      }
   }
}


void random_mom(void)
{
   if (mdfs==NULL)
      alloc_mdflds();

   random_alg(4*VOLUME_TRD,2,(*mdfs).mom);
   bnd_mom2zero();
}


void rotate_mom(double c1,double c2)
{
   int k;
   su3_alg_dble X,*m,*mm;

   if (mdfs==NULL)
      alloc_mdflds();

#pragma omp parallel private(k,X,m,mm)
   {
      k=omp_get_thread_num();

      m=(*mdfs).mom+k*4*VOLUME_TRD;
      mm=m+4*VOLUME_TRD;

      for (;m<mm;m++)
      {
         random_alg(1,0,&X);

         (*m).c1=c1*(*m).c1+c2*X.c1;
         (*m).c2=c1*(*m).c2+c2*X.c2;
         (*m).c3=c1*(*m).c3+c2*X.c3;
         (*m).c4=c1*(*m).c4+c2*X.c4;
         (*m).c5=c1*(*m).c5+c2*X.c5;
         (*m).c6=c1*(*m).c6+c2*X.c6;
         (*m).c7=c1*(*m).c7+c2*X.c7;
         (*m).c8=c1*(*m).c8+c2*X.c8;
      }
   }

   bnd_mom2zero();
}


qflt momentum_action(int icom)
{
   qflt rqsm;

   if (mdfs==NULL)
      alloc_mdflds();

   if (icom&0x1)
      rqsm=norm_square_alg(4*VOLUME_TRD,3,(*mdfs).mom);
   else
      rqsm=norm_square_alg(4*VOLUME_TRD,2,(*mdfs).mom);

   rqsm.q[0]*=0.5;
   rqsm.q[1]*=0.5;

   return rqsm;
}
