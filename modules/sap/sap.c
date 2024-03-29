
/*******************************************************************************
*
* File sap.c
*
* Copyright (C) 2005, 2011, 2012, 2013, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Multiplicative alternating Schwarz procedure for the solution of the
* Wilson-Dirac equation.
*
*   void sap(float mu,int isolv,int nmr,spinor *psi,spinor *rho)
*     Application of one cycle of the multiplicative Schwarz procedure to
*     the approximate solution psi of the Wilson-Dirac equation, assuming
*     the associated residue is stored in the field rho (see the notes). The
*     block Dirac equation is solved using nmr iterations of the ordinary
*     (isolv=0) or the even-odd preconditioned (isolv=1) minimal residual
*     algorithm. On exit, the new approximate solution and its residue are
*     returned in the fields psi and rho.
*
* Depending on whether the twisted-mass flag is set or not, the program
* solves the equation
*
*   (Dw+i*mu*gamma_5*1e)*psi=eta  or  (Dw+i*mu*gamma_5)*psi=eta,
*
* the residues of the calculated solution being
*
*   rho=eta-(Dw+i*mu*gamma5*1e)*psi or rho=eta-(Dw+i*mu*gamma5)*psi
*
* respectively. The twisted-mass flag is retrieved from the parameter data
* base (see flags/lat_parms.c).
*
* The program acts on the SAP_BLOCKS block grid. It is taken for granted
* that the grid is allocated and that the single-precision gauge and SW
* field on the grid are in the proper condition when sap() is called. In
* particular, the SW term must not be inverted if isolv=0, but should be
* inverted on the odd sites if isolv=1.
*
* The programs sap() is assumed to be called by the OpenMP master thread on
* all MPI processes simultaneously.
*
* If SSE (AVX) instructions are used, the Dirac spinors must be aligned to
* a 16 (32) byte boundary.
*
*******************************************************************************/

#define SAP_C

#define JVER_SAP

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "su3.h"
#include "flags.h"
#include "block.h"
#include "dirac.h"
#include "sap.h"
#include "global.h"

#if (defined AVX)
#include "avx.h"


static void update_flds0(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sln;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sln=psi+imb[0];

   for (;sb<sm;sb++)
   {
      sl=sln;
      sln=psi+imb[1];
      _prefetch_spinor(sln);

      _avx_spinor_load(*sl);
      _avx_spinor_load_up(*sb);
      _avx_spinor_add();
      _avx_spinor_store(*sl);

      rl=rho+imb[0];
      imb+=1;
      _avx_spinor_load_up(*rb);
      _avx_spinor_store_up(*rl);
      rb+=1;
   }

   _avx_zeroupper();
}


static void update_flds1(int vol,int *imb,float mu,spinor **s,spinor *psi,
                         spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sln;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sln=psi+imb[0];

   for (;sb<sm;sb++)
   {
      sl=sln;
      sln=psi+imb[1];
      _prefetch_spinor(sln);

      _avx_spinor_load(*sl);
      _avx_spinor_load_up(*sb);
      _avx_spinor_add();
      _avx_spinor_store(*sl);

      rl=rho+imb[0];
      imb+=1;
      _avx_spinor_load_up(*rb);
      _avx_spinor_store_up(*rl);
      rb+=1;
   }

   __asm__ __volatile__ ("vxorps %%ymm8, %%ymm8, %%ymm8 \n\t"
                         "vbroadcastss %0, %%ymm7 \n\t"
                         "vaddsubps %%ymm7, %%ymm8, %%ymm8 \n\t"
                         "vpermilps $0xb1, %%ymm8, %%ymm6 \n\t"
                         "vblendps $0xf, %%ymm6, %%ymm8, %%ymm7"
                         :
                         :
                         "m" (mu)
                         :
                         "xmm6", "xmm7", "xmm8");

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=sln;
      sln=psi+imb[1];
      _prefetch_spinor(sln);

      _avx_spinor_load(*sb);
      _avx_spinor_load_up(*rb);

      __asm__ __volatile__ ("vsubps %%ymm0, %%ymm3, %%ymm3 \n\t"
                            "vsubps %%ymm1, %%ymm4, %%ymm4 \n\t"
                            "vsubps %%ymm2, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm3", "xmm4", "xmm5");

      _avx_spinor_load(*sl);
      _avx_spinor_add();
      _avx_spinor_store_up(*sb);
      _avx_spinor_store(*sl);

      __asm__ __volatile__ ("vpermilps $0xb1, %%ymm3, %%ymm3 \n\t"
                            "vpermilps $0xb1, %%ymm4, %%ymm4 \n\t"
                            "vpermilps $0xb1, %%ymm5, %%ymm5 \n\t"
                            "vmulps %%ymm6, %%ymm3, %%ymm3 \n\t"
                            "vmulps %%ymm7, %%ymm4, %%ymm4 \n\t"
                            "vmulps %%ymm8, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm3", "xmm4", "xmm5");

      rl=rho+imb[0];
      imb+=1;
      _avx_spinor_store_up(*rl);
      rb+=1;
   }

   _avx_zeroupper();
}


static void update_flds2(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sln;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sln=psi+imb[0];

   for (;sb<sm;sb++)
   {
      sl=sln;
      sln=psi+imb[1];
      _prefetch_spinor(sln);

      _avx_spinor_load(*sl);
      _avx_spinor_load_up(*sb);
      _avx_spinor_add();
      _avx_spinor_store(*sl);

      rl=rho+imb[0];
      imb+=1;
      _avx_spinor_load_up(*rb);
      _avx_spinor_store_up(*rl);
      rb+=1;
   }

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=sln;
      sln=psi+imb[1];
      _prefetch_spinor(sln);

      _avx_spinor_load(*sb);
      _avx_spinor_load_up(*rb);

      __asm__ __volatile__ ("vsubps %%ymm0, %%ymm3, %%ymm3 \n\t"
                            "vsubps %%ymm1, %%ymm4, %%ymm4 \n\t"
                            "vsubps %%ymm2, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm3", "xmm4", "xmm5");

      _avx_spinor_load(*sl);
      _avx_spinor_add();
      _avx_spinor_store_up(*sb);
      _avx_spinor_store(*sl);

      __asm__ __volatile__ ("vxorps %%ymm3, %%ymm3, %%ymm3 \n\t"
                            "vxorps %%ymm4, %%ymm4, %%ymm4 \n\t"
                            "vxorps %%ymm5, %%ymm5, %%ymm5"
                            :
                            :
                            :
                            "xmm3", "xmm4", "xmm5");

      rl=rho+imb[0];
      imb+=1;
      _avx_spinor_store_up(*rl);
      rb+=1;
   }

   _avx_zeroupper();
}

#elif (defined x64)
#include "sse2.h"

static void update_flds0(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sls;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sls=psi+(*imb);

   for (;sb<sm;sb++)
   {
      sl=sls;
      rl=rho+(*imb);
      imb+=1;
      sls=psi+(*imb);

      _prefetch_spinor(sls);
      _sse_spinor_load(*sb);

      __asm__ __volatile__ ("addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*sl).c1.c1),
                            "m" ((*sl).c1.c2),
                            "m" ((*sl).c1.c3),
                            "m" ((*sl).c2.c1),
                            "m" ((*sl).c2.c2),
                            "m" ((*sl).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*sl).c3.c1),
                            "m" ((*sl).c3.c2),
                            "m" ((*sl).c3.c3),
                            "m" ((*sl).c4.c1),
                            "m" ((*sl).c4.c2),
                            "m" ((*sl).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*sl);
      _sse_spinor_load_up(*rb);
      _sse_spinor_store_up(*rl);
      rb+=1;
   }
}


static void update_flds1(int vol,int *imb,float mu,spinor **s,spinor *psi,
                         spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sls;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sls=psi+(*imb);

   for (;sb<sm;sb++)
   {
      sl=sls;
      rl=rho+(*imb);
      imb+=1;
      sls=psi+(*imb);

      _prefetch_spinor(sls);
      _sse_spinor_load(*sb);

      __asm__ __volatile__ ("addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*sl).c1.c1),
                            "m" ((*sl).c1.c2),
                            "m" ((*sl).c1.c3),
                            "m" ((*sl).c2.c1),
                            "m" ((*sl).c2.c2),
                            "m" ((*sl).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*sl).c3.c1),
                            "m" ((*sl).c3.c2),
                            "m" ((*sl).c3.c3),
                            "m" ((*sl).c4.c1),
                            "m" ((*sl).c4.c2),
                            "m" ((*sl).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*sl);
      _sse_spinor_load_up(*rb);
      _sse_spinor_store_up(*rl);
      rb+=1;
   }

   __asm__ __volatile__ ("movss %0, %%xmm12 \n\t"
                         "shufps $0x0, %%xmm12, %%xmm12 \n\t"
                         "mulps %1, %%xmm12 \n\t"
                         "movaps %%xmm12, %%xmm13 \n\t"
                         "movaps %%xmm12, %%xmm14"
                         :
                         :
                         "m" (mu),
                         "m" (_sse_sgn13)
                         :
                         "xmm12", "xmm13", "xmm14");

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=sls;
      rl=rho+(*imb);
      imb+=1;
      sls=psi+(*imb);

      _prefetch_spinor(sls);
      _sse_spinor_load(*rb);

      __asm__ __volatile__ ("subps %0, %%xmm0 \n\t"
                            "subps %2, %%xmm1 \n\t"
                            "subps %4, %%xmm2"
                            :
                            :
                            "m" ((*sb).c1.c1),
                            "m" ((*sb).c1.c2),
                            "m" ((*sb).c1.c3),
                            "m" ((*sb).c2.c1),
                            "m" ((*sb).c2.c2),
                            "m" ((*sb).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("subps %0, %%xmm3 \n\t"
                            "subps %2, %%xmm4 \n\t"
                            "subps %4, %%xmm5"
                            :
                            :
                            "m" ((*sb).c3.c1),
                            "m" ((*sb).c3.c2),
                            "m" ((*sb).c3.c3),
                            "m" ((*sb).c4.c1),
                            "m" ((*sb).c4.c2),
                            "m" ((*sb).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*sb);
      _sse_spinor_load_up(*sl);

      __asm__ __volatile__ ("addps %%xmm0, %%xmm6 \n\t"
                            "addps %%xmm1, %%xmm7 \n\t"
                            "addps %%xmm2, %%xmm8 \n\t"
                            "addps %%xmm3, %%xmm9 \n\t"
                            "addps %%xmm4, %%xmm10 \n\t"
                            "addps %%xmm5, %%xmm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11");

      _sse_spinor_store_up(*sl);

      __asm__ __volatile__ ("mulps %%xmm12, %%xmm0 \n\t"
                            "mulps %%xmm13, %%xmm1 \n\t"
                            "mulps %%xmm14, %%xmm2 \n\t"
                            "shufps $0xb1, %%xmm3, %%xmm3 \n\t"
                            "shufps $0xb1, %%xmm4, %%xmm4 \n\t"
                            "shufps $0xb1, %%xmm5, %%xmm5 \n\t"
                            "shufps $0xb1, %%xmm0, %%xmm0 \n\t"
                            "shufps $0xb1, %%xmm1, %%xmm1 \n\t"
                            "shufps $0xb1, %%xmm2, %%xmm2 \n\t"
                            "mulps %%xmm12, %%xmm3 \n\t"
                            "mulps %%xmm13, %%xmm4 \n\t"
                            "mulps %%xmm14, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      _sse_spinor_store(*rl);
      rb+=1;
   }
}


static void update_flds2(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl,*sls;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;
   sls=psi+(*imb);

   for (;sb<sm;sb++)
   {
      sl=sls;
      rl=rho+(*imb);
      imb+=1;
      sls=psi+(*imb);

      _prefetch_spinor(sls);
      _sse_spinor_load(*sb);

      __asm__ __volatile__ ("addps %0, %%xmm0 \n\t"
                            "addps %2, %%xmm1 \n\t"
                            "addps %4, %%xmm2"
                            :
                            :
                            "m" ((*sl).c1.c1),
                            "m" ((*sl).c1.c2),
                            "m" ((*sl).c1.c3),
                            "m" ((*sl).c2.c1),
                            "m" ((*sl).c2.c2),
                            "m" ((*sl).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("addps %0, %%xmm3 \n\t"
                            "addps %2, %%xmm4 \n\t"
                            "addps %4, %%xmm5"
                            :
                            :
                            "m" ((*sl).c3.c1),
                            "m" ((*sl).c3.c2),
                            "m" ((*sl).c3.c3),
                            "m" ((*sl).c4.c1),
                            "m" ((*sl).c4.c2),
                            "m" ((*sl).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*sl);
      _sse_spinor_load_up(*rb);
      _sse_spinor_store_up(*rl);
      rb+=1;
   }

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=sls;
      rl=rho+(*imb);
      imb+=1;
      sls=psi+(*imb);

      _prefetch_spinor(sls);
      _sse_spinor_load(*rb);

      __asm__ __volatile__ ("subps %0, %%xmm0 \n\t"
                            "subps %2, %%xmm1 \n\t"
                            "subps %4, %%xmm2"
                            :
                            :
                            "m" ((*sb).c1.c1),
                            "m" ((*sb).c1.c2),
                            "m" ((*sb).c1.c3),
                            "m" ((*sb).c2.c1),
                            "m" ((*sb).c2.c2),
                            "m" ((*sb).c2.c3)
                            :
                            "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__ ("subps %0, %%xmm3 \n\t"
                            "subps %2, %%xmm4 \n\t"
                            "subps %4, %%xmm5"
                            :
                            :
                            "m" ((*sb).c3.c1),
                            "m" ((*sb).c3.c2),
                            "m" ((*sb).c3.c3),
                            "m" ((*sb).c4.c1),
                            "m" ((*sb).c4.c2),
                            "m" ((*sb).c4.c3)
                            :
                            "xmm3", "xmm4", "xmm5");

      _sse_spinor_store(*sb);
      _sse_spinor_load_up(*sl);

      __asm__ __volatile__ ("addps %%xmm0, %%xmm6 \n\t"
                            "addps %%xmm1, %%xmm7 \n\t"
                            "addps %%xmm2, %%xmm8 \n\t"
                            "addps %%xmm3, %%xmm9 \n\t"
                            "addps %%xmm4, %%xmm10 \n\t"
                            "addps %%xmm5, %%xmm11"
                            :
                            :
                            :
                            "xmm6", "xmm7", "xmm8", "xmm9",
                            "xmm10", "xmm11");

      _sse_spinor_store_up(*sl);

      __asm__ __volatile__ ("xorpd %%xmm0, %%xmm0 \n\t"
                            "xorpd %%xmm1, %%xmm1 \n\t"
                            "xorpd %%xmm2, %%xmm2 \n\t"
                            "xorpd %%xmm3, %%xmm3 \n\t"
                            "xorpd %%xmm4, %%xmm4 \n\t"
                            "xorpd %%xmm5, %%xmm5"
                            :
                            :
                            :
                            "xmm0", "xmm1", "xmm2", "xmm3",
                            "xmm4", "xmm5");

      _sse_spinor_store(*rl);
      rb+=1;
   }
}

#else

static const spinor s0={{{0.0f}}};


static void update_flds0(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;

   for (;sb<sm;sb++)
   {
      sl=psi+(*imb);
      rl=rho+(*imb);
      imb+=1;

      _vector_add_assign((*sl).c1,(*sb).c1);
      _vector_add_assign((*sl).c2,(*sb).c2);
      _vector_add_assign((*sl).c3,(*sb).c3);
      _vector_add_assign((*sl).c4,(*sb).c4);

      (*rl)=(*rb);
      rb+=1;
   }
}


static void update_flds1(int vol,int *imb,float mu,spinor **s,spinor *psi,
                         spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;

   for (;sb<sm;sb++)
   {
      sl=psi+(*imb);
      rl=rho+(*imb);
      imb+=1;

      _vector_add_assign((*sl).c1,(*sb).c1);
      _vector_add_assign((*sl).c2,(*sb).c2);
      _vector_add_assign((*sl).c3,(*sb).c3);
      _vector_add_assign((*sl).c4,(*sb).c4);

      (*rl)=(*rb);
      rb+=1;
   }

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=psi+(*imb);
      rl=rho+(*imb);
      imb+=1;

      _vector_sub((*sb).c1,(*rb).c1,(*sb).c1);
      _vector_sub((*sb).c2,(*rb).c2,(*sb).c2);
      _vector_sub((*sb).c3,(*rb).c3,(*sb).c3);
      _vector_sub((*sb).c4,(*rb).c4,(*sb).c4);

      _vector_add_assign((*sl).c1,(*sb).c1);
      _vector_add_assign((*sl).c2,(*sb).c2);
      _vector_add_assign((*sl).c3,(*sb).c3);
      _vector_add_assign((*sl).c4,(*sb).c4);

      _vector_imul((*rl).c1,-mu,(*sb).c1);
      _vector_imul((*rl).c2,-mu,(*sb).c2);
      _vector_imul((*rl).c3, mu,(*sb).c3);
      _vector_imul((*rl).c4, mu,(*sb).c4);

      rb+=1;
   }
}


static void update_flds2(int vol,int *imb,spinor **s,spinor *psi,spinor *rho)
{
   spinor *sb,*rb,*sm;
   spinor *sl,*rl;

   sb=s[0];
   rb=s[1];
   sm=sb+vol;

   for (;sb<sm;sb++)
   {
      sl=psi+(*imb);
      rl=rho+(*imb);
      imb+=1;

      _vector_add_assign((*sl).c1,(*sb).c1);
      _vector_add_assign((*sl).c2,(*sb).c2);
      _vector_add_assign((*sl).c3,(*sb).c3);
      _vector_add_assign((*sl).c4,(*sb).c4);

      (*rl)=(*rb);
      rb+=1;
   }

   sm+=vol;

   for (;sb<sm;sb++)
   {
      sl=psi+(*imb);
      rl=rho+(*imb);
      imb+=1;

      _vector_sub((*sb).c1,(*rb).c1,(*sb).c1);
      _vector_sub((*sb).c2,(*rb).c2,(*sb).c2);
      _vector_sub((*sb).c3,(*rb).c3,(*sb).c3);
      _vector_sub((*sb).c4,(*rb).c4,(*sb).c4);

      _vector_add_assign((*sl).c1,(*sb).c1);
      _vector_add_assign((*sl).c2,(*sb).c2);
      _vector_add_assign((*sl).c3,(*sb).c3);
      _vector_add_assign((*sl).c4,(*sb).c4);

      (*rl)=s0;
      rb+=1;
   }
}

#endif

#ifdef JVER_SAP

static void
dosap( const int nbh , const block_t *b , const int nb , const int ic , const int isw ,
       const float mu , const int isolv , const int nmr , spinor *psi , spinor *rho )
{
  const tm_parms_t tm = tm_parms();
  const int eoflg = tm.eoflg;
  
  const int k = omp_get_thread_num();
  spinor **s = b[k].s;

  const int vol = isolv ? b->vol/2 : b->vol ;

  int n = ic^isw ? k+nbh : k ;
  const int nm = ic^isw ? nb : nbh ;

  if (isolv) {
    for (;n<nm;n+=NTHREAD) {
      assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,rho,1);
      Dwoo_blk(SAP_BLOCKS,n,0.0f,1,1);
      Dweo_blk(SAP_BLOCKS,n,1,1);
      blk_eo_mres(n,mu,nmr);
      Dwoe_blk(SAP_BLOCKS,n,0,0);
      Dwoo_blk(SAP_BLOCKS,n,0.0f,0,0);
      if (eoflg==1) {
	update_flds2(vol,b[n].imb,s,psi,rho);
      } else {
	update_flds1(vol,b[n].imb,mu,s,psi,rho);
      }
      Dw_bnd(SAP_BLOCKS,n,0,0);
    }
  } else {
    for (;n<nm;n+=NTHREAD) {
      assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,rho,1);
      blk_mres(n,mu,nmr);

      update_flds0(vol,b[n].imb,s,psi,rho);
      Dw_bnd(SAP_BLOCKS,n,0,0);
    }
  }
}

void sap(float mu,int isolv,int nmr,spinor *psi,spinor *rho)
{
  int nb = 0 , isw = 0 ;
  const block_t *b = blk_list(SAP_BLOCKS,&nb,&isw);

  // loop black and white SAP blocks
#pragma omp parallel
  {
    dosap( nb/2 , b , nb , 0 , isw , mu , isolv , nmr , psi , rho ) ;

    #pragma omp barrier
    
    sap_com_th( 0 , rho ) ;

    #pragma omp barrier
    
    dosap( nb/2 , b , nb , 1 , isw , mu , isolv , nmr , psi , rho ) ;

    #pragma omp barrier
    
    sap_com_th( 1 , rho ) ;
  }
}

#else

void sap(float mu,int isolv,int nmr,spinor *psi,spinor *rho)
{
   int nb,nbh,isw,eoflg,ic,vol;
   int k,n,nm;
   spinor **s;
   block_t *b;
   tm_parms_t tm;

   tm=tm_parms();
   eoflg=tm.eoflg;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   nbh=nb/2;
   if (isolv)
      vol=(*b).vol/2;
   else
      vol=(*b).vol;

   for (ic=0;ic<2;ic++)
   {
#pragma omp parallel private(k,n,nm,s)
      {
         k=omp_get_thread_num();

         s=b[k].s;

         if (ic^isw)
         {
            n=nbh+k;
            nm=nb;
         }
         else
         {
            n=k;
            nm=nbh;
         }

         if (isolv)
         {
            for (;n<nm;n+=NTHREAD)
            {
               assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,rho,1);
               Dwoo_blk(SAP_BLOCKS,n,0.0f,1,1);
               Dweo_blk(SAP_BLOCKS,n,1,1);
               blk_eo_mres(n,mu,nmr);
               Dwoe_blk(SAP_BLOCKS,n,0,0);
               Dwoo_blk(SAP_BLOCKS,n,0.0f,0,0);

               if (eoflg==1)
                  update_flds2(vol,b[n].imb,s,psi,rho);
               else
                  update_flds1(vol,b[n].imb,mu,s,psi,rho);
               Dw_bnd(SAP_BLOCKS,n,0,0);
            }
         }
         else
         {
            for (;n<nm;n+=NTHREAD)
            {
               assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,rho,1);
               blk_mres(n,mu,nmr);

               update_flds0(vol,b[n].imb,s,psi,rho);
               Dw_bnd(SAP_BLOCKS,n,0,0);
            }
         }
      }

      sap_com(ic,rho);
   }
}

#endif
