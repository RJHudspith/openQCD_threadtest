
/*******************************************************************************
*
* File ranlxs.c
*
* Copyright (C) 2005, 2019, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Random number generator "ranlxs". See the notes
*
*   "User's guide for ranlxs and ranlxd v3.4" (May 2019)
*
*   "Algorithms used in ranlxs and ranlxd v3.4" (May 2019)
*
* for a detailed description.
*
*   void ranlxs(double *r,int n)
*     Computes the next n single-precision random numbers and assigns
*     them to r[0],...,r[n-1].
*
*   void rlxs_init(int n,int level,int seed,int seed_shift)
*     Allocation and initialization of the states of n copies of the
*     ranlxs generator at the specified luxury level. The seed for the
*     k'th copy is set to seed+k*seed_shift. Admissible levels are 0,
*     1 and 2, while the parameters seed>=1 and seed_shift>=1 must be
*     such that the maximal seed is less than 2^31.
*
*   int rlxs_size(void)
*     Returns the number of integers required to save the states of the
*     allocated copies of the generator.
*
*   void rlxs_get(int *state)
*     Extracts the current states of the allocated copies of the generator
*     and stores the information in the array elements state[0],..,state[N-1],
*     where N is the value returned by rlxs_size().
*
*   void rlxs_reset(int *state)
*     Resets the generator to the state defined by state[0],..,state[N-1],
*     assuming the state was previously saved to this array by the program
*     rlxs_get().
*
* The allocated copies of the generator are stored in the static memory
* of this module. If rlxs_init() is called a second time, the states are
* reinitialized.
*
* The programs rlxs_init(),..,rlxs_reset() are called by the programs in
* ranlux.c, which should be used for the initialization of the generator
* in MPI programs.
*
* Except for ranlxs(), all programs in this module are assumed to be called
* by the OpenMP master thread. When ranlxs() is called by thread number k,
* the random numbers are generated by the k'th copy of the generator.
*
*******************************************************************************/

#define RANLXS_C

#include <stdlib.h>
#if (defined _OPENMP)
#include <omp.h>
#endif
#include "random.h"

static int init=0,next[96];
static int (*is_all)[16];
static float (*rs_all)[96];
static rlx_state_t *state_all;


static void alloc_arrays(int n)
{
   if ((init>0)&&(init!=n))
   {
      rlx_free_state(init,state_all);
      free(state_all);
      afree(rs_all);
      afree(is_all);
      init=0;
   }

   if ((init==0)&&(n>0))
   {
      is_all=amalloc(2*n*sizeof(*is_all),6);
      rs_all=amalloc(n*sizeof(*rs_all),6);
      state_all=malloc(n*sizeof(*state_all));

      error_loc((is_all==NULL)||(rs_all==NULL)||(state_all==NULL),1,
                "rlxs_alloc_arrays [ranlxs.c]","Unable to allocate arrays");

      rlx_alloc_state(n,state_all);
      init=n;
   }
}


void rlxs_init(int n,int level,int seed,int seed_shift)
{
   int k,l;

   alloc_arrays(n);

   for (k=0;k<n;k++)
   {
      if (level==0)
         state_all[k].pr=109;
      else if (level==1)
         state_all[k].pr=202;
      else
         state_all[k].pr=397;

      rlx_init(state_all+k,seed+k*seed_shift,0);
   }

   for (k=0;k<95;k++)
      next[k]=k+1;
   next[95]=0;

   for (k=0;k<n;k++)
   {
      is_all[k][0]=95;
      is_all[k][1]=0;

      for (l=2;l<16;l++)
         is_all[k][l]=0;
   }
}


void ranlxs(float *r,int n)
{
   int i,is,is_old,k;
   float *rs;
   rlx_state_t *state;

#if (defined _OPENMP)
   i=omp_get_thread_num();
#else
   i=0;
#endif
   is=is_all[i][0];
   is_old=is_all[i][1];
   rs=rs_all[i];
   state=state_all+i;

   for (k=0;k<n;k++)
   {
      is=next[is];

      if (is==is_old)
      {
         rlx_update(state);
         rlx_converts(state,rs);
         is=8*(*state).ir;
         is_old=is;
      }

      r[k]=rs[is];
   }

   is_all[i][0]=is;
   is_all[i][1]=is_old;
}


int rlxs_size(void)
{
   error_loc(init==0,1,"rlxs_size [ranlxs.c]","ranlxs is not initialized");

   return 2+102*init;
}


void rlxs_get(int *state)
{
   int k;

   error_loc(init==0,1,"rlxs_get [ranlxs.c]","ranlxs is not initialized");

   state[0]=rlxs_size();
   state[1]=state_all[0].pr;
   state+=2;

   for (k=0;k<init;k++)
   {
      rlx_get_state(state_all+k,state);
      state+=100;

      state[0]=state_all[k].ir;
      state[1]=is_all[k][0];
      state+=2;
   }
}


void rlxs_reset(int *state)
{
   int n,pr,ie,k;

   n=(state[0]-2)/102;
   pr=state[1];
   state+=2;

   alloc_arrays(n);

   ie=((n<1)||((pr!=109)&&(pr!=202)&&(pr!=397)));

   for (k=0;k<n;k++)
   {
      rlx_set_state(state,state_all+k);
      state+=100;

      state_all[k].pr=pr;
      state_all[k].ir=state[0];
      is_all[k][0]=state[1];
      is_all[k][1]=8*state[0];

      ie|=((state[0]<0)||(state[0]>=12));
      ie|=((state[1]<0)||(state[1]>=96));

      state+=2;
   }

   error_loc(ie,1,"rlxs_reset [ranlxs.c]","Unexpected input data");

   for (k=0;k<95;k++)
      next[k]=k+1;
   next[95]=0;

   for (k=0;k<n;k++)
      rlx_converts(state_all+k,rs_all[k]);
}
