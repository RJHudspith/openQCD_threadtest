
/*******************************************************************************
*
* File time4.c
*
* Copyright (C) 2013, 2016, 2018, 2019, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of Dw_blk_dble() and Dwhat_blk_dble().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,bc,count,nt;
   int k,n,nb,isw,bs[4];
   double phi[2],phi_prime[2],theta[3];
   double mu,wt1,wt2,wdt;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time4.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Timing of Dw_blk_dble() and Dwhat_blk_dble()\n");
      printf("--------------------------------------------\n\n");

      print_lattice_sizes();

      if ((VOLUME*sizeof(double))<(64*1024))
      {
         printf("The local size of the gauge field is %d KB\n",
                (int)((72*VOLUME*sizeof(double))/(1024)));
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(double))/(1024)));
      }
      else
      {
         printf("The local size of the gauge field is %d MB\n",
                (int)((72*VOLUME*sizeof(double))/(1024*1024)));
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(double))/(1024*1024)));
      }

#if (defined x64)
#if (defined AVX)
#if (defined FMA3)
      printf("Using AVX and FMA3 instructions\n");
#else
      printf("Using AVX instructions\n");
#endif
#else
      printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
      printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
      printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
      printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
      printf("SSE prefetch instructions are not used\n");
#endif
#endif
      printf("\n");

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);

      bc=find_opt(argc,argv,"-bc");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [time3.c]",
                    "Syntax: time3 [-bc <type>]");
   }

   set_lat_parms(5.5,1.0,0,NULL,0,1.978);
   print_lat_parms(0x2);

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.35;
   theta[1]=-1.25;
   theta[2]=0.78;
   set_bc_parms(bc,0.55,0.78,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(2);

   start_ranlux(0,12345);
   geometry();
   set_dfl_parms(bs,4);
   alloc_bgr(TEST_BLOCKS);

   set_sw_parms(-0.0123);
   mu=0.0785;

   random_ud();
   set_ud_phase();
   sw_term(NO_PTS);
   assign_ud2udblk(TEST_BLOCKS,0);
   assign_swd2swdblk(TEST_BLOCKS,0);

   b=blk_list(TEST_BLOCKS,&nb,&isw);
   random_sd((*b).vol,0,(*b).sd[0],1.0);

   nt=(int)(2.0e6f/(double)(VOLUME_TRD));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
#pragma omp parallel private(k,count,n)
      {
         k=omp_get_thread_num();

         for (count=0;count<nt;count++)
         {
            for (n=k;n<nb;n+=NTHREAD)
            {
               Dw_blk_dble(TEST_BLOCKS,n,mu,0,1);
               Dw_blk_dble(TEST_BLOCKS,n,mu,1,2);
            }
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=1.0e6*wdt/((double)(nt)*(double)(VOLUME_TRD));

   if (my_rank==0)
   {
      printf("Time per lattice point & thread for Dw_blk_dble():\n");
      printf("%4.3f micro sec (%d Mflops)\n\n",wdt,(int)(1920.0/wdt));
   }

   nt=(int)(2.0e6f/(double)(VOLUME_TRD));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
#pragma omp parallel private(k,count,n)
      {
         k=omp_get_thread_num();

         for (count=0;count<nt;count++)
         {
            for (n=k;n<nb;n+=NTHREAD)
            {
               Dwhat_blk_dble(TEST_BLOCKS,n,mu,0,1);
               Dwhat_blk_dble(TEST_BLOCKS,n,mu,1,2);
            }
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   wdt=1.0e6*wdt/((double)(nt)*(double)(VOLUME_TRD));

   if (my_rank==0)
   {
      printf("Time per lattice point & thread for Dwhat_blk_dble():\n");
      printf("%4.3f micro sec (%d Mflops)\n\n",wdt,(int)(1908.0/wdt));
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
