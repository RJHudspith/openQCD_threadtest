
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2007, 2011-2013, 2016, 2021 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Consistency checks on Aw_dble(),..,Awhat().
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dfl.h"
#include "little.h"
#include "global.h"

static int bc,Ns;


static void random_basis(void)
{
   int i;
   spinor **ws;

   ws=reserve_ws(Ns);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME_TRD,2,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }

   dfl_subspace(ws);
   release_ws();
}


static int is_zero(complex_dble *Amat)
{
   int n,ie;

   ie=1;

   for (n=0;n<(Ns*Ns);n++)
   {
      ie&=(Amat[n].re==0.0);
      ie&=(Amat[n].im==0.0);
   }

   return ie;
}


static int check_bndAwop(void)
{
   int nb,isw,ie,n,ifc;
   complex_dble **Ahop;
   Aw_dble_t A;
   block_t *b;

   ie=0;

   if (bc!=3)
   {
      b=blk_list(DFL_BLOCKS,&nb,&isw);
      A=Awop_dble();
      Ahop=A.Ahop;

      for (n=0;n<nb;n++)
      {
         if ((cpr[0]==0)&&((*b).bo[0]==0))
         {
            ie|=(is_zero(Ahop[8*n])^0x1);

            for (ifc=1;ifc<8;ifc++)
               ie|=is_zero(Ahop[8*n+ifc]);
         }
         else if ((cpr[0]==(NPROC0-1))&&(((*b).bo[0]+(*b).bs[0])==L0))
         {
            ie|=is_zero(Ahop[8*n]);
            ie|=(is_zero(Ahop[8*n+1])^0x1);

            for (ifc=2;ifc<8;ifc++)
               ie|=is_zero(Ahop[8*n+ifc]);
         }
         else
         {
            for (ifc=0;ifc<8;ifc++)
               ie|=is_zero(Ahop[8*n+ifc]);
         }

         b+=1;
      }
   }

   return ie;
}


int main(int argc,char *argv[])
{
   int my_rank,iop,ifail,is;
   int bs[4],nb,nv,nvh;
   double phi[2],phi_prime[2],theta[3];
   double mu,d;
   qflt rqsm;
   complex **wv,z;
   complex_dble **wvd,zd;
   void (*op)(complex *v,complex *w);
   void (*op_dble)(complex_dble *v,complex_dble *w);
   char *pr,*prd;
   FILE *fin=NULL,*flog=NULL;

   mpi_init(argc,argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Consistency checks on Aw_dble(),..,Awhat()\n");
      printf("------------------------------------------\n\n");

      print_lattice_sizes();

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);

      bc=find_opt(argc,argv,"-bc");
      is=find_opt(argc,argv,"-sw");

      if (bc!=0)
         error_root(sscanf(argv[bc+1],"%d",&bc)!=1,1,"main [check2.c]",
                    "Syntax: check2 [-bc <type>] [-sw <type>]");

      if (is!=0)
         error_root(sscanf(argv[is+1],"%d",&is)!=1,1,"main [check2.c]",
                    "Syntax: check2 [-bc <type>] [-sw <type>]");
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&bc,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&is,1,MPI_INT,0,MPI_COMM_WORLD);

   check_machine();
   set_lat_parms(5.5,1.0,0,NULL,is,1.978);
   print_lat_parms(0x2);

   phi[0]=0.123;
   phi[1]=-0.534;
   phi_prime[0]=0.912;
   phi_prime[1]=0.078;
   theta[0]=0.38;
   theta[1]=-1.25;
   theta[2]=0.54;
   set_bc_parms(bc,1.0,1.0,0.9012,1.2034,phi,phi_prime,theta);
   print_bc_parms(0x2);

   set_sw_parms(0.125);
   set_dfl_parms(bs,Ns);
   mu=0.0376;

   start_ranlux(0,123456);
   geometry();

   alloc_ws(Ns);
   alloc_wv(4);
   alloc_wvd(6);

   wv=reserve_wv(4);
   wvd=reserve_wvd(4);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;
   nvh=nv/2;

   random_ud();
   set_ud_phase();
   random_basis();

   ifail=set_Awhat(mu);
   error(ifail!=0,1,"main [check2.c]","Inversion of Ablk failed");

   zd.re=-1.0;
   zd.im=0.0;
   z.re=-1.0f;
   z.im=0.0f;

   for (iop=0;iop<6;iop++)
   {
      if (iop==0)
      {
         op=Awhat;
         pr= "Awhat()  ";
         op_dble=Awhat_dble;
         prd="Awhat_dble()  ";
      }
      else if (iop==1)
      {
         op=Aweeinv;
         pr= "Aweeinv()";
         op_dble=Aweeinv_dble;
         prd="Aweeinv_dble()";
      }
      else if (iop==2)
      {
         op=Awooinv;
         pr= "Awooinv()";
         op_dble=Awooinv_dble;
         prd="Awooinv_dble()";
      }
      else if (iop==3)
      {
         op=Awoe;
         pr= "Awoe()   ";
         op_dble=Awoe_dble;
         prd="Awoe_dble()   ";
      }
      else if (iop==4)
      {
         op=Aweo;
         pr= "Aweo()   ";
         op_dble=Aweo_dble;
         prd="Aweo_dble()   ";
      }
      else
      {
         op=Aw;
         pr= "Aw()     ";
         op_dble=Aw_dble;
         prd="Aw_dble()     ";
      }

      random_vd(nv,0,wvd[0],1.0);
      random_vd(nv,0,wvd[1],1.0);
      assign_vd2vd(nv,0,wvd[0],wvd[2]);
      assign_vd2vd(nv,0,wvd[1],wvd[3]);

      assign_vd2v(nv,0,wvd[0],wv[0]);
      assign_vd2v(nv,0,wvd[1],wv[1]);
      assign_v2v(nv,0,wv[0],wv[2]);
      assign_v2v(nv,0,wv[1],wv[3]);

      op_dble(wvd[0],wvd[1]);
      op(wv[0],wv[1]);

      mulc_vadd_dble(nv,0,wvd[2],wvd[0],zd);
      rqsm=vnorm_square_dble(nv,0,wvd[2]);
      error(rqsm.q[0]!=0.0,1,"main [check2.c]",
            "%s modifies the input field",prd);

      mulc_vadd(nv,0,wv[2],wv[0],z);
      d=(double)(vnorm_square(nv,0,wv[2]));
      error(d!=0.0,1,"main [check2.c]",
            "%s modifies the input field",pr);

      if ((iop<2)||(iop==4))
      {
         mulc_vadd_dble(nvh,0,wvd[3]+nvh,wvd[1]+nvh,zd);
         rqsm=vnorm_square_dble(nvh,0,wvd[3]+nvh);
         error(rqsm.q[0]!=0.0,1,"main [check2.c]",
               "%s modifies the odd components of the output field",prd);

         mulc_vadd(nvh,0,wv[3]+nvh,wv[1]+nvh,z);
         d=(double)(vnorm_square(nvh,0,wv[3]+nvh));
         error(d!=0.0,1,"main [check2.c]",
               "%s modifies the odd components of the output field",pr);

         assign_vd2v(nvh,0,wvd[1],wv[0]);
         mulc_vadd(nvh,0,wv[0],wv[1],z);
         d=(double)(vnorm_square(nvh,1,wv[0])/
                    vnorm_square(nvh,1,wv[1]));
         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));
      }

      if ((iop==2)||(iop==3))
      {
         mulc_vadd_dble(nvh,0,wvd[3],wvd[1],zd);
         rqsm=vnorm_square_dble(nvh,0,wvd[3]);
         error(rqsm.q[0]!=0.0,1,"main [check2.c]",
               "%s modifies the even components of the output field",prd);

         mulc_vadd(nvh,0,wv[3],wv[1],z);
         d=(double)(vnorm_square(nvh,0,wv[3]));
         error(d!=0.0,1,"main [check2.c]",
               "%s modifies the even components of the output field",pr);

         assign_vd2v(nvh,0,wvd[1]+nvh,wv[0]+nvh);
         mulc_vadd(nvh,0,wv[0]+nvh,wv[1]+nvh,z);
         d=(double)(vnorm_square(nvh,1,wv[0]+nvh)/
                    vnorm_square(nvh,1,wv[1]+nvh));

         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));
      }

      if (iop==5)
      {
         assign_vd2v(nv,0,wvd[1],wv[0]);
         mulc_vadd(nv,0,wv[0],wv[1],z);
         d=(double)(vnorm_square(nv,1,wv[0])/
                    vnorm_square(nv,1,wv[1]));
         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));
      }
   }

   ifail=set_Awhat(-mu);
   error(ifail!=0,1,"main [check2.c]","Inversion of Aee or Aoo failed");

   random_vd(nvh,0,wvd[0],1.0);
   set_vd2zero(nvh,0,wvd[0]+nvh);
   Aw_dble(wvd[0],wvd[1]);

   Aweeinv_dble(wvd[1],wvd[2]);
   mulc_vadd_dble(nvh,0,wvd[2],wvd[0],zd);
   rqsm=vnorm_square_dble(nvh,1,wvd[2]);
   d=rqsm.q[0];
   rqsm=vnorm_square_dble(nvh,1,wvd[0]);
   d/=rqsm.q[0];

   if (my_rank==0)
   {
      printf("\n");
      printf("Comparison of Aweeinv_dble() and Aw_dble(): %.1e\n",sqrt(d));
   }

   Awoe_dble(wvd[0],wvd[2]);
   mulc_vadd_dble(nvh,0,wvd[2]+nvh,wvd[1]+nvh,zd);
   rqsm=vnorm_square_dble(nvh,1,wvd[2]+nvh);
   d=rqsm.q[0];
   rqsm=vnorm_square_dble(nvh,1,wvd[1]+nvh);
   d/=rqsm.q[0];

   if (my_rank==0)
      printf("Comparison of Awoe_dble()    and Aw_dble(): %.1e\n",sqrt(d));

   random_vd(nvh,0,wvd[0]+nvh,1.0);
   set_vd2zero(nvh,0,wvd[0]);
   Aw_dble(wvd[0],wvd[1]);

   Awooinv_dble(wvd[1],wvd[2]);
   mulc_vadd_dble(nvh,0,wvd[2]+nvh,wvd[0]+nvh,zd);
   rqsm=vnorm_square_dble(nvh,1,wvd[2]+nvh);
   d=rqsm.q[0];
   rqsm=vnorm_square_dble(nvh,1,wvd[0]+nvh);
   d/=rqsm.q[0];

   if (my_rank==0)
      printf("Comparison of Awooinv_dble() and Aw_dble(): %.1e\n",sqrt(d));

   random_vd(nvh,0,wvd[2],1.0);
   assign_vd2vd(nvh,0,wvd[2],wvd[3]);
   Aweo_dble(wvd[0],wvd[2]);
   mulc_vadd_dble(nvh,0,wvd[3],wvd[2],zd);
   mulc_vadd_dble(nvh,0,wvd[3],wvd[1],zd);
   rqsm=vnorm_square_dble(nvh,1,wvd[3]);
   d=rqsm.q[0];
   rqsm=vnorm_square_dble(nvh,1,wvd[1]);
   d/=rqsm.q[0];

   if (my_rank==0)
      printf("Comparison of Aweo_dble()    and Aw_dble(): %.1e\n",sqrt(d));

   random_vd(nv,0,wvd[0],1.0);
   Awhat_dble(wvd[0],wvd[1]);
   Awoe_dble(wvd[0],wvd[2]);
   Awooinv_dble(wvd[2],wvd[3]);
   set_vd2zero(nvh,0,wvd[0]+nvh);
   Aw_dble(wvd[0],wvd[2]);
   Aweo_dble(wvd[3],wvd[2]);
   Aweeinv_dble(wvd[2],wvd[3]);

   mulc_vadd_dble(nvh,0,wvd[3],wvd[1],zd);
   rqsm=vnorm_square_dble(nvh,1,wvd[3]);
   d=rqsm.q[0];
   rqsm=vnorm_square_dble(nvh,1,wvd[1]);
   d/=rqsm.q[0];

   if (my_rank==0)
   {
      printf("Comparison of Aweeinv_dble(), Awooinv_dble(), \n");
      printf(" Awoe_dble(), Aweo_dble() and Awhat_dble(): %.1e\n\n",
             sqrt(d));
      fflush(flog);
   }

   ifail=check_bndAwop();
   error(ifail!=0,1,"main [check2.c]",
         "Hopping terms Aoe,Aeo at the lattice boundaries do not vanish");

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
