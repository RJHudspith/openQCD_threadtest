
/*******************************************************************************
*
* File lattice.h
*
* Copyright (C) 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef LATTICE_H
#define LATTICE_H

#ifndef BLOCK_H
#include "block.h"
#endif

typedef struct
{
   int nu0,nuk;
   int *iu0,*iuk;
} uidx_t;

typedef struct
{
   int nft[2];
   int *ift[2];
} ftidx_t;

/* BCNDS_C */
extern int *bnd_lks(int *n);
extern int *bnd_pts(int *n);
extern void set_bc(void);
extern int check_bc(double tol);
extern void bnd_s2zero(ptset_t set,spinor *s);
extern void bnd_sd2zero(ptset_t set,spinor_dble *sd);

/* BLK_GEOMETRY_C */
#if ((defined BLK_GEOMETRY_C)||(defined BLOCK_C))
extern void blk_geometry(block_t *b);
extern void blk_imbed(block_t *b);
extern void bnd_geometry(block_t *b);
extern void bnd_imbed(block_t*b);
#endif

/* FTIDX_C */
extern void set_ftidx(void);
extern ftidx_t *ftidx(void);
extern void plaq_ftidx(int n,int ix,int *ip);

/* GEOGEN */
extern int ipr_global(int *n);
#if ((defined GEOGEN_C)||(defined GEOMETRY_C))
extern void set_cpr(void);
extern void set_sbofs(void);
extern void set_iupdn(void);
extern void set_map(void);
#endif

/* GEOMETRY_C */
extern void geometry(void);
extern int global_time(int ix);
extern void ipt_global(int *x,int *ip,int *ix);

/* UIDX_C */
extern void set_uidx(void);
extern uidx_t *uidx(void);
extern void plaq_uidx(int n,int ix,int *ip);

#endif
