

 module module_bl_mynnedmf_common









  use ccpp_kind_types,  only : kind_phys


  use module_model_constants, only:         &
    & karman, g, p1000mb,                   &
    & cp, r_d, r_v, rcp, xlv, xlf, xls,     &
    & svp1, svp2, svp3, p608, ep_2, rvovrd, &                                                                         
    & cpv, cliq, cice, svpt0

 implicit none
 save






















 

 real(kind_phys),parameter:: grav    = g
 real(kind_phys),parameter:: t0c     = svpt0
 real(kind_phys),parameter:: tref    = 300.0     
 real(kind_phys),parameter:: TKmin   = 253.0     
 real(kind_phys),parameter:: tice    = 238.0     
 real(kind_phys),parameter:: tliq    = 269.0     
 real(kind_phys),parameter:: ep_3    = 1.-ep_2   
 real(kind_phys),parameter:: gtr     = grav/tref
 real(kind_phys),parameter:: rk      = cp/r_d
 real(kind_phys),parameter:: tv0     = p608*tref
 real(kind_phys),parameter:: tv1     = (1.+p608)*tref
 real(kind_phys),parameter:: xlscp   = (xlv+xlf)/cp
 real(kind_phys),parameter:: xlvcp   = xlv/cp
 real(kind_phys),parameter:: g_inv   = 1./grav


 real(kind_phys),parameter:: wfa_max = 800e10    
 real(kind_phys),parameter:: wfa_min = 1e6       
 real(kind_phys),parameter:: ifa_max = 500e6     
 real(kind_phys),parameter:: ifa_min = 0.0       
 real(kind_phys),parameter:: wfa_ht  = 2000.     
 real(kind_phys),parameter:: ifa_ht  = 10000.    





 real(kind_phys),parameter:: cphm_st = 5.0, cphm_unst = 16.0
 real(kind_phys),parameter:: cphh_st = 5.0, cphh_unst = 16.0


 real(kind_phys),parameter:: qmin    = 0.0
 real(kind_phys),parameter:: zmax    = 1.0
 real(kind_phys),parameter:: Sqfac   = 3.0
 real(kind_phys),parameter:: qkemin  = 1.e-5


 real(kind_phys),parameter:: rr2     = 0.7071068
 real(kind_phys),parameter:: rrp     = 0.3989423
 

 real(kind_phys),parameter:: pr      = 0.74       
 real(kind_phys),parameter:: g1      = 0.235      
 real(kind_phys),parameter:: b1      = 24.0
 real(kind_phys),parameter:: b2      = 15.0       
 real(kind_phys),parameter:: c2      = 0.729      
 real(kind_phys),parameter:: c3      = 0.340      
 real(kind_phys),parameter:: c4      = 0.0
 real(kind_phys),parameter:: c5      = 0.2
 real(kind_phys),parameter:: a1      = b1*( 1.0-3.0*g1 )/6.0
 
 real(kind_phys),parameter:: c1      = g1 -1.0/( 3.0*a1*2.88449914061481660)
 real(kind_phys),parameter:: a2      = a1*( g1-c1 )/( g1*pr )
 real(kind_phys),parameter:: g2      = b2/b1*( 1.0-c3 ) +2.0*a1/b1*( 3.0-2.0*c2 )

 real(kind_phys),parameter:: cc2     = 1.0-c2
 real(kind_phys),parameter:: cc3     = 1.0-c3
 real(kind_phys),parameter:: e1c     = 3.0*a2*b2*cc3
 real(kind_phys),parameter:: e2c     = 9.0*a1*a2*cc2
 real(kind_phys),parameter:: e3c     = 9.0*a2*a2*cc2*( 1.0-c5 )
 real(kind_phys),parameter:: e4c     = 12.0*a1*a2*cc2
 real(kind_phys),parameter:: e5c     = 6.0*a1*a1




 real(kind_phys),parameter:: zero    = 0.0
 real(kind_phys),parameter:: one     = 1.0
 real(kind_phys),parameter:: two     = 2.0
 real(kind_phys),parameter:: three   = 3.0
 real(kind_phys),parameter:: four    = 4.0
 real(kind_phys),parameter:: five    = 5.0
 real(kind_phys),parameter:: six     = 6.0
 real(kind_phys),parameter:: seven   = 7.0
 real(kind_phys),parameter:: eight   = 8.0
 real(kind_phys),parameter:: nine    = 9.0
 real(kind_phys),parameter:: ten     = 10.0
 real(kind_phys),parameter:: twenty  = 20.0
 real(kind_phys),parameter:: thirty  = 30.0
 real(kind_phys),parameter:: forty   = 40.0
 real(kind_phys),parameter:: fifty   = 50.0
 real(kind_phys),parameter:: hundred = 100.0
 real(kind_phys),parameter:: p01     = 0.01
 real(kind_phys),parameter:: p1      = 0.1
 real(kind_phys),parameter:: p2      = 0.2
 real(kind_phys),parameter:: p25     = 0.25
 real(kind_phys),parameter:: p3      = 0.3
 real(kind_phys),parameter:: p333    = 1.0/3.0
 real(kind_phys),parameter:: p4      = 0.4
 real(kind_phys),parameter:: p5      = 0.5
 real(kind_phys),parameter:: p6      = 0.6
 real(kind_phys),parameter:: p666    = 2.0/3.0
 real(kind_phys),parameter:: p7      = 0.7
 real(kind_phys),parameter:: p75     = 0.75
 real(kind_phys),parameter:: p8      = 0.8
 real(kind_phys),parameter:: p9      = 0.9
 real(kind_phys),parameter:: p95     = 0.95
 real(kind_phys),parameter:: p99     = 0.99 

 end module module_bl_mynnedmf_common
