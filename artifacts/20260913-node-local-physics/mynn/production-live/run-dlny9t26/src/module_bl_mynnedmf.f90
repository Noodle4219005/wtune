






































































































































module module_bl_mynnedmf

 implicit none




 logical,parameter:: debug_code = .false.
 integer,parameter:: idbg       = 452 
 integer,parameter:: jdbg       = 272 

contains









  subroutine mynnedmf(            i                 , j                 , &
             initflag           , restart           , cycling           , &
             delt               , dz1               , dx                , &
             
             u1                 , v1                , w1                , &
             th1                , sqv1              , sqc1              , &
             sqi1               , sqs1              , qnc1              , &
             qni1               , qnwfa1            , qnifa1            , &
             qnbca1             , ozone1            , pres1             , &
             ex1                , rho1              , tk1               , &
             
             xland              , ts                , qsfc              , &
             ps                 , ust               , ch                , &
             hfx                , qfx               , znt               , &
             wspd               , uoce              , voce              , &
             
             qke1               , qke_adv1          , el1               , &
             sh1                , sm1               , kh1               , &
             km1                ,                                         &
             
             nchem              , ndvel             ,                     &
             chem1              , vdep              , frp               , &
             emis_ant_no        , mix_chem,enh_mix  , settle1           , &
             
             scalars            , nscalars          ,                     &
             
             tsq1               , qsq1              , cov1              , &
             
             du1                , dv1               , dth1              , &
             dqv1               , dqc1              , dqi1              , &
             dqnc1              , dqni1             , dqs1              , &
             dqnwfa1            , dqnifa1           , dqnbca1           , &
             dozone1            , rthraten1         ,                     &
             
             pblh               , kpbl              , maxwidth_dd       , &
             maxwidth           , maxmf             , ztop_plume        , &
             excess_h           , excess_q          , maxmf_dd          , &
             maxtkeprod         , cldtop_cooling    , ent_eff           , &
             
             dqke1              , qwt1              , qshear1           , &
             qbuoy1             , qdiss1            ,                     &
             
             qc_bl1             , qi_bl1            , cldfra_bl1        , &
             
             bl_mynn_tkeadvect  , tke_budget        , bl_mynn_cloudpdf  , &
             bl_mynn_mixlength  , bl_mynn_closure   , bl_mynn_ess       , &
             bl_mynn_edmf       , bl_mynn_edmf_mom  , bl_mynn_edmf_tke  , &
             bl_mynn_mixscalars , bl_mynn_mixaerosols,bl_mynn_mixnumcon , &
             bl_mynn_output     , bl_mynn_cloudmix  , bl_mynn_mixqt     , &
             bl_mynn_edmf_dd    ,                                         &
             
             edmf_a1            , edmf_w1           , edmf_qt1          , &
             edmf_thl1          , edmf_ent1         , edmf_qc1          , &
             sub_thl1           , sub_sqv1          , det_thl1          , &
             det_sqv1           ,                                         &
             
             spp_pbl            , pattern_spp_pbl1  ,                     &
             
             FLAG_QC            , FLAG_QI           , FLAG_QNC          , &
             FLAG_QNI           , FLAG_QS           , FLAG_QNWFA        , &
             FLAG_QNIFA         , FLAG_QNBCA        , FLAG_OZONE        , &
             KTS                , KTE               , errmsg, errflg      )


 use module_bl_mynnedmf_common, only: cp,xlvcp,xlscp,p608,p1000mb,rcp,&
      karman,gtr,cphh_st,cphm_unst,cphh_unst,b1,zero,one,p01,p5,      &
      five,kind_phys
    
 integer, intent(in) :: initflag
 logical, intent(in) :: restart,cycling
 integer, intent(in) :: tke_budget
 integer, intent(in) :: bl_mynn_cloudpdf
 integer, intent(in) :: bl_mynn_mixlength
 integer, intent(in) :: bl_mynn_edmf
 integer, intent(in) :: bl_mynn_edmf_dd
 logical, intent(in) :: bl_mynn_tkeadvect
 integer, intent(in) :: bl_mynn_edmf_mom
 integer, intent(in) :: bl_mynn_edmf_tke
 integer, intent(in) :: bl_mynn_mixscalars
 integer, intent(in) :: bl_mynn_mixaerosols
 integer, intent(in) :: bl_mynn_mixnumcon
 integer, intent(in) :: bl_mynn_output
 integer, intent(in) :: bl_mynn_cloudmix
 integer, intent(in) :: bl_mynn_mixqt
 integer, intent(in) :: bl_mynn_ess
 real(kind_phys), intent(in) :: bl_mynn_closure

 logical, intent(in) :: FLAG_QI,FLAG_QNI,FLAG_QC,FLAG_QNC,&
                        FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA, &
                        FLAG_OZONE,FLAG_QS

 logical, intent(in) :: mix_chem,enh_mix

 integer, intent(in) :: KTS,KTE

 character(len=*),intent(out):: &
    errmsg        
 integer,intent(out):: &
    errflg        



 real(kind_phys), intent(in)    :: delt
 real(kind_phys), intent(in)    :: dx
 real(kind_phys), intent(in)    :: ust,ch,qsfc,ps,wspd,xland
 real(kind_phys), intent(in)    :: ts,znt,hfx,qfx,uoce,voce
 real(kind_phys), intent(inout) :: pblh,excess_h,excess_q
 real(kind_phys), intent(inout) :: maxmf,maxwidth,ztop_plume
 real(kind_phys), intent(inout) :: maxmf_dd,maxwidth_dd,maxtkeprod,cldtop_cooling,ent_eff
 integer,         intent(in)    :: i,j
 integer,         intent(inout) :: kpbl

 real(kind_phys) :: psig_bl,psig_shcu,rmol,rmolh
 integer :: imd,jmd
 integer :: k,kproblem


 real(kind_phys), dimension(kts:kte), intent(in)      ::            &
       dz1,u1,v1,w1,th1,pres1,ex1,rho1,tk1,rthraten1
 real(kind_phys), dimension(kts:kte), intent(inout)   ::            &
       sqv1,sqc1,sqi1,sqs1,qni1,qnc1,qnwfa1,qnifa1,qnbca1,ozone1,   &
       qke1,tsq1,qsq1,cov1,qke_adv1,                                &
       sh1,sm1,el1,                                                 & 
       du1,dv1,dth1,dqv1,dqc1,dqi1,dqs1,                            &
       dqni1,dqnc1,dqnwfa1,dqnifa1,dqnbca1,dozone1,                 &
       qc_bl1,qi_bl1,cldfra_bl1,edmf_a1,edmf_w1,                    &
       edmf_qt1,edmf_thl1,edmf_ent1,edmf_qc1,                       &
       sub_thl1,sub_sqv1,det_thl1,det_sqv1
 real(kind_phys), dimension(kts:kte), intent(inout), optional ::    &
       qwt1,qshear1,qbuoy1,qdiss1,dqke1
 real(kind_phys), dimension(kts:kte), intent(out)     ::            & 
       kh1,km1

 real(kind_phys), dimension(kts:kte)                  ::            &
       qc_bl1_old,qi_bl1_old,cldfra_bl1_old,dummy1,dummy2,          &
       diss_heat1,                                                  &
       thl1,thv1,thlv1,qv1,qc1,qi1,qs1,sqw1,                        &
       thl_tot1,qc_tot1,qi_tot1,                                    &
       dfm1, dfh1, dfq1, tcd1, qcd1,                                &
       pdk1, pdt1, pdq1, pdc1,                                      &
       vt1, vq1, sgm1, kzero1
 real(kind_phys), dimension(kts:kte)                  ::            &
       edmf_u1,edmf_v1,edmf_qv1
 

 integer, intent(in) ::   nchem, ndvel
 real(kind_phys), dimension(kts:kte,nchem), intent(inout) :: chem1
 real(kind_phys), dimension(kts:kte,nchem), intent(in   ) :: settle1
 real(kind_phys), dimension(ndvel), intent(in)    :: vdep
 real(kind_phys),                   intent(in)    :: frp,emis_ant_no
 real(kind_phys), dimension(kts:kte+1,nchem)      :: s_awchem1
 integer :: ic

 integer, intent(in) ::   nscalars 
 real(kind_phys), dimension(kts:kte,nscalars), intent(inout) :: scalars
 real(kind_phys), dimension(kts:kte+1,nscalars)      :: s_awscalars1



 real(kind_phys), dimension(kts:kte) ::                             &
       dth1mf,dqv1mf,dqc1mf,du1mf,dv1mf
 real(kind_phys), dimension(kts:kte) ::                             &
       edmf_a_dd1,edmf_w_dd1,edmf_qt_dd1,edmf_thl_dd1,              &
       edmf_ent_dd1,edmf_qc_dd1
 real(kind_phys), dimension(kts:kte) ::                             &
       sub_u1,sub_v1,det_sqc1,det_u1,det_v1
 real(kind_phys), dimension(kts:kte+1) ::                           & 
       s_aw1,s_awthl1,s_awqt1,                                      &
       s_awqv1,s_awqc1,s_awu1,s_awv1,s_awqke1,s_awqsq1,             &
       s_awqnc1,s_awqni1,s_awqnwfa1,s_awqnifa1,                     &
       s_awqnbca1
 real(kind_phys), dimension(kts:kte+1) ::                           & 
       sd_aw1,sd_awthl1,sd_awqt1,                                   &
       sd_awqv1,sd_awqc1,sd_awu1,sd_awv1,sd_awqke1,                 &
       sd_awqi1,sd_awqnc1,sd_awqni1,                                &
       sd_awqnwfa1,sd_awqnifa1
 integer:: ktop_plume
 real(kind_phys), dimension(kts:kte+1) :: zw1              

 real(kind_phys), dimension(kts:kte) :: TKEprod_dn,TKEprod_up






 integer,parameter:: bl_mynn_stfunc = 0











 real(kind_phys),parameter:: CKmod   = 1.0


 real(kind_phys),parameter:: taue    = 80.



 real(kind_phys),parameter:: upwind  = 1.0 
                                           
                                           
                                           

 
 real(kind_phys) :: cpm,sqcg,flt,fltv,flq,flqv,flqc,                &
       pmz,phh,exnerg,zet,phi_m,                                    &
       afk,abk,th_sfc,wsp

 logical :: INITIALIZE_QKE,problem


 integer,  intent(in)                             :: spp_pbl
 real(kind_phys), dimension(kts:kte), intent(in)  :: pattern_spp_pbl1


 integer :: nsub
 real(kind_phys) :: delt2

    errmsg = " "
    errflg = 0

    if (debug_code) then 
       problem = .false.
       do k=kts,kte
          wsp  = sqrt(u1(k)**2 + v1(k)**2)
          if (abs(hfx) > 1200. .or. abs(qfx) > 0.001 .or.           &
              wsp > 270. .or. tk1(k) > 380. .or. tk1(k) < 160. .or. &
              sqv1(k) < zero .or. sqc1(k) < zero .or.               &
              (ps-pres1(kts)) < zero) then
             kproblem = k
             problem = .true.
             print*,"Incoming problem at: i=",i," j=",j," k=",k
             print*," QFX=",qfx," HFX=",hfx
             print*," wsp=",wsp," T=",tk1(k)
             print*," qv=",sqv1(k)," qc=",sqc1(k)
             print*," u*=",ust," wspd=",wspd
             print*," xland=",xland," ts=",ts
             print*," ps=",ps,"delp1=",ps-pres1(kts)
             print*," znt=",znt," dx=",dx," dz(1)=",dz1(1)
          endif
       enddo
       if (problem) then
          print*,"===tk:",tk1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qv:",sqv1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qc:",sqc1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qi:",sqi1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"====u:",u1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"====v:",v1(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"====p:",pres1(max(kproblem-3,1):min(kproblem+3,kte))
       endif
    endif


    IMD=10
    JMD=10


    
    edmf_a1      =zero
    edmf_w1      =zero
    edmf_qt1     =zero
    edmf_thl1    =zero
    edmf_ent1    =zero
    edmf_qc1     =zero
    edmf_u1      =zero
    edmf_v1      =zero
    edmf_qv1     =zero
    sub_thl1     =zero
    sub_sqv1     =zero
    det_thl1     =zero
    det_sqv1     =zero
    edmf_a_dd1   =zero
    edmf_w_dd1   =zero
    edmf_qc_dd1  =zero
    
    ktop_plume   =0    
    ztop_plume   =zero
    maxwidth     =zero
    maxmf        =zero
    excess_h     =zero
    excess_q     =zero
    kzero1       =zero

    




    if (initflag > 0 .and. .not.restart) then

       
       if ( (restart .or. cycling)) then
          if (maxval(qke1(:)) < 0.0002) then
             initialize_qke = .true.
             
          else
             initialize_qke = .false.
             
          endif
       else 
          initialize_qke = .true.
          
       endif
 
       if (.not.restart .or. .not.cycling) THEN
          sh1         =zero
          sm1         =zero
          el1         =zero
          tsq1        =zero
          qsq1        =zero
          cov1        =zero
          cldfra_bl1  =zero
          qc_bl1      =zero
          qi_bl1      =zero
          qke1        =zero
          qke_adv1    =zero
       end if
       dqc1           =zero
       dqi1           =zero
       dqni1          =zero
       dqnc1          =zero
       dqnwfa1        =zero
       dqnifa1        =zero
       dqnbca1        =zero
       dozone1        =zero
       qc_bl1_old     =zero
       cldfra_bl1_old =zero
       sgm1           =zero
       vt1            =zero
       vq1            =zero
       km1            =zero
       kh1            =zero

       if (tke_budget .eq. 1) then
          qwt1        =zero
          qshear1     =zero
          qbuoy1      =zero
          qdiss1      =zero
          dqke1       =zero
       endif

       zw1(kts)=zero
       do k=kts,kte
          
          sqw1(k)=sqv1(k)+sqc1(k)+sqi1(k)
          thl1(k)=th1(k) - xlvcp/ex1(k)*sqc1(k) &
               &         - xlscp/ex1(k)*(sqi1(k))
          
          
          
          
          thlv1(k)=thl1(k)*(one+p608*sqv1(k))
          thv1(k)=th1(k)*(one+p608*sqv1(k) - (sqc1(k)+sqi1(k)))
          zw1(k+1)=zw1(k)+dz1(k)
       enddo

       if (INITIALIZE_QKE) then
          
          
          
          do k=kts,kte
             qke1(k)=five * ust * MAX((ust*700._kind_phys - zw1(k))/(MAX(ust,p01)*700.), p01)
          enddo
       endif



       call get_pblh(kts,kte,pblh,thv1,qke1,ust,zw1,dz1,xland,kpbl)
             


       call scale_aware(dx,pblh,psig_bl,psig_shcu)

       




       rmol = zero
       rmolh= zero
       call mym_initialize (                & 
            &kts,kte,ckmod,taue,xland,pblh, &
            &dz1, dx, zw1, pres1, ex1,      &
            &u1, v1, thl1, sqv1,            &
            &th1, thv1, thlv1,              &
            &sh1, sm1,                      &
            &ust, rmol, rmolh,              &
            &el1, qke1,                     &
            &tsq1, qsq1, cov1,              &
            &psig_bl, cldfra_bl1,           &
            &bl_mynn_mixlength,             &
            &bl_mynn_ess,                   &
            &edmf_w1,edmf_a1,               &
            &edmf_w_dd1,edmf_a_dd1,         &
            &initialize_qke,                &
            &spp_pbl,pattern_spp_pbl1       )

       if (.not.restart) then
          
          IF (bl_mynn_tkeadvect) THEN
             DO k=KTS,KTE
                qke_adv1(k)=qke1(k)
             ENDDO
          ENDIF
       ENDIF











    ENDIF 



    IF (bl_mynn_tkeadvect) THEN
       qke1(kts:kte)=qke_adv1(kts:kte)
    ENDIF

    
    if (tke_budget .eq. 1) then
       dqke1(kts:kte)=qke1(kts:kte)
    endif
    cldfra_bl1_old(kts:kte)=cldfra_bl1(kts:kte)
    qc_bl1_old(kts:kte)=qc_bl1(kts:kte)
    qi_bl1_old(kts:kte)=qi_bl1(kts:kte)
    cldfra_bl1 =zero
    qc_bl1     =zero
    qi_bl1     =zero
    dqc1       =zero
    dqi1       =zero
    dqs1       =zero
    dqni1      =zero
    dqnc1      =zero
    dqnwfa1    =zero
    dqnifa1    =zero
    dqnbca1    =zero
    dozone1    =zero
    
    edmf_a1    =zero
    edmf_w1    =zero
    edmf_qc1   =zero
    edmf_qv1   =zero
    edmf_thl1  =zero
    edmf_u1    =zero
    edmf_v1    =zero
    s_aw1      =zero
    s_awthl1   =zero
    s_awqt1    =zero
    s_awqv1    =zero
    s_awqc1    =zero
    s_awu1     =zero
    s_awv1     =zero
    s_awqke1   =zero
    s_awqsq1   =zero
    s_awqnc1   =zero
    s_awqni1   =zero
    s_awqnwfa1 =zero
    s_awqnifa1 =zero
    s_awqnbca1 =zero
    s_awchem1(kts:kte+1,1:nchem) = zero
    s_awscalars1(kts:kte+1,1:nscalars) = zero
    
    edmf_a_dd1 =zero
    edmf_w_dd1 =zero
    edmf_qc_dd1=zero
    sd_aw1     =zero
    sd_awthl1  =zero
    sd_awqt1   =zero
    sd_awqv1   =zero
    sd_awqc1   =zero
    sd_awqi1   =zero
    sd_awqnc1  =zero
    sd_awqni1  =zero
    sd_awqnwfa1=zero
    sd_awqnifa1=zero
    sd_awu1    =zero
    sd_awv1    =zero
    sd_awqke1  =zero
    sub_thl1   =zero
    sub_sqv1   =zero
    sub_u1     =zero
    sub_v1     =zero
    det_thl1   =zero
    det_sqv1   =zero
    det_sqc1   =zero
    det_u1     =zero
    det_v1     =zero

    zw1(kts)=zero
    do k = kts,kte
       zw1(k+1)=zw1(k)+dz1(k)
       qv1(k) = sqv1(k)/(one-sqv1(k))
       qc1(k) =	sqc1(k)/(one-sqv1(k))
       qi1(k) =	sqi1(k)/(one-sqv1(k))
       qs1(k) =	sqs1(k)/(one-sqv1(k))
       
       sqw1(k)= sqv1(k)+sqc1(k)+sqi1(k)
       thl1(k)= th1(k) - xlvcp/ex1(k)*sqc1(k) &
             &         - xlscp/ex1(k)*(sqi1(k))
       
       
       
       
       thv1(k)=th1(k)*(one+p608*sqv1(k) - (sqc1(k)+sqi1(k)))
    enddo 



    call get_pblh(kts,kte,pblh,thv1,qke1,ust,zw1,dz1,xland,kpbl)





    call scale_aware(dx,pblh,psig_bl,psig_shcu)

    sqcg= zero   
    cpm=cp*(one + 0.84_kind_phys*max(sqv1(kts),1e-8_kind_phys))
    exnerg=(ps/p1000mb)**rcp

    
    
    
    
    
    
    
    flqv = qfx/rho1(kts)
    flqc = zero 
    th_sfc = ts/ex1(kts)

    
    flq =flqv+flqc                   
    flt =hfx/(rho1(kts)*cpm )-xlvcp*flqc/ex1(kts)  
    fltv=flt + flqv*p608*th_sfc      

    
    rmol= -karman*gtr*fltv/max(ust**3,1.0e-6_kind_phys)
    rmolh=-karman*gtr*flt /max(ust**3,1.0e-6_kind_phys)
    zet = p5*dz1(kts)*rmol
    zet = max(zet, -10._kind_phys)
    zet = min(zet,  10._kind_phys)
    
    if (bl_mynn_stfunc == 0) then
       
       if ((xland-1.5) .ge. zero) then       
          if ( zet >= zero ) then
             
             pmz = one + 3.0_kind_phys * zet
             phh = one +  cphh_st * zet
          else
             pmz = one/    (one-cphm_unst*zet)**0.25 - zet
             phh = one/sqrt(one-cphh_unst*zet)
          end if
       else                                  
          if ( zet >= zero ) then
             
             pmz = 0.95_kind_phys + 3.5_kind_phys * zet
             phh = one + cphh_st       * zet
          else
             
             pmz = 0.95_kind_phys/(one-cphm_unst*zet)**0.25 - zet
             phh = one       /sqrt(one-cphh_unst*zet)
          end if
       endif
       phi_m = phim(zet)
    else
       
       phi_m = phim(zet)
       pmz   = phi_m - zet
       phh   = phih(zet)
    end if






    call mym_condensation (kts,kte,                   &
         &dx,dz1,zw1,xland,                           &
         &thl1,sqw1,sqv1,sqc1,sqi1,sqs1,              &
         &pres1,ex1,tsq1,qsq1,cov1,                   &
         &sh1,el1,bl_mynn_cloudpdf,                   &
         &qc_bl1,qi_bl1,cldfra_bl1,                   &
         &pblh,hfx,                                   &
         &vt1, vq1, th1, sgm1,                        &
         &bl_mynn_closure,                            &
         &spp_pbl, pattern_spp_pbl1                   )


    TKEprod_up = zero
    if (bl_mynn_edmf > 0) then
       
       call DMP_mf(i,j,                               &
            &kts,kte,delt,zw1,dz1,pres1,rho1,         &
            &bl_mynn_edmf_mom,                        &
            &bl_mynn_edmf_tke,                        &
            &bl_mynn_mixscalars,                      &
            &bl_mynn_mixaerosols,                     &
            &bl_mynn_mixnumcon,                       &
            &bl_mynn_closure,                         &
            &u1,v1,w1,th1,thl1,thv1,tk1,              &
            &sqw1,sqv1,sqc1,qke1,qsq1,                &
            &qnc1,qni1,qnwfa1,qnifa1,qnbca1,          &
            &ex1,vt1,vq1,sgm1,                        &
            &ust,flt,fltv,flq,flqv,                   &
            &pblh,kpbl,dx,                            &
            &xland,th_sfc,                            &
            
            
            
            &edmf_a1,edmf_w1,edmf_qt1,                &
            &edmf_thl1,edmf_ent1,edmf_qc1,            &
            &edmf_qv1,edmf_u1,edmf_v1,                &
            
            &s_aw1,s_awthl1,s_awqt1,                  &
            &s_awqv1,s_awqc1,                         &
            &s_awu1,s_awv1,s_awqke1,s_awqsq1,         &
            &s_awqnc1,s_awqni1,                       &
            &s_awqnwfa1,s_awqnifa1,s_awqnbca1,        &
            &sub_thl1,sub_sqv1,                       &
            &sub_u1,sub_v1,                           &
            &det_thl1,det_sqv1,det_sqc1,              &
            &det_u1,det_v1,                           &
            
            &nchem,chem1,s_awchem1,                   &
            &mix_chem,                                &
            &nscalars,scalars,s_awscalars1,           &
            &qc_bl1,cldfra_bl1,                       &
            &qc_bl1_old,cldfra_bl1_old,               &
            &FLAG_QC,FLAG_QI,                         &
            &FLAG_QNC,FLAG_QNI,                       &
            &FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA,        &
            &Psig_shcu,                               &
            &maxwidth,ktop_plume,                     &
            &maxmf,ztop_plume,excess_h,excess_q,      &
            &spp_pbl,pattern_spp_pbl1,                &
            &TKEprod_up,el1                           )
    endif

    
    
    
    
    tkeprod_dn     = zero
    maxtkeprod     = zero
    cldtop_cooling = zero
    ent_eff        = zero
    maxmf_dd       = zero
    maxwidth_dd    = zero
    if (bl_mynn_edmf_dd > 0) then
       call ddmp_mf(kts,kte,delt,dx,zw1,dz1,pres1,    &
            &u1,v1,th1,thl1,thv1,tk1,                 &
            &sqw1,sqv1,sqc1,sqi1,qnc1,qni1,           &
            &qnwfa1,qnifa1,                           &
            &qke1,rho1,ex1,                           &
            &qc_bl1,qi_bl1,cldfra_bl1,                &
            &ust,flt,flq,fltv,                        &
            &pblh,kpbl,                               &
            &edmf_a_dd1,edmf_w_dd1,edmf_qt_dd1,       &
            &edmf_thl_dd1,edmf_ent_dd1,               &
            &edmf_qc_dd1,                             &
            &sd_aw1,sd_awthl1,sd_awqt1,               &
            &sd_awqv1,sd_awqc1,sd_awqi1,              &
            &sd_awqnc1,sd_awqni1,                     &
            &sd_awqnwfa1,sd_awqnifa1,                 &
            &sd_awu1,sd_awv1,                         &
            &sd_awqke1,                               &
            &tkeprod_dn,el1,                          &
            &rthraten1,psig_bl,                       &
            &maxmf_dd,maxwidth_dd                     )
       
       
       
       call topdown_cloudrad(kts,kte,                 &
            &dz1,zw1,fltv,u1(kts),v1(kts),            &
            &xland,kpbl,pblh,                         &
            &sqc1,sqi1,sqw1,thl1,th1,                 &
            &ex1,pres1,rho1,thv1,                     &
            &cldfra_bl1,qc_bl1,qi_bl1,rthraten1,      &
            &tkeprod_dn,psig_bl,                      &
            maxtkeprod,cldtop_cooling,ent_eff         )
    endif

    
    
    delt2 = delt 

    
    do k=kts,kte
       qc_tot1(k) =max(qc_bl1(k),sqc1(k))
       qi_tot1(k) =max(qi_bl1(k),sqi1(k)+sqs1(k))
       thlv1(k)   =(th1(k) - xlvcp/ex1(k)*qc_tot1(k)  &
            &              - xlscp/ex1(k)*qi_tot1(k)) &
            &              * (one+p608*sqv1(k))
       thl_tot1(k)=th1(k)  - xlvcp/ex1(k)*qc_tot1(k)  &
            &              - xlscp/ex1(k)*qi_tot1(k)
       thv1(k)    =th1(k)*(one+p608*sqv1(k) - (qc_tot1(k)+qi_tot1(k)))
    enddo

    call mym_turbulence(                                 &
            &kts,kte,xland,bl_mynn_closure,ckmod,taue,   &
            &dz1, dx, zw1, pres1, ex1,                   &
            &u1, v1, thl1, thv1, thlv1,                  &
            &sqc1, sqw1,                                 &
            &qke1, tsq1, qsq1, cov1,                     &
            &vt1, vq1,                                   &
            &rmol, rmolh, flt, fltv, flq,                &
            &pblh, th1,                                  &
            &sh1,sm1,el1,                                &
            &Dfm1,Dfh1,Dfq1,                             &
            &Tcd1,Qcd1,Pdk1,                             &
            &Pdt1,Pdq1,Pdc1,                             &
            &qWT1,qSHEAR1,qBUOY1,qDISS1,                 &
            &tke_budget,                                 &
            &Psig_bl,Psig_shcu,                          &
            &cldfra_bl1,bl_mynn_mixlength,               &
            &bl_mynn_ess,                                &
            &edmf_w1,edmf_a1,                            &
            &edmf_w_dd1,edmf_a_dd1,                      &
            &TKEprod_dn,TKEprod_up,                      &
            &spp_pbl,pattern_spp_pbl1                    )




    call mym_predict(kts,kte,bl_mynn_closure,            &
            &delt2, dz1,                                 &
            &ust, flt, flq, pmz, phh,                    &
            &el1, dfq1, rho1, pdk1, pdt1, pdq1, pdc1,    &
            &qke1, tsq1, qsq1, cov1,                     &
            &s_aw1, s_awqke1, s_awqsq1,                  &
            &bl_mynn_edmf, bl_mynn_edmf_tke,             &
            &qWT1, qDISS1, tke_budget, upwind            )



    do k=kts,kte-1
       diss_heat1(k) = min(max((qke1(k)**1.5_kind_phys)/(b1*max(p5*(el1(k)+el1(k+1)),one))/cp, zero),0.002_kind_phys)
       diss_heat1(k) = diss_heat1(k) * exp(-10000._kind_phys/max(pres1(k),one)) 
    enddo
    diss_heat1(kte) = zero



    call mynn_tendencies(kts,kte,i,                      &
            &delt, dz1, zw1, xland, pblh, rho1,          &
            &u1, v1, th1, tk1, qv1,                      &
            &qc1, qi1, kzero1, qnc1, qni1,               & 
            &ps, pres1, ex1, thl1,                       &
            &sqv1, sqc1, sqi1, kzero1, sqw1,             & 
            &thl_tot1, qc_tot1, qi_tot1,                 &
            &qnwfa1, qnifa1, qnbca1, ozone1,             &
            &ust,flt,flq,flqv,flqc,                      &
            &wspd,uoce,voce,                             &
            &tsq1, qsq1, cov1,                           &
            &tcd1, qcd1,                                 &
            &dfm1, dfh1, dfq1,                           &
            &Du1, Dv1, Dth1, Dqv1,                       &
            &Dqc1, Dqi1, Dqs1, Dqnc1, Dqni1,             &
            &Dqnwfa1, Dqnifa1, Dqnbca1,                  &
            &Dozone1,                                    &
            &diss_heat1,                                 &
            
            &s_aw1,s_awthl1,s_awqt1,                     &
            &s_awqv1,s_awqc1,s_awu1,s_awv1,              &
            &s_awqnc1,s_awqni1,                          &
            &s_awqnwfa1,s_awqnifa1,s_awqnbca1,           &
            &sd_aw1,sd_awthl1,sd_awqt1,                  &
            &sd_awqv1,sd_awqc1,sd_awqi1,                 &
            &sd_awqnc1,sd_awqni1,                        &
            &sd_awqnwfa1,sd_awqnifa1,                    &
            &sd_awu1,sd_awv1,                            &
            &sub_thl1,sub_sqv1,                          &
            &sub_u1,sub_v1,                              &
            &det_thl1,det_sqv1,det_sqc1,                 &
            &det_u1,det_v1,                              &
            &FLAG_QC,FLAG_QI,FLAG_QNC,                   &
            &FLAG_QNI,FLAG_QS,                           &
            &FLAG_QNWFA,FLAG_QNIFA,                      &
            &FLAG_QNBCA,FLAG_OZONE,                      &
            &cldfra_bl1,                                 &
            &bl_mynn_cloudmix,                           &
            &bl_mynn_mixqt,                              &
            &bl_mynn_edmf,                               &
            &bl_mynn_edmf_mom,                           &
            &bl_mynn_mixscalars,                         &
            &bl_mynn_mixaerosols,                        &
            &bl_mynn_mixnumcon,                          &
            &upwind                                      )

    if ( mix_chem ) then
          call mynn_mix_chem(kts,kte,i,                  &
               &delt, dz1, pblh,                         &
               &nchem, ndvel,                            &
               &chem1, vdep, settle1,                    &
               &rho1, flt,                               &
               &tcd1, qcd1,                              &
               &dfh1,                                    &
               &s_aw1,s_awchem1,                         &
               &emis_ant_no,                             &
               &frp,                                     &
               &enh_mix,                                 &
               &bl_mynn_edmf, upwind                     )
       do ic = 1,nchem
          do k = kts,kte
             chem1(k,ic) = max(1.e-12, chem1(k,ic))
          enddo
       enddo
    endif

    if ( bl_mynn_mixscalars .eq. 1) then
       call mynn_mix_scalars(kts,kte,i,               &
            &delt, dz1,                               &
            &nscalars, scalars,                       &
            &rho1, flt,                               &
            &tcd1, qcd1,                              &
            &dfh1,                                    &
            &s_aw1,s_awscalars1,                      &
            &bl_mynn_edmf, upwind                     )
    endif
       
    
    call retrieve_exchange_coeffs(kts,kte,               &
         dfm1, dfh1, dz1, km1, kh1                       )

    
    if (tke_budget .eq. 1) then
       
       
       k=kts
       qSHEAR1(k)   = 4.*(ust**3*phi_m/(karman*dz1(k)))-qSHEAR1(k+1) 
       qBUOY1(k)    = 4.*(-ust**3*zet/(karman*dz1(k)))-qBUOY1(k+1) 
       
       do k = kts,kte-1
          dummy1(k) = p5*(qSHEAR1(k)+qSHEAR1(k+1)) 
          dummy2(k) = p5*(qBUOY1(k)+qBUOY1(k+1)) 
          dqke1(k)  = p5*(qke1(k)-dqke1(k))/delt
       enddo
       qSHEAR1      = dummy1
       qBUOY1       = dummy2
       
       k=kte
       qSHEAR1(k)   = zero
       qBUOY1(k)    = zero
       qWT1(k)      = zero
       qDISS1(k)    = zero
       dqke1(k)     = zero
    endif

    
    if (bl_mynn_output > 0) then 
       
       if (bl_mynn_output == 2 .and. bl_mynn_edmf_dd == 1) then
          edmf_a1(kts:kte)   =edmf_a_dd1(kts:kte)
          edmf_w1(kts:kte)   =edmf_w_dd1(kts:kte)
          edmf_qt1(kts:kte)  =edmf_qt_dd1(kts:kte)
          edmf_thl1(kts:kte) =edmf_thl_dd1(kts:kte)
          edmf_ent1(kts:kte) =edmf_ent_dd1(kts:kte)
          edmf_qc1(kts:kte)  =edmf_qc_dd1(kts:kte)
       endif
    endif

    
    if ( debug_code .and. (i .eq. idbg)) THEN
       if ( ABS(QFX)>.001)print*,&
          "SUSPICIOUS VALUES AT: i=",i," QFX=",QFX
       if ( ABS(HFX)>1100.)print*,&
          "SUSPICIOUS VALUES AT: i=",i," HFX=",HFX
       do k = kts,kte
          IF ( sh1(k) < 0. .OR. sh1(k)> 200.)print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," sh=",sh1(k)
          IF ( ABS(vt1(k)) > 2.0 )print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," vt=",vt1(k)
          IF ( ABS(vq1(k)) > 7000.)print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," vq=",vq1(k)
          IF ( qke1(k) < -1. .OR. qke1(k)> 200.)print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," qke=",qke1(k)
          IF ( el1(k) < 0. .OR. el1(k)> 1500.)print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," el1=",el1(k)
          IF ( km1(k) < 0. .OR. km1(k)> 2000.)print*,&
             "SUSPICIOUS VALUES AT: i,k=",i,k," km=",km1(k)
          IF (cldfra_bl1(k) < zero .OR. cldfra_bl1(k)> one)THEN
             PRINT*,"SUSPICIOUS VALUES: CLDFRA_BL=",cldfra_bl1(k),&
                                          " qc_bl=",qc_bl1(k)
          endif
       enddo 
    endif

    
    qke_adv1(kts:kte)=qke1(kts:kte)


  END SUBROUTINE mynnedmf




























































  SUBROUTINE  mym_initialize (                                & 
       &            kts,kte,ckmod,taue,xland,pblh,            &
       &            dz, dx, zw, p, exner,                     &
       &            u, v, thl, qw,                            &

       &            theta, thv, thlv, sh, sm,                 &
       &            ust, rmol, rmolh, el,                     &
       &            Qke, Tsq, Qsq, Cov,                       &
       &            Psig_bl, cldfra_bl1,                      &
       &            bl_mynn_mixlength,                        &
       &            bl_mynn_ess,                              &
       &            edmf_w1,edmf_a1,                          &
       &            edmf_w_dd1,edmf_a_dd1,                    &
       &            INITIALIZE_QKE,                           &
       &            spp_pbl,pattern_spp_pbl1                  )


    use module_bl_mynnedmf_common, only: b1,b2,gtr,karman,    &
         qkemin,zero,one,two,three,hundred,p01,p333,p5,p666,  &
         kind_phys
    
    integer, intent(in)           :: kts,kte
    integer, intent(in)           :: bl_mynn_mixlength
    integer, intent(in)           :: bl_mynn_ess
    logical, intent(in)           :: INITIALIZE_QKE

    real(kind_phys), intent(in)   :: rmol, rmolh, Psig_bl, xland, pblh
    real(kind_phys), intent(in)   :: dx, ust, ckmod, taue
    real(kind_phys), dimension(kts:kte),   intent(in) :: dz
    real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
    real(kind_phys), dimension(kts:kte),   intent(in) :: u,v,thl,&
         &thlv,qw,cldfra_bl1,edmf_w1,edmf_a1,edmf_w_dd1,edmf_a_dd1,&
         &p,exner
    real(kind_phys), dimension(kts:kte),   intent(inout) :: tsq,qsq,cov
    real(kind_phys), dimension(kts:kte),   intent(inout) :: el,qke
    real(kind_phys), dimension(kts:kte) ::                       &
         &ql,pdk,pdt,pdq,pdc,dtl,dqw,dtv,                        &
         &gm,gh,sm,sh,qkw,qtw,vt,vq,qpe
    integer :: k,l,lmax
    real(kind_phys):: phm,vkz,elq,elv,b1l,b2l,pmz=1.,phh=1.,     &
         &flt=0.,fltv=0.,flq=0.,tmpq
    real(kind_phys):: ustm
    real(kind_phys), dimension(kts:kte) :: theta,thv
    real(kind_phys), dimension(kts:kte) :: pattern_spp_pbl1
    integer ::spp_pbl


    DO k = kts,kte
       ql(k) = zero
       vt(k) = zero
       vq(k) = zero
       qkw(k)= zero
       qtw(k)= zero
    END DO
    ustm = max(p01, ust)


    CALL mym_level2 ( kts,kte,                      &
         &            ckmod, bl_mynn_ess,           &
         &            zw, dz, xland, pblh,          &
         &            u, v, thl, thv, thlv,         &
         &            theta, p, exner,              &
         &            qke, cldfra_bl1,              &
         &            edmf_a1, edmf_w1,             &
         &            qw, ql, vt, vq,               &
         &            dtl, dqw, dtv, gm, gh, sm, sh )



    el(kts) = zero
    IF (INITIALIZE_QKE) THEN
       
       qke(kts) = 1.5_kind_phys * ustm**2 * ( b1*pmz )**p666
       DO k = kts+1,kte
          
          
          qke(k)=qke(kts)*MAX((ust*700._kind_phys - zw(k))/(ustm*700._kind_phys), p01)
       ENDDO
    ENDIF

    phm      = phh*b2 / ( b1*pmz )**p333
    tsq(kts) = min(three,          phm*( max(zero,flt)/ustm )**2)
    qsq(kts) = min(5e-6_kind_phys, phm*( max(zero,flq)/ustm )**2)
    cov(kts) = phm*( flt/ustm )*( flq/ustm )
    qpe(kts) = min(three, max(zero, two * tsq(kts) * gtr**2 * taue**2))

    DO k = kts+1,kte
       vkz = karman*zw(k)
       el(k) = vkz/( one + vkz/hundred )


       tsq(k) = zero
       qsq(k) = zero
       cov(k) = zero
       qpe(k) = zero
    END DO



    lmax = 5

    DO l = 1,lmax


       CALL mym_length (                          &
            &            kts,kte,xland,           &
            &            dz, dx, zw,              &
            &            rmol, rmolh,             &
            &            flt, fltv, flq,          &
            &            vt, vq,                  &
            &            u, v, qke, qpe,          &
            &            dtv,                     &
            &            el,                      &
            &            pblh,theta,              &
            &            qkw, qtw,                &
            &            Psig_bl,cldfra_bl1,      &
            &            bl_mynn_mixlength,       &
            &            edmf_w1,edmf_a1,         &
            &            edmf_w_dd1,edmf_a_dd1    )

       DO k = kts+1,kte
          elq    = el(k)*qkw(k)
          pdk(k) = elq*( sm(k)*gm(k) + &
               &         sh(k)*gh(k) )
          pdt(k) = elq*  sh(k)*dtl(k)**2
          pdq(k) = elq*  sh(k)*dqw(k)**2
          pdc(k) = elq*  sh(k)*dtl(k)*dqw(k)
       END DO


       vkz = karman*p5*dz(kts)
       elv = p5*( el(kts+1)+el(kts) ) /  vkz
       IF (INITIALIZE_QKE)THEN 
          
          qke(kts) = one * ustm**2 * ( b1*pmz*elv    )**p666 
       ENDIF

       phm      = phh*b2 / ( b1*pmz/elv**2 )**p333
       tsq(kts) = min(three,          phm*( max(zero,flt)/ustm )**2)
       qsq(kts) = min(5e-6_kind_phys, phm*( max(zero,flq)/ustm )**2)
       cov(kts) = phm*( flt/ustm )*( flq/ustm )
       
       DO k = kts+1,kte-1
          b1l = b1*0.25*( el(k+1)+el(k) )
          
          
          tmpq=MIN(MAX(b1l*( pdk(k+1)+pdk(k) ),qkemin),125._kind_phys)

          IF (INITIALIZE_QKE)THEN
             qke(k) = tmpq**p666
          ENDIF

          IF ( qke(k) .LE. zero ) THEN
             b2l = zero
          ELSE
             b2l = b2*( b1l/b1 ) / SQRT( qke(k) )
          END IF

          tsq(k) = min(three,          max(zero, b2l*( pdt(k+1)+pdt(k) )))
          qsq(k) = min(5e-6_kind_phys, max(zero, b2l*( pdq(k+1)+pdq(k) )))
          cov(k) = b2l*( pdc(k+1)+pdc(k) )
       END DO

    END DO

    IF (INITIALIZE_QKE)THEN
       qke(kts)=p5*(qke(kts)+qke(kts+1))
       qke(kte)=qke(kte-1)
    ENDIF
    tsq(kte)=tsq(kte-1)
    qsq(kte)=qsq(kte-1)
    cov(kte)=cov(kte-1)

  END SUBROUTINE mym_initialize

  









































  SUBROUTINE  mym_level2 (kts,kte,                &
       &            ckmod, bl_mynn_ess,           &
       &            zw, dz, xland, pblh,          &
       &            u, v, thl, thv, thlv,         &
       &            th, p, exner,                 &
       &            qke, cldfra,                  &
       &            edmf_a, edmf_w,               &
       &            qw, ql, vt, vq,               &
       &            dtl, dqw, dtv, gm, gh, sm, sh )


 use module_bl_mynnedmf_common, only: tv0,gtr,cp, &
      qkemin,a1,a2,b1,c1,c2,c5,g1,g2,zero,one,two,&
      three,four,ten,twenty,hundred,p1,p2,p25,p3, &
      p5,p95,kind_phys
    
 integer, intent(in)   :: kts,kte


 real(kind_phys), intent(in)::xland,pblh,ckmod
 integer, intent(in)        ::bl_mynn_ess
 real(kind_phys), dimension(kts:kte), intent(in)  :: dz
 real(kind_phys), dimension(kts:kte), intent(in)  :: u,v, &
      &thl,qw,ql,vt,vq,thv,thlv,th,p,exner,edmf_a,edmf_w, &
      &qke,cldfra
 real(kind_phys), dimension(kts:kte+1), intent(in):: zw
 real(kind_phys), dimension(kts:kte), intent(out) ::      &
      &dtl,dqw,dtv,gm,gh,sm,sh
 real(kind_phys), dimension(kts:kte):: ri
 integer :: k
 real(kind_phys):: rfc,f1,f2,rf1,rf2,smc,shc,             &
      &ri1,ri2,ri3,ri4,duz,dtz,dqz,vtt,vqq,dtq,dzk,       &
      &afk,abk,rf
 real(kind_phys):: a2fac,thv1,thv0,eth0,eth1,lambda,qsat, &
      &xl,tabs,clam0,clam,wt,mfi,qkei,cldfrai,ugrid,      &
      &cfac,uonset,taper
 real(kind_phys), parameter:: thvp = 0.0 







 dtl(kts)=zero
 dqw(kts)=zero
 dtv(kts)=zero
 gm(kts)=zero
 gh(kts)=zero
 sm(kts)=zero
 sh(kts)=zero
 ri(kts)=zero

 select case (bl_mynn_ess)  

    case (0) 

       do k = kts+1,kte
          dzk = p5 *( dz(k)+dz(k-1) )
          afk = dz(k)/( dz(k)+dz(k-1) )
          abk = one -afk
          duz = ( u(k)-u(k-1) )**2 +( v(k)-v(k-1) )**2
          duz =   duz                    /dzk**2
          dtz = ( thl(k)-thl(k-1) )/( dzk )
          dqz = ( qw(k)-qw(k-1) )/( dzk )

          vtt =  one +vt(k)*abk +vt(k-1)*afk  
          vqq =  tv0 +vq(k)*abk +vq(k-1)*afk  
          dtq =  vtt*dtz +vqq*dqz

          dtl(k) =  dtz
          dqw(k) =  dqz
          dtv(k) =  dtq

          gm(k)  =  duz
          gh(k)  = -dtq*gtr

          
          ri(k) = -gh(k)/MAX( duz, 1.0e-10_kind_phys )
       enddo
       
    case (1) 
       
       
       
       
       
       
       
       
       ugrid  = sqrt(u(kts)**2 + v(kts)**2)
       uonset = 15._kind_phys
       taper  = one - p95*min(one, max(zero, ugrid - uonset)/twenty) 
       if ((xland-1.5).GE.zero) then 
          clam0 = 1.8_kind_phys * taper
       else                          
          
          
          clam0 = 0.08_kind_phys
       endif

       do k = kts+1,kte
          dzk   = p5 *  ( dz(k)+dz(k-1) )
          afk   = dz(k)/( dz(k)+dz(k-1) )
          abk   = one -afk
          duz   = ( u(k)-u(k-1) )**2 + ( v(k)-v(k-1) )**2
          duz   =   duz               / dzk**2
          dtz   = ( thl(k)-thl(k-1) ) / dzk
          dqz   = ( qw(k) -qw(k-1)  ) / dzk

          
          wt    = min(one, max(zero, zw(k)-(pblh+hundred))/max(400._kind_phys, p3*pblh)) 
          clam  = clam0*(one-wt) + p2*wt
          
          clam  = clam * min(one, max(zero, zw(k)-twenty)/200.0_kind_phys)
          
          
          
          thv1  = thvp*thv(k)   + (one-thvp)*thlv(k)
          thv0  = thvp*thv(k-1) + (one-thvp)*thlv(k-1)
          
          
          
          tabs  = th(k)*exner(k)
          xl    = xl_blend(tabs)            
          qsat  = qsat_blend(tabs,p(k))     
          eth1  = th(k)*exp(xl*qsat/(cp*tabs))
          
          tabs  = th(k-1)*exner(k-1)
          xl    = xl_blend(tabs)            
          qsat  = qsat_blend(tabs,p(k-1))   
          eth0  = th(k-1)*exp(xl*qsat/(cp*tabs))

          
          mfi   = max(zero, min(p2, (p5*(edmf_a(k)+edmf_a(k-1))) * (p5*(edmf_w(k)+edmf_w(k-1)))))

          
          
          qkei  = max(qkemin, p5*(qke(k)+qke(k-1)))  
          qkei  = min(p5, sqrt(p25*qkei))*p2
          
          
          cldfrai =  p5*(cldfra(k)+cldfra(k-1)) 
          
          cfac    = max( 0.01_kind_phys, min(2.7_kind_phys * (cldfrai - one)**2, one))
          
          cldfrai = min(p1, cldfrai*cfac)*ten
          
          
          
          lambda= clam * max(mfi, qkei) * cldfrai
          
          dtq   = ( thv1-thv0 )/dzk + lambda*min(zero, ( eth1-eth0 )/dzk )
          

          dtl(k)=  dtz
          dqw(k)=  dqz
          dtv(k)=  dtq

          gm(k) =  duz
          gh(k) = -dtq*gtr

          
          ri(k) = -gh(k)/MAX( duz, 1.0e-10_kind_phys )
       enddo
       
    end select
    
    do k = kts+1,kte
       
       a2fac = one/(one + ckmod*MAX(ri(k),zero))

       rfc = g1/( g1+g2 )
       f1  = b1*( g1-c1 ) +three*a2*a2fac *( one    -c2 )*( one-c5 ) &
           &              +two  *a1*( three-two*c2 )
       f2  = b1*( g1+g2 ) -three*a1*( one      -c2 )
       rf1 = b1*( g1-c1 )/f1
       rf2 = b1*  g1     /f2
       smc = a1   /(a2*a2fac)*  f1/f2
       shc = three*(a2*a2fac)*( g1+g2 )

       ri1 = p5/smc
       ri2 = rf1*smc
       ri3 = four*rf2*smc -two*ri2
       ri4 = ri2**2

       
       rf = MIN( ri1*( ri(k) + ri2-SQRT(ri(k)**2 - ri3*ri(k) + ri4) ), rfc )

       sh(k) = shc*( rfc-rf )/( one-rf )
       sm(k) = smc*( rf1-rf )/( rf2-rf ) * sh(k)
    enddo


  END SUBROUTINE mym_level2



















  SUBROUTINE  mym_length (                     & 
    &            kts,kte,xland,                &
    &            dz, dx, zw,                   &
    &            rmol, rmolh, flt, fltv, flq,  &
    &            vt, vq,                       &
    &            u1, v1, qke, qpe,             &
    &            dtv,                          &
    &            el,                           &
    &            pblh, theta, qkw, qtw,        &
    &            Psig_bl, cldfra_bl1,          &
    &            bl_mynn_mixlength,            &
    &            edmf_w1,edmf_a1,              &
    &            edmf_w_dd1,edmf_a_dd1         )
    

    use module_bl_mynnedmf_common, only:       &
         qkemin,qmin,tv0,gtr,karman,grav,zmax, &
         zero,one,two,four,five,ten,twenty,    &
         thirty,forty,fifty,hundred,           &
         p1,p2,p25,p3,p333,p4,p5,kind_phys
    
    integer, intent(in)   :: kts,kte


    integer, intent(in)   :: bl_mynn_mixlength
    real(kind_phys), dimension(kts:kte),   intent(in) :: dz
    real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
    real(kind_phys), intent(in) :: rmol,rmolh,flt,fltv,flq,Psig_bl,xland
    real(kind_phys), intent(in) :: dx,pblh
    real(kind_phys), dimension(kts:kte), intent(in)   :: u1,v1,  &
         &qke,qpe,vt,vq,cldfra_bl1,edmf_w1,edmf_a1,edmf_w_dd1,edmf_a_dd1
    real(kind_phys), dimension(kts:kte), intent(out)  :: el
    real(kind_phys), dimension(kts:kte), intent(inout):: qkw,qtw
    real(kind_phys), dimension(kts:kte), intent(in)   :: dtv
    real(kind_phys):: elt,vsc
    real(kind_phys), dimension(kts:kte), intent(in) :: theta
    real(kind_phys), dimension(kts:kte) :: qtke,elBLmin,elBLavg,thetaw
    real(kind_phys):: wt,wt2,pblh2,h1,h2,hs,elBLmin0,elBLavg0,cldavg

    
    
    real(kind_phys):: cns,   &   
            alp1,            &   
            alp2,            &   
            alp3,            &   
            alp4,            &   
            alp5,            &   
            alp6                 

    
    
    
    
    real(kind_phys), parameter :: minpblh     = 250.  
    real(kind_phys), parameter :: maxdz       = 750.  
                                     
                                     
    real(kind_phys), parameter :: mindz       = 250.  

    
    real(kind_phys), parameter :: ZSLH        = 100.  
    real(kind_phys), parameter :: CSL         = 2.    
    real(kind_phys), parameter :: qkw_elb_min = 0.05  

    integer :: i,j,k
    real(kind_phys):: afk,abk,zwk,zwk1,dzk,qdz,vflx,bv,tau_cloud,      &
           & wstar,elb,els,elf,el_stab,el_mf,el_stab_mf,elb_mf,elt_max,&
           & pblh_plus_ent,uonset,ugrid,wt_u1,wt_u2,el_les,qkw_mf,     &
           & z_m,el_unstab,els1,alp3z,cpblh,wt_dx
    real(kind_phys), parameter :: ctau = 1000. 




    SELECT CASE(bl_mynn_mixlength)

      CASE (0) 

        cns  = 2.7_kind_phys
        alp1 = 0.23_kind_phys
        alp2 = one
        alp3 = five
        alp4 = hundred
        alp5 = p3

        
        pblh2= pblh+dz(kts)          
        h1   = max(p3*pblh2,mindz)
        h1   = min(h1,maxdz)         
        h2   = h1*p5                

        qkw(kts) = SQRT(MAX(qke(kts), qkemin))
        DO k = kts+1,kte
           afk = dz(k)/( dz(k)+dz(k-1) )
           abk = one -afk
           qkw(k) = SQRT(MAX(qke(k)*abk+qke(k-1)*afk, qkemin))
        END DO
        qtw = qkw

        elt = 1.0e-5_kind_phys
        vsc = 1.0e-5_kind_phys

        
        k   = kts+1
        zwk = zw(k)
        DO WHILE (zwk .LE. pblh2+h1)
           dzk = p5*( dz(k)+dz(k-1) )
           qdz = MAX( qkw(k)-qmin, 0.03 )*dzk
           elt = elt +qdz*zwk
           vsc = vsc +qdz
           k   = k+1
           zwk = zw(k)
        END DO

        elt  = alp1*elt/vsc
        vflx = ( vt(kts)+one )*flt +( vq(kts)+tv0 )*flq
        vsc  = ( gtr*elt*MAX( vflx, zero ) )**p333

        
        el(kts) = zero
        zwk1    = zw(kts+1)

        DO k = kts+1,kte
           zwk = zw(k)              

           
           IF ( dtv(k) .GT. zero ) THEN
              bv  = SQRT( gtr*dtv(k) )
              elb = alp2*qkw(k) / bv &
                  &       *( one + alp3/alp2*&
                  &SQRT( vsc/( bv*elt ) ) )
              elf = alp2 * qkw(k)/bv

           ELSE
              elb = 1.0e10_kind_phys
              elf = elb
           ENDIF

           
           IF ( rmol .GT. 0.0 ) THEN
              els  = karman*zwk/(one+cns*MIN( zwk*rmol, zmax ))
           ELSE
              els  = karman*zwk*( one - alp4* zwk*rmol )**p2
           END IF

           
           
           

           wt=p5*TANH((zwk - (pblh2+h1))/h2) + p5

           el(k) = MIN(elb/( elb/elt+elb/els+one ),elf)

        END DO

      CASE (1) 

        
        ugrid = sqrt(u1(kts)**2 + v1(kts)**2)
        uonset= 15._kind_phys
        wt_u1 = one - p2*min(one, max(zero, ugrid - uonset)/forty) 
        wt_u2 = one - p5*min(one, max(zero, ugrid - uonset)/forty) 
        
        wt_dx = one - min(one, (max(zero, dx-4000._kind_phys)/12000._kind_phys))
        
        cns   = 3.5_kind_phys
        alp1  = 0.23_kind_phys*wt_dx + (one-wt_dx)*0.24_kind_phys
        alp2  = p3*wt_dx             + (one-wt_dx)*0.40_kind_phys
        alp3  = five * wt_u2 
        if ((xland-1.5).GE.zero) then 
           alp4  = twenty * wt_u2
        else
           alp4  = 15.0_kind_phys
        endif
        alp5  = p3
        alp6  = fifty

        
        pblh2 = max(pblh,200._kind_phys)
        h1    = max(p3*pblh2,300._kind_phys)
        h1    = min(h1,600._kind_phys)   
        h2    = h1*p5                    

        qtke(kts)  =max(p5*qke(kts), p5*qkemin) 
        thetaw(kts)=theta(kts)            
        qkw(kts)   =sqrt(max(qke(kts)         , qkemin))
        qtw(kts)   =sqrt(max(qke(kts)+qpe(kts), qkemin))
        DO k = kts+1,kte
           afk      = dz(k)/( dz(k)+dz(k-1) )
           abk      = one -afk
           qkw(k)   = sqrt(max(qke(k)*abk+qke(k-1)*afk, qkemin))
           qtw(k)   = sqrt(max((qke(k)+qpe(k))*abk+(qke(k-1)+qpe(k-1))*afk, qkemin))
           qtke(k)  = max(p5*(qkw(k)**2), 0.005_kind_phys) 
           thetaw(k)= theta(k)*abk + theta(k-1)*afk
        END DO

        elt = 1.0e-5_kind_phys
        vsc = 1.0e-5_kind_phys

        
        k   = kts+1
        zwk = zw(k)
        DO WHILE (zwk .LE. pblh2+max(200.,min(p3*pblh2,600._kind_phys)))
           dzk = p5*( dz(k)+dz(k-1) )
           qdz = min(max( qkw(k), 0.01_kind_phys ), 30.0_kind_phys)*dzk
           elt = elt +qdz*zwk
           vsc = vsc +qdz
           k   = k+1
           zwk = zw(k)
        END DO

        if ((xland-1.5).GE.zero) then 
           elt_max=350.+100.*min(one, max(zero, ugrid - twenty)/thirty)
        else
           elt_max=400._kind_phys
        endif
        elt = MIN( MAX( alp1*elt/vsc, 8._kind_phys), elt_max)
        
        
        vflx= fltv-0.005_kind_phys
        vsc = ( gtr*elt*MAX( vflx, zero ) )**p333

        
        el(kts) = zero
        zwk1    = zw(kts+1)         

        
        CALL boulac_length(kts,kte,zw,dz,qtke,thetaw,elBLmin,elBLavg)

        DO k = kts+1,kte
           zwk    = zw(k)          
           qkw_mf = max((p5*((edmf_a1(k)+edmf_a1(k-1))))*(p5*((edmf_w1(k)+edmf_w1(k-1)))), &
                  & abs(edmf_a_dd1(k-1)*edmf_w_dd1(k-1)))
           
           cpblh  = min((zwk + p25*pblh2)/pblh2, one)
           qkw_mf = cpblh * qkw_mf
           alp3z  = cpblh * alp3
           
           IF ( dtv(k) .GT. 0.0 ) THEN
              bv     = max( sqrt( gtr*dtv(k) ), 0.001_kind_phys)
              elb_mf = alp6*qkw_mf/bv
              elb_mf = elb_mf / (one + (elb_mf/100._kind_phys))
              elb    = alp2*(max(qkw(k),qkw_elb_min))/bv    &
                   &  *( one + alp3z*SQRT( vsc/(bv*elt) ) )
              elb    = max(elb, elb_mf)
              elb    = MIN(elb, zwk)
              elf    = one * max(qkw(k), qkw_elb_min)/bv
              elf    = max(elf, elb_mf)
              elBLavg(k) = MAX(alp5*elBLavg(k), elb_mf)
           ELSE
              elb        = 1.0e10_kind_phys
              elf        = elb
              elBLavg(k) = p4*elBLavg(k)
           ENDIF

           z_m = MAX(karman, zwk - four)
           
           IF ( rmol .GT. 0.0 ) THEN
              els  = karman*zwk/(one+cns*MIN( zwk*rmol, zmax ))
              els1 = karman*z_m/(one+cns*MIN( zwk*rmol, zmax ))
           ELSE
              els  = karman*zwk*( one - alp4* zwk*rmol)**p2
              els1 = karman*z_m*( one - alp4* zwk*rmol)**p2
           END IF

           
           wt =p5*TANH((zwk - (pblh2+h1))/h2) + p5
           
           
           
           
           
           
           
           el_unstab = els/(one + (els1/elt))
           el(k)     = min(el_unstab, elb)
           el(k)     = min(el(k), elf)  
           if ((xland-1.5).GE.zero) then 
              el(k)  = el(k)*wt_u1
           endif
           el(k)     = el(k)*(one-wt) + elBLavg(k)*wt

           
           
           
           
           

        END DO

     CASE (2) 

        Uonset = 3.5_kind_phys + dz(kts)*p1
        Ugrid  = sqrt(u1(kts)**2 + v1(kts)**2)
        cns    = 3.5_kind_phys 
        alp1   = 0.22_kind_phys
        alp2   = p3
        alp3   = 2.5_kind_phys
        alp4   = five
        alp5   = alp2  
        alp6   = fifty 

        
        
        pblh2=MAX(pblh, 200._kind_phys)
        
        
        h1=MAX(p3*pblh2,300._kind_phys)
        h1=MIN(h1,600._kind_phys)
        h2=h1*p5                

        qtke(kts)=MAX(p5*qke(kts), p5*qkemin) 
        qkw(kts) = SQRT(MAX(qke(kts)         , qkemin))
        qtw(kts) = SQRT(MAX(qke(kts)+qpe(kts), qkemin))
        DO k = kts+1,kte
           afk    = dz(k)/( dz(k)+dz(k-1) )
           abk    = one -afk
           qkw(k) = SQRT(MAX(qke(k)*abk+qke(k-1)*afk                    , qkemin))
           qtw(k) = SQRT(MAX((qke(k)+qpe(k))*abk+(qke(k-1)+qpe(k-1))*afk, qkemin))
           qtke(k)= p5*qkw(k)**2  
        END DO

        elt = 1.0e-5
        vsc = 1.0e-5

        
        PBLH_PLUS_ENT = MAX(pblh+h1, 100.)
        k = kts+1
        zwk = zw(k)
        DO WHILE (zwk .LE. PBLH_PLUS_ENT)
           dzk = p5*( dz(k)+dz(k-1) )
           qdz = min(max( qkw(k)-qmin, 0.03 ), 30.0)*dzk
           elt = elt +qdz*zwk
           vsc = vsc +qdz
           k   = k+1
           zwk = zw(k)
        END DO

        elt = MIN( MAX(alp1*elt/vsc, ten), 400.)
        
        
        vflx = fltv
        vsc  = ( gtr*elt*MAX( vflx, zero ) )**p333

        
        el(kts) = zero
        zwk1    = zw(kts+1)

        DO k = kts+1,kte
           zwk = zw(k)              
           dzk = p5*( dz(k)+dz(k-1) )
           cldavg = p5*(cldfra_bl1(k-1)+cldfra_bl1(k))
           qkw_mf = max((p5*(edmf_a1(k)+edmf_a1(k-1)))*(p5*(edmf_w1(k)+edmf_w1(k-1))), &
                  & abs(edmf_a_dd1(k-1)*edmf_w_dd1(k-1)))

           
           IF ( dtv(k) .GT. 0.0 ) THEN
              
              bv  = MAX( SQRT( gtr*dtv(k) ), 0.001)  
              
              elb_mf = MAX(alp2*qkw(k),                    &
                  &        alp6*qkw_mf) / bv               &
                  &  *( one + alp3*SQRT( vsc/( bv*elt ) ) )
              elb = MIN(MAX(alp5*qkw(k), alp6*qkw_mf)/bv, zwk)

              
              wstar = 1.25*(gtr*pblh*MAX(vflx,1.0e-4))**p333
              tau_cloud = MIN(MAX(ctau * wstar/grav, 30.), 150.)
              
              wt=p5*TANH((zwk - (pblh2+h1))/h2) + p5
              tau_cloud = tau_cloud*(1.-wt) + fifty*wt
              elf = MIN(MAX(tau_cloud*SQRT(MIN(qtke(k),forty)), &
                  &         alp6*qkw_mf/bv), zwk)

              
              
              
              
              
              

           ELSE
              
              
              
              
              
              
              
              
              
              
              
              wstar     = 1.25*(gtr*pblh*MAX(vflx,1.0e-4))**p333
              tau_cloud = MIN(MAX(ctau * wstar/grav, fifty), 200.)
              
              wt        = p5*TANH((zwk - (pblh2+h1))/h2) + p5
              
              tau_cloud = tau_cloud*(one-wt) + MAX(hundred,dzk*p25)*wt

              elb       = MIN(tau_cloud*SQRT(MIN(qtke(k),40.)), zwk)
              
              elf       = elb 
              elb_mf    = elb
         END IF
         elf    = elf/(one + (elf/800.))  
         elb_mf = MAX(elb_mf, 0.01) 

         
         IF ( rmol .GT. 0.0 ) THEN
            els  = karman*zwk/(one+cns*MIN( zwk*rmol, zmax ))
         ELSE
            els  =  karman*zwk*( one - alp4* zwk*rmol)**p2
         END IF

         
         wt=p5*TANH((zwk - (pblh2+h1))/h2) + p5

         
         el(k) = SQRT( els**2/(one + (els**2/elt**2) +(els**2/elb_mf**2)))
         el(k) = el(k)*(one-wt) + elf*wt
       END DO

    END SELECT

    
    DO k = kts+1,kte
       el_les = p25*p5*( dz(k)+dz(k-1) )
       el(k)  = el(k)*Psig_bl + (one-Psig_bl)*min(el_les,el(k))
    ENDDO
      

  END SUBROUTINE mym_length














  SUBROUTINE boulac_length0(k,kts,kte,zw,dz,qtke,theta,lb1,lb2)
















    use module_bl_mynnedmf_common, only: gtr,zero,two,p1,p5,kind_phys
    
     integer, intent(in) :: k,kts,kte
     real(kind_phys), dimension(kts:kte), intent(in) :: qtke,dz,theta
     real(kind_phys), intent(out) :: lb1,lb2
     real(kind_phys), dimension(kts:kte+1), intent(in) :: zw

     
     integer :: izz, found
     real(kind_phys):: dlu,dld
     real(kind_phys):: dzt, zup, beta, zup_inf, bbb, tl, zdo, zdo_sup, zzz


     
     
     
     zup=zero
     dlu=zw(kte+1)-zw(k)-dz(k)*p5
     zzz=zero
     zup_inf=zero
     beta=gtr           

     

     if (k .lt. kte) then      
        found = 0
        izz=k
        DO WHILE (found .EQ. 0)

           if (izz .lt. kte) then
              dzt=dz(izz)                   
              zup=zup-beta*theta(k)*dzt     
              
              zup=zup+beta*(theta(izz+1)+theta(izz))*dzt*p5 
              zzz=zzz+dzt                   
              
              if (qtke(k).lt.zup .and. qtke(k).ge.zup_inf) then
                 bbb=(theta(izz+1)-theta(izz))/dzt
                 if (bbb .ne. zero) then
                    
                    tl=(-beta*(theta(izz)-theta(k)) + &
                      & sqrt( max(zero,(beta*(theta(izz)-theta(k)))**2 + &
                      &       two*bbb*beta*(qtke(k)-zup_inf))))/bbb/beta
                 else
                    if (theta(izz) .ne. theta(k))then
                       tl=(qtke(k)-zup_inf)/(beta*(theta(izz)-theta(k)))
                    else
                       tl=zero
                    endif
                 endif
                 dlu=zzz-dzt+tl
                 
                 found =1
              endif
              zup_inf=zup
              izz=izz+1
           ELSE
              found = 1
           ENDIF

        ENDDO

     endif

     
     
     
     zdo=zero
     zdo_sup=zero
     dld=zw(k)
     zzz=zero

     
     if (k .gt. kts) then  

        found = 0
        izz=k
        DO WHILE (found .EQ. 0)

           if (izz .gt. kts) then
              dzt=dz(izz-1)
              zdo=zdo+beta*theta(k)*dzt
              
              zdo=zdo-beta*(theta(izz-1)+theta(izz))*dzt*p5
              zzz=zzz+dzt
              
              if (qtke(k).lt.zdo .and. qtke(k).ge.zdo_sup) then
                 bbb=(theta(izz)-theta(izz-1))/dzt
                 if (bbb .ne. zero) then
                    tl=(beta*(theta(izz)-theta(k))+ &
                      & sqrt( max(zero,(beta*(theta(izz)-theta(k)))**2 + &
                      &       two*bbb*beta*(qtke(k)-zdo_sup))))/bbb/beta
                 else
                    if (theta(izz) .ne. theta(k)) then
                       tl=(qtke(k)-zdo_sup)/(beta*(theta(izz)-theta(k)))
                    else
                       tl=zero
                    endif
                 endif
                 dld=zzz-dzt+tl
                 
                 found = 1
              endif
              zdo_sup=zdo
              izz=izz-1
           ELSE
              found = 1
           ENDIF
        ENDDO

     endif

     
     
     
     
     
     dld = min(dld,zw(k+1))
     lb1 = min(dlu,dld)     
     
     dlu=MAX(p1,MIN(dlu,1000._kind_phys))
     dld=MAX(p1,MIN(dld,1000._kind_phys))
     lb2 = sqrt(dlu*dld)    
     

     if (k .eq. kte) then
        lb1 = zero
        lb2 = zero
     endif
     
     

  END SUBROUTINE boulac_length0










  SUBROUTINE boulac_length(kts,kte,zw,dz,qtke,theta,lb1,lb2)







    use module_bl_mynnedmf_common, only: gtr,zero,one,two,p1,p5,kind_phys
    
     integer, intent(in) :: kts,kte
     real(kind_phys), dimension(kts:kte),   intent(in) :: qtke,dz,theta
     real(kind_phys), dimension(kts:kte),   intent(out):: lb1,lb2
     real(kind_phys), dimension(kts:kte+1), intent(in) :: zw

     
     integer :: iz, izz, found
     real(kind_phys), dimension(kts:kte) :: dlu,dld
     real(kind_phys), parameter :: Lmax=1500.  
     real(kind_phys):: dzt, zup, beta, zup_inf, bbb, tl, zdo, zdo_sup, zzz

     

     do iz=kts,kte

        
        
        
        zup    =zero
        dlu(iz)=zw(kte+1)-zw(iz)-dz(iz)*p5
        zzz    =zero
        zup_inf=zero
        beta   =gtr           

        

        if (iz .lt. kte) then      

          found = 0
          izz=iz
          DO WHILE (found .EQ. 0)

            if (izz .lt. kte) then
              dzt=dz(izz)                    
              zup=zup-beta*theta(iz)*dzt     
              
              zup=zup+beta*(theta(izz+1)+theta(izz))*dzt*p5 
              zzz=zzz+dzt                   
              
              if (qtke(iz).lt.zup .and. qtke(iz).ge.zup_inf) then
                 bbb=(theta(izz+1)-theta(izz))/dzt
                 if (bbb .ne. 0.) then
                    
                    tl=(-beta*(theta(izz)-theta(iz)) + &
                      & sqrt( max(zero,(beta*(theta(izz)-theta(iz)))**2 + &
                      &       two*bbb*beta*(qtke(iz)-zup_inf))))/bbb/beta
                 else
                    if (theta(izz) .ne. theta(iz))then
                       tl=(qtke(iz)-zup_inf)/(beta*(theta(izz)-theta(iz)))
                    else
                       tl=zero
                    endif
                 endif            
                 dlu(iz)=zzz-dzt+tl
                 
                 found =1
              endif
              zup_inf=zup
              izz=izz+1
             ELSE
              found = 1
            ENDIF

          ENDDO

        endif
                   
        
        
        
        zdo    =zero
        zdo_sup=zero
        dld(iz)=zw(iz)
        zzz    =zero

        
        if (iz .gt. kts) then  

          found = 0
          izz=iz       
          DO WHILE (found .EQ. 0) 

            if (izz .gt. kts) then
              dzt=dz(izz-1)
              zdo=zdo+beta*theta(iz)*dzt
              
              zdo=zdo-beta*(theta(izz-1)+theta(izz))*dzt*p5
              zzz=zzz+dzt
              
              if (qtke(iz).lt.zdo .and. qtke(iz).ge.zdo_sup) then
                 bbb=(theta(izz)-theta(izz-1))/dzt
                 if (bbb .ne. 0.) then
                    tl=(beta*(theta(izz)-theta(iz))+ &
                      & sqrt( max(zero,(beta*(theta(izz)-theta(iz)))**2 + &
                      &       two*bbb*beta*(qtke(iz)-zdo_sup))))/bbb/beta
                 else
                    if (theta(izz) .ne. theta(iz)) then
                       tl=(qtke(iz)-zdo_sup)/(beta*(theta(izz)-theta(iz)))
                    else
                       tl=zero
                    endif
                 endif            
                 dld(iz)=zzz-dzt+tl
                 
                 found = 1
              endif
              zdo_sup=zdo
              izz=izz-1
            ELSE
              found = 1
            ENDIF
          ENDDO

        endif

        
        
        
        
        dld(iz) = min(dld(iz),zw(iz+1))
        
        dlu(iz)=MAX(p1, dlu(iz)/(one + (dlu(iz)/Lmax)) )
        dld(iz)=MAX(p1, dld(iz)/(one + (dld(iz)/Lmax)) )

        
        lb1(iz) = min(dlu(iz),dld(iz))
        
        lb2(iz) = sqrt(dlu(iz)*dld(iz))

        
        

     ENDDO

     lb1(kte) = lb1(kte-1)
     lb2(kte) = lb2(kte-1)
        
  END SUBROUTINE boulac_length


















































  SUBROUTINE  mym_turbulence (                                &
    &            kts,kte,                                     &
    &            xland,closure,ckmod,taue,                    &
    &            dz, dx, zw, p, exner,                        &
    &            u, v, thl, thv, thlv, ql, qw,                &
    &            qke, tsq, qsq, cov,                          &
    &            vt, vq,                                      &
    &            rmol, rmolh, flt, fltv, flq,                 &
    &            pblh,theta,                                  &
    &            sh, sm,                                      &
    &            El,                                          &
    &            Dfm, Dfh, Dfq, Tcd, Qcd, Pdk, Pdt, Pdq, Pdc, &
    &		 qWT1,qSHEAR1,qBUOY1,qDISS1,                  &
    &            tke_budget,                                  &
    &            Psig_bl,Psig_shcu,cldfra_bl1,                &
    &            bl_mynn_mixlength,                           &
    &            bl_mynn_ess,                                 &
    &            edmf_w1,edmf_a1,                             &
    &            edmf_w_dd1,edmf_a_dd1,                       &
    &            TKEprod_dn,TKEprod_up,                       &
    &            spp_pbl,pattern_spp_pbl1                     )


    use module_bl_mynnedmf_common, only: gtr,tv0,             &
         a1,a2,b1,b2,c1,e1c,e2c,e3c,e4c,e5c,cc3,              &
         zero,one,two,three,four,seven,nine,p1,p2,p5,         &
         kind_phys
    
    integer, intent(in)   :: kts,kte


    integer, intent(in)               :: bl_mynn_mixlength
    integer, intent(in)               :: tke_budget
    integer, intent(in)               :: bl_mynn_ess
    real(kind_phys), intent(in)       :: closure,ckmod,taue
    real(kind_phys), dimension(kts:kte),   intent(in) :: dz
    real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
    real(kind_phys), intent(in)       :: rmol,rmolh,flt,fltv,flq,          &
         &Psig_bl,Psig_shcu,xland,dx,pblh
    real(kind_phys), dimension(kts:kte), intent(in) :: u,v,thl,thv,        &
         &thlv,qw,ql,vt,vq,qke,tsq,qsq,cov,cldfra_bl1,edmf_w1,edmf_a1,     &
         &edmf_w_dd1,edmf_a_dd1,TKEprod_dn,TKEprod_up,p,exner

    real(kind_phys), dimension(kts:kte), intent(out) :: dfm,dfh,dfq,       &
         &pdk,pdt,pdq,pdc,tcd,qcd,el

    real(kind_phys), dimension(kts:kte), intent(inout), optional ::        &
         qWT1,qSHEAR1,qBUOY1,qDISS1
    real(kind_phys):: q3sq_old,qWTP_old,qWTP_new
    real(kind_phys):: dudz,dvdz,dTdz,upwp,vpwp,Tpwp

    real(kind_phys), dimension(kts:kte) :: qkw,qtw,dtl,dqw,dtv,gm,gh,sm,sh,qpe

    integer :: k

    real(kind_phys):: e6c,dzk,afk,abk,vtt,vqq,                             &
         &cw25,clow,cupp,gamt,gamq,smd,gamv,elq,elh,elqt

    real(kind_phys):: cldavg
    real(kind_phys), dimension(kts:kte), intent(in) :: theta

    real(kind_phys)::  a2fac, duz, ri 

    real(kind_phys):: auh,aum,adh,adm,aeh,aem,Req,Rsl,Rsl2,                &
           gmelq,sm20,sh20,sm25max,sh25max,sm25min,sh25min,                &
           sm_pbl,sh_pbl,pblh2,wt,slht,mfmax,cpblh

    DOUBLE PRECISION  q2sq, t2sq, r2sq, c2sq, elsq, gmel, ghel
    DOUBLE PRECISION  q3sq, t3sq, r3sq, c3sq, dlsq, qdiv
    DOUBLE PRECISION  e1, e2, e3, e4, enum, eden, wden


    integer,         intent(in)                   :: spp_pbl
    real(kind_phys), dimension(kts:kte)           :: pattern_spp_pbl1
    real(kind_phys):: shb, Prlim
    real(kind_phys), parameter :: Prlimit_fre = 6.0 
    real(kind_phys), parameter :: Prlimit_sfc = 4.0 













    qkw  = zero
    qtw  = zero
    pblh2= max(pblh, 200._kind_phys)
    if (closure .lt. 2.7) then
       qpe = zero
    else
       do k = kts,kte
          
          
          cpblh  = min((zw(k) + p1*pblh2)/(0.8_kind_phys*pblh2), one)
          
          qpe(k) = min(three, max(zero, cpblh * two * tsq(k) * gtr**2 * taue**2))
       enddo
    endif

    CALL mym_level2 (kts,kte,                   &
    &            ckmod, bl_mynn_ess,            &
    &            zw, dz, xland, pblh,           &
    &            u, v, thl, thv, thlv,          &
    &            theta, p, exner,               &
    &            qke, cldfra_bl1,               &
    &            edmf_a1, edmf_w1,   	      	&
    &            qw, ql, vt, vq,                &
    &            dtl, dqw, dtv, gm, gh, sm, sh  )

    CALL mym_length (                           &
    &            kts,kte,xland,                 &
    &            dz, dx, zw,                    &
    &            rmol, rmolh, flt, fltv, flq,   &
    &            vt, vq,                        &
    &            u, v, qke, qpe,                &
    &            dtv,                           &
    &            el,                            &
    &            pblh, theta,                   &
    &            qkw, qtw,                      &
    &            Psig_bl, cldfra_bl1,           &
    &            bl_mynn_mixlength,             &
    &            edmf_w1,edmf_a1,               &
    &            edmf_w_dd1,edmf_a_dd1          )


    DO k = kts+1,kte
       dzk  = p5 *( dz(k)+dz(k-1) )
       afk  = dz(k)/( dz(k)+dz(k-1) )
       abk  = one -afk
       elsq = el(k)**2
       q3sq = qkw(k)**2
       q2sq = b1*elsq*( sm(k)*gm(k)+sh(k)*gh(k) )

       sh20 = MAX(sh(k), 1e-5_kind_phys)
       sm20 = MAX(sm(k), 1e-5_kind_phys)
       sh(k)= MAX(sh(k), 1e-5_kind_phys)

       
       duz = ( u(k)-u(k-1) )**2 +( v(k)-v(k-1) )**2
       duz =   duz                    /dzk**2
       
       ri = -gh(k)/MAX( duz, 1.0e-10_kind_phys )
       
       a2fac = one/(one + ckmod*max(ri,zero))

       
       
       
       

       
       if (ri >= one) then
          
          Prlim = min(seven*ri, Prlimit_fre)
       elseif (ri >= 0.01 .and. ri <= one) then
          
          Prlim = min(6.873_kind_phys*ri + one/(6.873_kind_phys*ri), Prlimit_fre)
       else
          
          Prlim = Prlimit_fre
       end if
       wt    = min(one, zw(k)/max(pblh, 200._kind_phys))  
       Prlim = wt*Prlim + (one-wt)*min(Prlimit_sfc, Prlim)


       gmel = gm(k)*elsq
       ghel = gh(k)*elsq


       
       IF ( debug_code ) THEN
         IF (sh(k)<0.0 .OR. sm(k)<0.0) THEN
           print*,"MYNN; mym_turbulence 2.0; sh=",sh(k)," k=",k
           print*," gm=",gm(k)," gh=",gh(k)," sm=",sm(k)
           print*," q2sq=",q2sq," q3sq=",q3sq," q3/q2=",q3sq/q2sq
           print*," qke=",qke(k)," el=",el(k)," ri=",ri
           print*," PBLH=",pblh," u=",u(k)," v=",v(k)
         ENDIF
       ENDIF





       dlsq =  elsq
       IF ( q3sq/dlsq .LT. -gh(k) ) q3sq = -dlsq*gh(k)

       IF ( q3sq .LT. q2sq ) THEN
          
          qdiv = SQRT( q3sq/q2sq )   

          
          
          
          
          
          
          
          
          
          
          
          

          
          sh(k) = sh(k) * qdiv
          sm(k) = sm(k) * qdiv

          
          
          
          
          
          
          e1   = q3sq - e1c*ghel*a2fac * qdiv**2
          e2   = q3sq - e2c*ghel*a2fac * qdiv**2
          e3   = e1   + e3c*ghel*a2fac**2 * qdiv**2
          e4   = e1   - e4c*ghel*a2fac * qdiv**2
          eden = e2*e4 + e3*e5c*gmel * qdiv**2
          eden = MAX( eden, 1.0d-20 )
          
          
          
       ELSE
          
          
          
          
          
          e1   = q3sq - e1c*ghel*a2fac
          e2   = q3sq - e2c*ghel*a2fac
          e3   = e1   + e3c*ghel*a2fac**2
          e4   = e1   - e4c*ghel*a2fac
          eden = e2*e4 + e3*e5c*gmel
          eden = MAX( eden, 1.0d-20 )

          qdiv = one
          
          sm(k) = q3sq*a1*( e3-three*c1*e4       )/eden
          
          
          
          sh(k) = q3sq*(a2*a2fac)*( e2+three*c1*e5c*gmel )/eden
       END IF 

       
       gmelq    = max(real(gmel/q3sq, kind_phys), 1e-8_kind_phys)
       sm25max  = four  
       sh25max  = four  
       sm25min  = zero 
       sh25min  = zero 

       
       
       IF ( debug_code ) THEN
         IF ((sh(k)<sh25min .OR. sm(k)<sm25min .OR. &
              sh(k)>sh25max .OR. sm(k)>sm25max) ) THEN
           print*,"In mym_turbulence 2.5: k=",k
           print*," sm=",sm(k)," sh=",sh(k)
           print*," ri=",ri," Pr=",sm(k)/max(sh(k),1e-8_kind_phys)
           print*," gm=",gm(k)," gh=",gh(k)
           print*," q2sq=",q2sq," q3sq=",q3sq, q3sq/q2sq
           print*," qke=",qke(k)," el=",el(k)
           print*," PBLH=",pblh," u=",u(k)," v=",v(k)
           print*," SMnum=",q3sq*a1*( e3-three*c1*e4)," SMdenom=",eden
           print*," SHnum=",q3sq*(a2*a2fac)*( e2+three*c1*e5c*gmel ),&
                  " SHdenom=",eden
         ENDIF
       ENDIF

       
       IF ( sh(k) > sh25max ) sh(k) = sh25max
       IF ( sh(k) < sh25min ) sh(k) = sh25min
       
       

       shb   = max(sh(k), 0.004_kind_phys)
       sm(k) = min(sm(k), Prlim*shb)


       IF ( closure .GE. 3.0 ) THEN
          t2sq = qdiv*b2*elsq*sh(k)*dtl(k)**2
          r2sq = qdiv*b2*elsq*sh(k)*dqw(k)**2
          c2sq = qdiv*b2*elsq*sh(k)*dtl(k)*dqw(k)
          t3sq = MAX( tsq(k)*abk+tsq(k-1)*afk, zero )
          r3sq = MAX( qsq(k)*abk+qsq(k-1)*afk, zero )
          c3sq =      cov(k)*abk+cov(k-1)*afk


          c3sq = SIGN( MIN( ABS(c3sq), SQRT(t3sq*r3sq) ), c3sq )

          vtt  = one +vt(k)*abk +vt(k-1)*afk
          vqq  = tv0 +vq(k)*abk +vq(k-1)*afk

          t2sq = vtt*t2sq +vqq*c2sq
          r2sq = vtt*c2sq +vqq*r2sq
          c2sq = MAX( vtt*t2sq+vqq*r2sq, 0.0d0 )
          t3sq = vtt*t3sq +vqq*c3sq
          r3sq = vtt*c3sq +vqq*r3sq
          c3sq = MAX( vtt*t3sq+vqq*r3sq, 0.0d0 )

          cw25 = e1*( e2 + three*c1*e5c*gmel*qdiv**2 )/( three*eden )


          dlsq =  elsq
          IF ( q3sq/dlsq .LT. -gh(k) ) q3sq = -dlsq*gh(k)



          
          auh = 27._kind_phys*a1*((a2*a2fac)**2)*b2*(gtr)**2
          aum = 54._kind_phys*(a1**2)*(a2*a2fac)*b2*c1*(gtr)
          adh = nine*a1*((a2*a2fac)**2)*(12._kind_phys*a1 + three*b2)*(gtr)**2
          adm = 18._kind_phys*(a1**2)*(a2*a2fac)*(b2 - three*(a2*a2fac))*(gtr)

          aeh = (nine*a1*((a2*a2fac)**2)*b1 +nine*a1*((a2*a2fac)**2)*            &
                (12._kind_phys*a1 + three*b2))*(gtr)
          aem = three*a1*(a2*a2fac)*b1*(three*(a2*a2fac) + three*b2*c1 +         &
                (18._kind_phys*a1*c1 - b2)) +                                    &
                (18._kind_phys)*(a1**2)*(a2*a2fac)*(b2 - three*(a2*a2fac))

          Req = -aeh/aem
          Rsl = (auh + aum*Req)/(three*adh + three*adm*Req)
          
          Rsl = 0.12_kind_phys   
          Rsl2= one - two*Rsl    
          
          


          
          

          
          
          
          
          e2   = q3sq - e2c*ghel*a2fac * qdiv**2
          e3   = q3sq + e3c*ghel*a2fac**2 * qdiv**2
          e4   = q3sq - e4c*ghel*a2fac * qdiv**2
          eden = e2*e4  + e3 *e5c*gmel * qdiv**2

          
          
          
          wden = cc3*gtr**2 * dlsq**2/elsq * qdiv**2 &
               &        *( e2*e4c*a2fac - e3c*e5c*gmel*a2fac**2 * qdiv**2 )

          IF ( wden .NE. zero ) THEN
             
             clow = q3sq*( 0.12-cw25 )*eden/wden
             cupp = q3sq*( 0.76-cw25 )*eden/wden
             
             

             IF ( wden .GT. zero ) THEN
                c3sq  = MIN( MAX( c3sq, c2sq+clow ), c2sq+cupp )
             ELSE
                c3sq  = MAX( MIN( c3sq, c2sq+clow ), c2sq+cupp )
             END IF
          END IF

          e1   = e2 + e5c*gmel * qdiv**2
          eden = MAX( eden, 1.0d-20 )


          
          
          e6c  = 3.0*(a2*a2fac)*cc3*gtr * dlsq/elsq

          
          
          
          IF ( t2sq .GE. zero ) THEN
             enum = MAX( qdiv*e6c*( t3sq-t2sq ), 0.0d0 )
          ELSE
             enum = MIN( qdiv*e6c*( t3sq-t2sq ), 0.0d0 )
          ENDIF
          gamt =-e1  *enum    /eden

          
          
          
          IF ( r2sq .GE. zero ) THEN
             enum = MAX( qdiv*e6c*( r3sq-r2sq ), 0.0d0 )
          ELSE
             enum = MIN( qdiv*e6c*( r3sq-r2sq ), 0.0d0 )
          ENDIF
          gamq =-e1  *enum    /eden

          

          
          enum = MAX( qdiv*e6c*( c3sq-c2sq ), 0.0d0)

          
          
          smd  = dlsq*enum*gtr/eden * qdiv**2 * (e3c*a2fac**2 + &
               & e4c*a2fac)*a1/(a2*a2fac)

          gamv = e1  *enum*gtr/eden
          sm(k) = sm(k) +smd

          
          
          qdiv = one

          
          IF ( debug_code ) THEN
            IF (sh(k)<-0.3 .OR. sm(k)<-0.3 .OR. &
              qke(k) < -0.1 .or. ABS(smd) .gt. 2.0) THEN
              print*," MYNN; mym_turbulence3.0; sh=",sh(k)," k=",k
              print*," gm=",gm(k)," gh=",gh(k)," sm=",sm(k)
              print*," q2sq=",q2sq," q3sq=",q3sq," q3/q2=",q3sq/q2sq
              print*," qke=",qke(k)," el=",el(k)," ri=",ri
              print*," PBLH=",pblh," u=",u(k)," v=",v(k)
            ENDIF
          ENDIF



       ELSE

          gamt = zero
          gamq = zero
          gamv = zero
       END IF



       cldavg = p5*(cldfra_bl1(k-1) + cldfra_bl1(k))
       mfmax  = max(p5*(edmf_a1(k-1)+edmf_a1(k))*p5*(edmf_w1(k-1)+edmf_w1(k)), abs(edmf_a_dd1(k)*edmf_w_dd1(k)))
       
       sm(k) = max(sm(k), 0.04_kind_phys*min(10._kind_phys*mfmax,one) )
       sh(k) = max(sh(k), 0.04_kind_phys*min(10._kind_phys*mfmax,one) )
       
       sm(k) = max(sm(k), 0.04_kind_phys*min(cldavg, p5) )
       sh(k) = max(sh(k), 0.04_kind_phys*min(cldavg, p5) )
       
       if (tkeprod_dn(k)>1e-6) sm(k) = max(sm(k), 0.01_kind_phys)
       if (tkeprod_dn(k)>1e-6) sh(k) = max(sh(k), 0.01_kind_phys)
       
       if (closure .ge. 2.7 .and. closure .lt. 3.0) then
          sm(k) = max(sm(k),min(p2, max(zero,three*qpe(k))))
       endif
       
       elq   = el(k)*qkw(k)
       elqt  = el(k)*qtw(k)  
       elh   = elq*qdiv

       
       
       pdk(k) = elq *(sm(k)*gm(k)                    &
            & +       sh(k)*gh(k)+gamv ) +           &
            &   p5*TKEprod_dn(k)         +           & 
            &   p5*TKEprod_up(k)
       pdt(k) = elh *( sh(k)*dtl(k)+gamt )*dtl(k)
       pdq(k) = elh *( sh(k)*dqw(k)+gamq )*dqw(k)
       pdc(k) = elh *( sh(k)*dtl(k)+gamt )*dqw(k)*p5 &
            & + elh *( sh(k)*dqw(k)+gamq )*dtl(k)*p5

       
       tcd(k) = elq*gamt
       qcd(k) = elq*gamq

       
       dfm(k) = elqt*sm(k) / dzk
       dfh(k) = elq *sh(k) / dzk



       dfq(k) =     dfm(k)


   IF (tke_budget .eq. 1) THEN
       









       


       
       
       qSHEAR1(k) = elq*sm(k)*gm(k) 

       
       
       
       
       
       
       qBUOY1(k) = elq*(sh(k)*gh(k)+gamv)  +         &
       &           p5*TKEprod_dn(k)       +          & 
       &           p5*TKEprod_up(k) 

       
       
       
       
    ENDIF

    END DO

    
    dfm(kts) = zero
    dfh(kts) = zero
    dfq(kts) = zero
    pdk(kts) = zero
    pdt(kts) = zero
    pdq(kts) = zero
    pdc(kts) = zero
    tcd(kts) = zero
    qcd(kts) = zero

    tcd(kte) = zero
    qcd(kte) = zero


    DO k = kts,kte-1
       dzk = dz(k)
       tcd(k) = ( tcd(k+1)-tcd(k) )/( dzk )
       qcd(k) = ( qcd(k+1)-qcd(k) )/( dzk )
    END DO

    if (spp_pbl==1) then
       DO k = kts,kte
          dfm(k)= dfm(k) + dfm(k)* pattern_spp_pbl1(k) * 1.5 * MAX(exp(-MAX(zw(k)-8000.,zero)/2000.),0.001)
          dfh(k)= dfh(k) + dfh(k)* pattern_spp_pbl1(k) * 1.5 * MAX(exp(-MAX(zw(k)-8000.,zero)/2000.),0.001)
       END DO
    endif



  END SUBROUTINE mym_turbulence















































SUBROUTINE  mym_predict (kts,kte,                                     &
     &            closure,                                            &
     &            delt, dz,                                           &
     &            ust, flt, flq, pmz, phh,                            &
     &            el,  dfq, rho,                                      &
     &            pdk, pdt, pdq, pdc,                                 &
     &            qke, tsq, qsq, cov,                                 &
     &            s_aw1, s_awqke1, s_awqsq1,                          &
     &            bl_mynn_edmf, bl_mynn_edmf_tke,                     &
     &            qWT1, qDISS1,tke_budget, upwind                     )


  use module_bl_mynnedmf_common, only: qkemin,karman,sqfac,b1,b2,     &
       zero,one,two,three,p5,kind_phys
  
  integer, intent(in) :: kts,kte    


real(kind_phys), intent(in)    :: closure,upwind
integer, intent(in) :: bl_mynn_edmf,bl_mynn_edmf_tke,tke_budget
real(kind_phys), dimension(kts:kte), intent(in) :: dz, dfq, el, rho
real(kind_phys), dimension(kts:kte), intent(inout) :: pdk, pdt, pdq, pdc
real(kind_phys), intent(in)    :: flt, flq, pmz, phh
real(kind_phys), intent(in)    :: ust, delt
real(kind_phys), dimension(kts:kte), intent(inout) :: qke,tsq, qsq, cov
real(kind_phys), dimension(kts:kte+1), intent(inout) :: s_awqke1,s_awqsq1,s_aw1
    

real(kind_phys), dimension(kts:kte), intent(out) :: qWT1, qDISS1
real(kind_phys), dimension(kts:kte) :: tke_up,dzinv  

    
integer :: k
real(kind_phys), dimension(kts:kte) :: qkw, bp, rp, df3q
real(kind_phys):: vkz,pdk1,phm,pdt1,pdq1,pdc1,b1l,b2l,onoff
real(kind_phys), dimension(kts:kte) :: dtz,upcont,dncont
real(kind_phys), dimension(kts:kte) :: a,b,c,d,x

real(kind_phys), dimension(kts:kte) :: rhoinv
real(kind_phys), dimension(kts:kte+1) :: rhoz,kqdz,kmdz


IF (bl_mynn_edmf_tke == 0) THEN
   onoff=zero
ELSE
   onoff=one
ENDIF


vkz = karman*p5*dz(kts)



DO k = kts,kte

   qkw(k) = SQRT( MAX( qke(k), qkemin ) )
   df3q(k)=Sqfac*dfq(k)
   dtz(k)=delt/dz(k)
END DO




rhoz(kts)  =rho(kts)
rhoinv(kts)=one/rho(kts)
kqdz(kts)  =rhoz(kts)*df3q(kts)
kmdz(kts)  =rhoz(kts)*dfq(kts)
DO k=kts+1,kte
   rhoz(k)  =(rho(k)*dz(k-1) + rho(k-1)*dz(k))/(dz(k-1)+dz(k))
   rhoz(k)  =    MAX(rhoz(k),1E-4_kind_phys)
   rhoinv(k)=one/MAX(rho(k) ,1E-4_kind_phys)
   kqdz(k)  = rhoz(k)*df3q(k) 
   kmdz(k)  = rhoz(k)*dfq(k)  
ENDDO
rhoz(kte+1)=rhoz(kte)
kqdz(kte+1)=rhoz(kte+1)*df3q(kte)
kmdz(kte+1)=rhoz(kte+1)*dfq(kte)

if (bl_mynn_edmf == 1) then
   
   DO k=kts+1,kte-1
      kqdz(k) = MAX(kqdz(k),  p5* s_aw1(k))
      kqdz(k) = MAX(kqdz(k), -p5*(s_aw1(k)-s_aw1(k+1)))
      kmdz(k) = MAX(kmdz(k),  p5* s_aw1(k))
      kmdz(k) = MAX(kmdz(k), -p5*(s_aw1(k)-s_aw1(k+1)))




   ENDDO
endif




pdk1 = two*ust**3*pmz/( vkz )
phm  = two/ust   *phh/( vkz )
pdt1 = phm*flt**2
pdq1 = phm*flq**2
pdc1 = phm*flt*flq


pdk(kts) = pdk1 - pdk(kts+1)




pdt(kts) = pdt(kts+1)
pdq(kts) = pdq(kts+1)
pdc(kts) = pdc(kts+1)





DO k = kts,kte-1
   b1l = b1*p5*( el(k+1)+el(k) )
   bp(k) = two*qkw(k) / b1l
   rp(k) = pdk(k+1) + pdk(k)
ENDDO
bp(kte) = zero


if (bl_mynn_edmf > 1) then

    DO k=kts+1,kte-1
       upcont(k)= s_awqke1(k)- s_aw1(k)*(qke(k)*upwind+qke(k-1)*(one-upwind))
       dncont(k)=zero 
    ENDDO
    
    upcont(kte)=zero
    dncont(kte)=zero

    k=kts
    a(1)=zero
    b(1)=one + dtz(k)*kqdz(k+1)*rhoinv(k) + bp(k)*delt
    c(1)=    - dtz(k)*kqdz(k+1)*rhoinv(k)
    d(1)=max(qkemin, qke(k)) + rp(k)*delt                        &
        &    - dtz(k)*(upcont(k+1)+dncont(k+1))

    DO k=kts+1,kte-1
       a(k)=   - dtz(k)*kqdz(k)*rhoinv(k)
       b(k)=one+ dtz(k)*(kqdz(k)+kqdz(k+1))*rhoinv(k) + bp(k)*delt
       c(k)=   - dtz(k)*kqdz(k+1)*rhoinv(k)
       d(k)=max(qkemin, qke(k)) + rp(k)*delt                     &
          &    - dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
    ENDDO

else 
   DO k=kts,kte-1





      a(k)=   - dtz(k)*kqdz(k)*rhoinv(k)                         &
          &   + 0.5*dtz(k)*rhoinv(k)*s_aw1(k)*onoff
      b(k)=1. + dtz(k)*(kqdz(k)+kqdz(k+1))*rhoinv(k)             &
          &   + 0.5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))*onoff &
          &   + bp(k)*delt
      c(k)=   - dtz(k)*kqdz(k+1)*rhoinv(k)                       &
          &   - 0.5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff
      d(k)=rp(k)*delt + qke(k)                                   &
          &   + dtz(k)*rhoinv(k)*(s_awqke1(k)-s_awqke1(k+1))*onoff
   ENDDO
endif







a(kte)=0.
b(kte)=1.
c(kte)=0.
d(kte)=qke(kte)


CALL tridiag2(kte,a,b,c,d,x)

DO k=kts,kte

   qke(k)=max(x(k), qkemin)
   qke(k)=min(qke(k), 150.)
ENDDO
      
   

IF (tke_budget .eq. 1) THEN
   
   tke_up=p5*qke
   dzinv=one/dz
   k=kts
   qWT1(k)=dzinv(k)*(                                                &
        &  (kqdz(k+1)*(tke_up(k+1)-tke_up(k))-kqdz(k)*tke_up(k))     &
        &  + p5*rhoinv(k)*(s_aw1(k+1)*tke_up(k+1)                    &
        &  +      (s_aw1(k+1)-s_aw1(k))*tke_up(k)                    &
        &  +      (s_awqke1(k)-s_awqke1(k+1)))*onoff) 
   DO k=kts+1,kte-1
      qWT1(k)=dzinv(k)*(                                             &
            & (kqdz(k+1)*(tke_up(k+1)-tke_up(k))-kqdz(k)*(tke_up(k)-tke_up(k-1))) &
            &  + p5*rhoinv(k)*(s_aw1(k+1)*tke_up(k+1)                &
            &  +      (s_aw1(k+1)-s_aw1(k))*tke_up(k)                &
            &  -                  s_aw1(k)*tke_up(k-1)               &
            &  +      (s_awqke1(k)-s_awqke1(k+1)))*onoff) 
   ENDDO
   k=kte
   qWT1(k)=dzinv(k)*(-kqdz(k)*(tke_up(k)-tke_up(k-1))                &
       &  + p5*rhoinv(k)*(-s_aw1(k)*tke_up(k)-s_aw1(k)*tke_up(k-1)+s_awqke1(k))*onoff) 
   
   qDISS1=bp*tke_up 
END IF


if ( closure > 2.5 ) then
   
   
   
   do k = kts,kte-1
      b2l   = b2*p5*( el(k+1)+el(k) )
      bp(k) = two*qkw(k) / b2l
      rp(k) = pdq(k+1) + pdq(k)
   enddo
   bp(kte) = zero


   if (bl_mynn_edmf > 1) then
      do k=kts+1,kte-1
         upcont(k)= s_awqsq1(k)- s_aw1(k)*(qsq(k)*upwind+qsq(k-1)*(one-upwind))
         dncont(k)=zero 
      enddo
      
      upcont(kte)=zero
      dncont(kte)=zero

      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*kmdz(k+1)*rhoinv(k) + bp(k)*delt
      c(1)=    - dtz(k)*kmdz(k+1)*rhoinv(k)
      d(1)=max(zero, qsq(k)) + rp(k)*delt                        &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      do k=kts+1,kte-1
         a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k) + bp(k)*delt
         c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)
         d(k)=max(zero, qsq(k)) + rp(k)*delt                     &
            &    - dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      enddo
       
   else 

      do k=kts,kte-1
         a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)                         &
             &   + 0.5*dtz(k)*rhoinv(k)*s_aw1(k)*onoff
         b(k)=1. + dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k)             &
             &   + 0.5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))*onoff &
             &   + bp(k)*delt
         c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)                       &
             &   - 0.5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff
         d(k)=rp(k)*delt + qsq(k)                                   &
             &   + dtz(k)*rhoinv(k)*(s_awqsq1(k)-s_awqsq1(k+1))*onoff
      enddo
   endif

   a(kte)=-one
   b(kte)=one
   c(kte)=zero
   d(kte)=zero 

   
   call tridiag2(kte,a,b,c,d,x)

   do k=kts,kte
      
      qsq(k)=max(x(k), 1e-17_kind_phys)
      qsq(k)=min(qsq(k),6e-6_kind_phys)
   enddo

else
   
   do k = kts,kte-1
      if ( qkw(k) .le. zero ) then
         b2l = zero
      else
         b2l = b2*0.25_kind_phys*( el(k+1)+el(k) )/qkw(k)
      end if
      qsq(k) = min(6e-6_kind_phys, max(1e-17_kind_phys, b2l*( pdq(k+1)+pdq(k) )))
   enddo
   qsq(kte)=qsq(kte-1)
endif
    


    IF ( closure .GE. 2.7 ) THEN





       DO k = kts,kte-1
          b2l   = b2*p5*( el(k+1)+el(k) )
          bp(k) = two*qkw(k) / b2l
          rp(k) = pdt(k+1) + pdt(k) 
       END DO
       

       






       DO k=kts,kte-1
          
          
          
          

          a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)
          b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k) + bp(k)*delt
          c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)
          d(k)=rp(k)*delt + max(zero, tsq(k))
       ENDDO








       a(kte)=-1. 
       b(kte)=one
       c(kte)=zero
       d(kte)=zero
       

       CALL tridiag2(kte,a,b,c,d,x)

       DO k=kts,kte
          
          tsq(k)=min(three, max(zero,x(k)))
       ENDDO
    ELSE
       
       DO k = kts,kte-1
          IF ( qkw(k) .LE. zero ) THEN
             b2l = zero
          ELSE
             b2l = b2*0.25*( el(k+1)+el(k) )/qkw(k)
          END IF
          tsq(k) = min(three, max(zero, b2l*( pdt(k+1)+pdt(k) )))
       END DO
       tsq(kte)=tsq(kte-1)
    ENDIF

    
    IF ( closure .GE. 3.0 ) THEN

       DO k = kts,kte-1
          b2l = b2*p5*( el(k+1)+el(k) )
          bp(k) = two*qkw(k) / b2l
          rp(k) = pdc(k+1) + pdc(k) 
       END DO
       







       DO k=kts,kte-1
          
          
          
          

          a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)
          b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k) + bp(k)*delt
          c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)
          d(k)=rp(k)*delt + cov(k)
       ENDDO








       a(kte)=-1. 
       b(kte)=one
       c(kte)=zero
       d(kte)=zero


       CALL tridiag2(kte,a,b,c,d,x)
       
       DO k=kts,kte

          cov(k)=x(k)
       ENDDO
       
    ELSE

       
       DO k = kts,kte-1
          IF ( qkw(k) .LE. zero ) THEN
             b2l = zero
          ELSE
             b2l = b2*0.25*( el(k+1)+el(k) )/qkw(k)
          END IF
          cov(k) = b2l*( pdc(k+1)+pdc(k) )
       END DO
       cov(kte)=cov(kte-1)
      
    END IF


  END SUBROUTINE mym_predict
  


































  SUBROUTINE  mym_condensation (kts,kte,   &
    &            dx, dz, zw, xland,        &
    &            thl, qw, qv, qc, qi, qs,  &
    &            p,exner,                  &
    &            tsq, qsq, cov,            &
    &            Sh, el, bl_mynn_cloudpdf, &
    &            qc_bl1, qi_bl1,           &
    &            cldfra_bl1,               &
    &            PBLH1,HFX1,               &
    &            Vt, Vq, th, sgm,          &
    &            closure,                  &
    &            spp_pbl, pattern_spp_pbl1 )


    use module_bl_mynnedmf_common, only: cp,xlv,r_d,r_v,xlvcp,ep_2,   &
         ep_3,p608,cpv,rr2,rrp,tv0,tice,tliq,b2,                      &
         zero,one,two,three,four,five,ten,hundred,p1,p2,p5,           &
         kind_phys
    
    integer, intent(in)   :: kts,kte, bl_mynn_cloudpdf


    real(kind_phys), intent(in)      :: HFX1,xland
    real(kind_phys), intent(in)      :: dx,pblh1,closure
    real(kind_phys), dimension(kts:kte), intent(in) :: dz
    real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
    real(kind_phys), dimension(kts:kte), intent(in) :: p,exner,thl,qw,   &
         &qv,qc,qi,qs,tsq,qsq,cov,th

    real(kind_phys), dimension(kts:kte), intent(inout) :: vt,vq,sgm

    real(kind_phys), dimension(kts:kte) :: alp,a,bet,b,ql,q1,RH
    real(kind_phys), dimension(kts:kte), intent(out) :: qc_bl1,qi_bl1,   &
         &cldfra_bl1
    DOUBLE PRECISION :: t3sq, r3sq, c3sq

    real(kind_phys):: qsl,esat,qsat,dqsl,cld0,q1k,qlk,eq1,qll,           &
         &q2p,pt,rac,qt,t,xl,rsl,cpm,Fng,qww,alpha,beta,bb,sgmq,sgmc,    &
         &ls,wt,wt2,cld_factor,fac_damp,liq_frac,ql_ice,ql_water,        &
         &qmq,qsat_tk,q1_rh,rh_adj,zsl,maxqc,cldfra_rh,cldfra_qsq,       &
         &cldfra_rh0,cldfra_rh1,cldfra_qsq0,cldfra_qsq1,clim,qlim
    
    real(kind_phys), parameter :: qlim_sfc =0.008
    real(kind_phys), parameter :: qlim_pbl =0.020
    real(kind_phys), parameter :: qlim_trp =0.025
    
    real(kind_phys), parameter :: clim_sfc =0.010
    real(kind_phys), parameter :: clim_pbl =0.020
    real(kind_phys), parameter :: clim_trp =0.030
    real(kind_phys), parameter :: rhcrit   =0.83 
    real(kind_phys), parameter :: rhmax    =1.10 
    integer :: i,j,k

    real(kind_phys):: erf

    
    real:: dth,dtl,dqw,dzk,els
    real(kind_phys), dimension(kts:kte), intent(in) :: Sh,el

    
    real(kind_phys)           :: zagl,damp,PBLH2
    real(kind_phys)           :: cfmax

    
    real(kind_phys)           :: theta1, theta2, ht1, ht2
    integer                   :: k_tropo


    integer,  intent(in)      :: spp_pbl
    real(kind_phys), dimension(kts:kte) :: pattern_spp_pbl1
    real(kind_phys)           :: qw_pert






    DO k = kte-3, kts, -1
       theta1 = th(k)
       theta2 = th(k+2)
       ht1 = 44307.692 * (one - (p(k)/101325.)**0.190)
       ht2 = 44307.692 * (one - (p(k+2)/101325.)**0.190)
       if ( (((theta2-theta1)/(ht2-ht1)) .lt. 10./1500. ) .AND.       &
     &                       (ht1.lt.19000.) .and. (ht1.gt.4000.) ) then 
          goto 86
       endif
    ENDDO
 86   continue
    k_tropo = MAX(kts+2, k+2)

    zagl = 0.

    SELECT CASE(bl_mynn_cloudpdf)

      CASE (0) 

        DO k = kts,kte-1
           t  = th(k)*exner(k)











           
           esat = esat_blend(t)
           
           
           qsl=ep_2*esat/max(1.e-4,(p(k)-ep_3*esat))
           
           dqsl = qsl*ep_2*xlv/( r_d*t**2 )

           alp(k) = one/( one+dqsl*xlvcp )
           bet(k) = dqsl*exner(k)

           
           
           t3sq = MAX( tsq(k), zero )
           r3sq = MAX( qsq(k), zero )
           c3sq =      cov(k)
           c3sq = SIGN( MIN( ABS(c3sq), SQRT(t3sq*r3sq) ), c3sq )
           r3sq = r3sq +bet(k)**2*t3sq -2.0*bet(k)*c3sq
           
           qmq  = qw(k) -qsl
           
           sgm(k) = SQRT( MAX( r3sq, 1.0d-10 ))
           
           q1(k)   = qmq / sgm(k)
           
           cldfra_bl1(k) = 0.5*( one+erf( q1(k)*rr2 ) )

           q1k  = q1(k)
           eq1  = rrp*EXP( -0.5*q1k*q1k )
           qll  = MAX( cldfra_bl1(k)*q1k + eq1, zero )
           
           ql(k) = alp(k)*sgm(k)*qll
           
           liq_frac = min(one, max(zero,(t-240.0)/29.0))
           qc_bl1(k) = liq_frac*ql(k)
           qi_bl1(k) = (one - liq_frac)*ql(k)

           
           q2p = xlvcp/exner(k)
           pt = thl(k) +q2p*ql(k) 

           
           qt   = one +p608*qw(k) -(1.+p608)*(qc_bl1(k)+qi_bl1(k))*cldfra_bl1(k)
           rac  = alp(k)*( cldfra_bl1(K)-qll*eq1 )*( q2p*qt-(1.+p608)*pt )

           
           
           
           vt(k) =      qt-one -rac*bet(k)
           vq(k) = p608*pt-tv0 +rac

        END DO

      CASE (1, -1) 
                       
        DO k = kts,kte-1
           t  = th(k)*exner(k)
           
           esat = esat_blend(t)
           
           
           qsl=ep_2*esat/max(1.e-4,(p(k)-ep_3*esat))
           
           dqsl = qsl*ep_2*xlv/( r_d*t**2 )

           alp(k) = one/( one+dqsl*xlvcp )
           bet(k) = dqsl*exner(k)

           if (k .eq. kts) then 
             dzk = 0.5*dz(k)
           else
             dzk = dz(k)
           end if
           dth = 0.5*(thl(k+1)+thl(k)) - 0.5*(thl(k)+thl(MAX(k-1,kts)))
           dqw = 0.5*(qw(k+1) + qw(k)) - 0.5*(qw(k) + qw(MAX(k-1,kts)))
           sgm(k) = SQRT( MAX( (alp(k)**2 * MAX(el(k)**2, p1) * &
                             b2 * MAX(Sh(k),0.03))/four * &
                      (dqw/dzk - bet(k)*(dth/dzk ))**2 , 1.0e-10) )
           qmq   = qw(k) -qsl
           q1(k) = qmq / sgm(k)
           cldfra_bl1(K) = p5*( one+erf( q1(k)*rr2 ) )


           
           
           q1k  = q1(k)
           eq1  = rrp*EXP( -0.5*q1k*q1k )
           qll  = MAX( cldfra_bl1(K)*q1k + eq1, zero )
           
           ql (k) = alp(k)*sgm(k)*qll
           liq_frac = min(one, max(zero,(t-240.0)/29.0))
           qc_bl1(k) = liq_frac*ql(k)
           qi_bl1(k) = (one - liq_frac)*ql(k)

           
           q2p = xlvcp/exner(k)
           pt = thl(k) +q2p*ql(k) 

           
           qt   = one +p608*qw(k) -(1.+p608)*(qc_bl1(k)+qi_bl1(k))*cldfra_bl1(k)
           rac  = alp(k)*( cldfra_bl1(K)-qll*eq1 )*( q2p*qt-(1.+p608)*pt )

           
           
           
           vt(k) =      qt-one -rac*bet(k)
           vq(k) = p608*pt-tv0 +rac

        END DO

      CASE (2, -2)

        
        
        pblh2=MAX(ten,pblh1)
        DO k = kts,kte-1
           zagl   = zw(k) + p5*dz(k)

           t      = th(k)*exner(k)
           xl     = xl_blend(t)              
           qsat_tk= qsat_blend(t,  p(k))     
           rh(k)  = MAX(MIN(rhmax, qw(k)/MAX(1E-10_kind_phys,qsat_tk)),0.001_kind_phys)

           
           dqsl   = qsat_tk*ep_2*xlv/( r_d*t**2 )
           alp(k) = one/(one + dqsl*xlvcp )
           bet(k) = dqsl*exner(k)
 
           rsl    = xl*qsat_tk / (r_v*t**2)  
                                             
           cpm    = cp + qw(k)*cpv           
           a(k)   = one/(one + xl*rsl/cpm)   
           b(k)   = a(k)*rsl                 

           
           qw_pert= qw(k) + qw(k)*p5*pattern_spp_pbl1(k)*real(spp_pbl,kind=kind_phys)

           
           qmq    = qw_pert - qsat_tk          

           
           
           r3sq   = max( qsq(k), zero )
           t3sq   = max( tsq(k), zero )
           c3sq   =      cov(k)
           c3sq   = SIGN( MIN( ABS(c3sq), SQRT(t3sq*r3sq) ), c3sq )

           if (closure .lt. 2.7) then
              r3sq = r3sq
           elseif ((closure .ge. 2.7) .and. (closure .lt. 3.0)) then
              r3sq = r3sq +bet(k)**2*t3sq
           else
              r3sq = r3sq +bet(k)**2*t3sq -two*bet(k)*c3sq
           endif
           
           
           sgm(k) = max(1e-13, sqrt( real(r3sq,kind_phys) ))
           
           sgm(k) = min( sgm(k), qw(k)*p5 )
           
           
           wt     = min(one, max(zero, dz(k)-100.)/500.) 
           sgm(k) = sgm(k) + sgm(k)*p2*wt 
           
           sgmq   = sgm(k)
           sgmc   = sgm(k)
           
           
           wt     = min(one, max(zero, (zagl - (pblh2+10.)))/250.) 
           clim   = clim_pbl*(one-wt) + clim_trp*wt
           zsl    = min(150., max(50., p1*pblh2))        
           wt     = min(one, max(zero, zagl - zsl)/150.)  
           clim   = clim_sfc*(one-wt) + clim*wt
           sgmc   = max( sgmc, qw(k)*clim )
           
           sgmc   = max(1e-13, sgmc)
           
           if (qmq .ge. zero) sgmc = max(0.02*qw(k), sgmc)
           
           q1(k)  = qmq  / sgmc  

           
           
           wt2    = min(one, max(zero, zagl - pblh2)/300.) 
           
           if ((qi(k)+qs(k))>1.e-10 .and. (zagl .gt. pblh2)) then
              rh_adj  =min(rhmax, rhcrit + wt2*0.037_kind_phys*(max(zero, ten + log10(qi(k)+qs(k))) ))
              rh_adj  =max(rh(k), rh_adj)
              
              q1_rh   =-3. + three*(rh_adj-rhcrit)/(one-rhcrit)
              q1(k)   =max(q1_rh, q1(k) )
           endif
           
           if (qc(k)>1.e-5 .and. (zagl .gt. pblh2)) then
              rh_adj  =min(rhmax, rhcrit + wt2*0.07_kind_phys*(max(zero, five + log10(qc(k))) ))
              rh_adj  =max(rh(k), rh_adj)
              
              q1_rh   =-3. + three*(rh_adj-rhcrit)/(one-rhcrit)
              q1(k)   =max(q1_rh, q1(k) )
           endif

           q1k   = q1(k)          

           
           
           
           
           
           
           
           
           
           

           
           wt2           = min(one, max(zero, (zagl - (pblh1-100.))/200.)) 

           
           cldfra_qsq0   = max(zero, min(one, p5+0.35*atan(3.6*(q1k+0.05))))
           
           cldfra_qsq1   = max(zero, min(one, p5+0.39*atan(1.6*(q1k+0.55))))
           cldfra_qsq    = cldfra_qsq0*(one-wt2) + cldfra_qsq1*wt2

           
           cldfra_rh0    = min(one, max(zero, 0.56*tanh((rh(k)-0.976)/0.030)+p5))
           cldfra_rh1    = min(one, max(zero, 0.55*tanh((rh(k)-0.955)/0.055)+p5))
           cldfra_rh     = zero 

           cldfra_bl1(k) = max(cldfra_qsq, cldfra_rh)
           
           
           
           wt     = min(one, max(zero, (zagl - (pblh2+ten)))/300.) 
           qlim   = qlim_pbl*(one-wt) + qlim_trp*wt
           zsl    = min(150., max(50., p1*pblh2)) 
           wt     = min(one, max(zero, zagl - zsl)/200.)  
           qlim   = qlim_sfc*(one-wt) + qlim*wt
           sgmq   = max(sgmq, qw(k)*qlim)
           
           ql_water = min(sgmq, 0.025*qw(k))*cldfra_bl1(k)
           ql_ice   = min(sgmq, 0.025*qw(k))*cldfra_bl1(k)

           
















           
           
           
           

           if (cldfra_bl1(k) < 0.001) then
              ql_ice        = zero
              ql_water      = zero
              cldfra_bl1(k) = zero
           endif

           liq_frac = MIN(one, MAX(zero, (t-tice)/(tliq-tice)))
           qc_bl1(k) = liq_frac*ql_water       
           qi_bl1(k) = (one-liq_frac)*ql_ice

           
           
           if (k .ge. k_tropo) then
              cldfra_bl1(K) = zero
              qc_bl1(k)     = zero
              qi_bl1(k)     = zero
           endif

           
           
           
           
           if ((xland-1.5).GE.zero) then  
              q1k=max(Q1(k),-2.5)
           else                           
              q1k=max(Q1(k),-2.0)
           endif
           
           
           if (q1k .ge. one) then
              Fng = one
           elseif (q1k .ge. -1.7 .and. q1k .lt. one) then
              Fng = exp(-0.4*(q1k-one))
           elseif (q1k .ge. -2.5 .and. q1k .lt. -1.7) then
              Fng = three + exp(-3.8*(q1k+1.7))
           else
              Fng = min(23.9 + exp(-1.6*(q1k+2.5)), 60._kind_phys)
           endif

           cfmax = min(cldfra_bl1(k), 0.6_kind_phys)
           
           zsl   = min(max(25., p1*pblh2), hundred)
           wt    = min(zagl/zsl, one) 
           cfmax = cfmax*wt

           bb = b(k)*t/th(k) 
                             
                             
                             
                             
                             
           qww   = one + p608*qw(k)
           alpha = p608*th(k)
           beta  = (th(k)/t)*(xl/cp) - 1.61*th(k)
           vt(k) = qww   - cfmax*beta*bb*Fng   - one
           vq(k) = alpha + cfmax*beta*a(k)*Fng - tv0
           
           
           
           
           
        enddo

      END SELECT 

      
      IF (bl_mynn_cloudpdf .LT. 0) THEN
         DO k = kts,kte-1
            cldfra_bl1(k) = zero
            qc_bl1(k)     = zero
            qi_bl1(k)     = zero
         END DO
      ENDIF

      ql(kte)        = ql(kte-1)
      vt(kte)        = vt(kte-1)
      vq(kte)        = vq(kte-1)
      qc_bl1(kte)    = zero
      qi_bl1(kte)    = zero
      cldfra_bl1(kte)= zero
    RETURN


  END SUBROUTINE mym_condensation





  SUBROUTINE mynn_tendencies(kts,kte,i,       &
       &delt,dz,zw,xland,pblh,rho,            &
       &u,v,th,tk,qv,qc,qi,qs,qnc,qni,        &
       &psfc,p,exner,                         &
       &thl,sqv,sqc,sqi,sqs,sqw,              &
       &thl_tot1,qc_tot1,qi_tot1,             &
       &qnwfa,qnifa,qnbca,ozone,              &
       &ust,flt,flq,flqv,flqc,wspd,           &
       &uoce,voce,                            &
       &tsq,qsq,cov,                          &
       &tcd,qcd,                              &
       &dfm,dfh,dfq,                          &
       &Du,Dv,Dth,Dqv,Dqc,Dqi,Dqs,Dqnc,Dqni,  &
       &Dqnwfa,Dqnifa,Dqnbca,Dozone,          &
       &diss_heat,                            &
       &s_aw1,s_awthl1,                       &
       &s_awqt1,s_awqv1,s_awqc1,              &
       &s_awu1,s_awv1,                        &
       &s_awqnc1,s_awqni1,                    &
       &s_awqnwfa1,s_awqnifa1,s_awqnbca1,     &
       &sd_aw1,sd_awthl1,sd_awqt1,sd_awqv1,   &
       &sd_awqc1,sd_awqi1,                    &
       &sd_awqnc1,sd_awqni1,                  &
       &sd_awqnwfa1,sd_awqnifa1,              &
       &sd_awu1,sd_awv1,                      &
       &sub_thl,sub_sqv,                      &
       &sub_u,sub_v,                          &
       &det_thl,det_sqv,det_sqc,              &
       &det_u,det_v,                          &
       &FLAG_QC,FLAG_QI,FLAG_QNC,FLAG_QNI,    &
       &FLAG_QS,                              &
       &FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA,     &
       &FLAG_OZONE,                           &
       &cldfra_bl1,                           &
       &bl_mynn_cloudmix,                     &
       &bl_mynn_mixqt,                        &
       &bl_mynn_edmf,                         &
       &bl_mynn_edmf_mom,                     &
       &bl_mynn_mixscalars,                   &
       &bl_mynn_mixaerosols,                  &
       &bl_mynn_mixnumcon,                    &
       &upwind                                )


    use module_bl_mynnedmf_common, only: r_d,p608,grav,xlvcp,xlscp,       &
         wfa_max,wfa_min,ifa_max,ifa_min,wfa_ht,ifa_ht,zero,one,          &
         p333,p5,kind_phys
    
    integer, intent(in) :: kts,kte,i


    integer, intent(in) :: bl_mynn_cloudmix,bl_mynn_mixqt,                &
                           bl_mynn_edmf,bl_mynn_edmf_mom,                 &
                           bl_mynn_mixscalars,bl_mynn_mixaerosols,        &
                           bl_mynn_mixnumcon
    logical, intent(in) :: FLAG_QI,FLAG_QNI,FLAG_QC,FLAG_QS,              &
         &FLAG_QNC,FLAG_QNWFA,FLAG_QNIFA,FLAG_QNBCA,FLAG_OZONE








    real(kind_phys), dimension(kts:kte+1), intent(in) :: s_aw1,            &
         &s_awthl1,s_awqt1,s_awqnc1,s_awqni1,s_awqv1,s_awqc1,s_awu1,s_awv1,&
         &s_awqnwfa1,s_awqnifa1,s_awqnbca1,                                &
         &sd_aw1,sd_awthl1,sd_awqt1,sd_awqv1,sd_awqc1,sd_awqi1,            &
         &sd_awqnc1,sd_awqni1,sd_awqnwfa1,sd_awqnifa1,sd_awu1,sd_awv1

    real(kind_phys), dimension(kts:kte), intent(in) :: sub_thl,sub_sqv,   &
         &sub_u,sub_v,det_thl,det_sqv,det_sqc,det_u,det_v
    real(kind_phys), dimension(kts:kte), intent(in) :: u,v,th,tk,qv,qc,qi,&
         &qs,qni,qnc,rho,p,exner,dfq,dz,zw,tsq,qsq,cov,tcd,qcd,           &
         &cldfra_bl1,diss_heat,thl_tot1,qc_tot1,qi_tot1
    real(kind_phys), dimension(kts:kte), intent(inout) :: thl,sqw,sqv,sqc,&
         &sqi,sqs,qnwfa,qnifa,qnbca,ozone,dfm,dfh
    real(kind_phys), dimension(kts:kte), intent(inout) :: du,dv,dth,dqv,  &
         &dqc,dqi,dqs,dqni,dqnc,dqnwfa,dqnifa,dqnbca,dozone
    real(kind_phys), intent(in) :: flt,flq,flqv,flqc,uoce,voce
    real(kind_phys), intent(in) :: ust,delt,psfc,wspd,xland,pblh,upwind
    
    real(kind_phys):: wsp,wsp2,tk2,th2
    logical :: problem
    integer :: kproblem





    real(kind_phys), dimension(kts:kte) :: dtz,dfhc,dfmc,delp
    real(kind_phys), dimension(kts:kte) :: sqv2,sqc2,sqi2,sqs2,sqw2,      &
          &qni2,qnc2,qnwfa2,qnifa2,qnbca2,ozone2
    real(kind_phys), dimension(kts:kte) :: zfac,plumeKh,rhoinv
    real(kind_phys), dimension(kts:kte) :: upcont,dncont 
    real(kind_phys), dimension(kts:kte) :: a,b,c,d,x
    real(kind_phys), dimension(kts:kte+1) :: rhoz,                        & 
          &khdz,kmdz
    real(kind_phys):: rhs,gfluxm,gfluxp,dztop,maxdfh,mindfh,maxcf,maxKh
    real(kind_phys):: t,esat,qsl,onoff,kh,km,dzk,rhosfc,ustovwsp
    real(kind_phys):: qvflux,aero_min,aero_max
    real(kind_phys):: th_new,portion_qc,portion_qi,condensate,qsat
    integer :: k,kk

    
    
    real(kind_phys), parameter :: nonloc  = 1.0
    real(kind_phys), parameter :: nc_min  = 100.0
    real(kind_phys), parameter :: ni_min  = 1e-6
    real(kind_phys) :: wfa_max2, wt

    dztop=p5*(dz(kte)+dz(kte-1))

    
    
    
    IF (bl_mynn_edmf_mom == 0) THEN
       onoff=zero
    ELSE
       onoff=one
    ENDIF

    
    IF (wspd .le. 1e-6) THEN
       ustovwsp=zero
    ELSE 
       ustovwsp=ust/wspd
    ENDIF
    
    
    
    rhosfc     = psfc/(R_d*(tk(kts)+p608*qv(kts)))
    dtz(kts)   = delt/dz(kts)
    rhoz(kts)  = rho(kts)
    rhoinv(kts)= one/rho(kts)
    khdz(kts)  = rhoz(kts)*dfh(kts)
    kmdz(kts)  = rhoz(kts)*dfm(kts)
    DO k=kts+1,kte
       dtz(k)   = delt/dz(k)
       rhoz(k)  = (rho(k)*dz(k-1) + rho(k-1)*dz(k))/(dz(k-1)+dz(k))
       rhoz(k)  =  MAX(rhoz(k),1E-4_kind_phys)
       rhoinv(k)= one/MAX(rho(k),1E-4_kind_phys)
       dzk      = p5 *( dz(k)+dz(k-1) )
       khdz(k)  = rhoz(k)*dfh(k)
       kmdz(k)  = rhoz(k)*dfm(k)
    ENDDO
    rhoz(kte+1)= rhoz(kte)
    khdz(kte+1)= rhoz(kte+1)*dfh(kte)
    kmdz(kte+1)= rhoz(kte+1)*dfm(kte)

    
    
    DO k=kts,kte 
       
       
       delp(k) = rho(k)*grav*dz(k)
    ENDDO
    
    if ( delp(kts) < p333*delp(kts+1) )delp(kts)=p333*delp(kts+1)

    
    if (bl_mynn_edmf == 1) then
       DO k=kts+1,kte-1
          khdz(k) = max(khdz(k),  p5*(s_aw1(k) +sd_aw1(k)))
          khdz(k) = max(khdz(k), -p5*(s_aw1(k) -s_aw1(k+1))  &
                                 -p5*(sd_aw1(k)-sd_aw1(k+1)) )
          kmdz(k) = max(kmdz(k),  p5*(s_aw1(k) +sd_aw1(k)))
          kmdz(k) = max(kmdz(k), -p5*(s_aw1(k) -s_aw1(k+1))  &
                                 -p5*(sd_aw1(k)-sd_aw1(k+1)) )
       ENDDO
    endif

    dth(kts:kte) = zero  




 if (bl_mynn_edmf > 1) then

    do k=kts+1,kte-1
       upcont(k)=onoff*(s_awu1(k) - s_aw1(k)*(u(k)*upwind+u(k-1)*(one-upwind)))
       dncont(k)=onoff*(sd_awu1(k)-sd_aw1(k)*(u(k)*upwind+u(k-1)*(one-upwind)))
    enddo
    
    upcont(kte)=zero
    dncont(kte)=zero


    k=kts
    a(1)=zero
    b(1)=one + dtz(k)*(kmdz(k+1)+rhosfc*ust*ustovwsp)*rhoinv(k) 
    c(1)=    - dtz(k)*kmdz(k+1)*rhoinv(k)
    d(1)=u(k)+ dtz(k)*uoce*ust**2/wspd*rho(k)                      &
             - dtz(k)*(upcont(k+1)+dncont(k+1))

    DO k=kts+1,kte-1
       a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)
       b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k) 
       c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)
       d(k)=u(k) -dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
           &  + sub_u(k)*delt + det_u(k)*delt
    ENDDO

 else 
    
    k=kts
    
    a(k)=  -dtz(k)*kmdz(k)*rhoinv(k)
    b(k)=one+dtz(k)*(kmdz(k+1)+rhosfc*ust*ustovwsp)*rhoinv(k)      &
           & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff                &
           & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
    c(k)=  -dtz(k)*kmdz(k+1)*rhoinv(k)                             &
           & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff                &
           & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
    d(k)=u(k)  + dtz(k)*uoce*ust**2/wspd                           &
           & - dtz(k)*rhoinv(k)*s_awu1(k+1)*onoff                  &
           & + dtz(k)*rhoinv(k)*sd_awu1(k+1)*onoff                 &
           & + sub_u(k)*delt + det_u(k)*delt

    do k=kts+1,kte-1
       a(k)=  -dtz(k)*kmdz(k)*rhoinv(k)                            &
           &  + p5*dtz(k)*rhoinv(k)*s_aw1(k)*onoff                 &
           &  + p5*dtz(k)*rhoinv(k)*sd_aw1(k)*onoff
       b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k)              &
           &  + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))*onoff    &
           &  + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))*onoff
       c(k)=  - dtz(k)*kmdz(k+1)*rhoinv(k)                         &
           &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff               &
           &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
       d(k)=u(k) + dtz(k)*rhoinv(k)*(s_awu1(k)-s_awu1(k+1))*onoff  &
           &  - dtz(k)*rhoinv(k)*(sd_awu1(k)-sd_awu1(k+1))*onoff   &
           &  + sub_u(k)*delt + det_u(k)*delt
    enddo

 endif
 













    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=u(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte

       du(k)=(x(k)-u(k))/delt
    ENDDO





 if (bl_mynn_edmf > 1) then

    DO k=kts+1,kte-1
       upcont(k)=onoff*(s_awv1(k)-s_aw1(k)*(v(k)*upwind+v(k-1)*(one-upwind)))
       dncont(k)=onoff*(sd_awv1(k)-sd_aw1(k)*(v(k)*upwind+v(k-1)*(one-upwind)))
    ENDDO
    
    upcont(kte)=0.
    dncont(kte)=0.


    k=kts
    a(1)=zero
    b(1)=one + dtz(k)*(kmdz(k+1)+rhosfc*ust*ustovwsp)*rhoinv(k)
    c(1)=    - dtz(k)*kmdz(k+1)*rhoinv(k)
    d(1)=v(k)+ dtz(k)*voce*ust**2/wspd*rho(k)                      &
             - dtz(k)*(upcont(k+1)+dncont(k+1))

    DO k=kts+1,kte-1
       a(k)=   - dtz(k)*kmdz(k)*rhoinv(k)
       b(k)=one+ dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k)
       c(k)=   - dtz(k)*kmdz(k+1)*rhoinv(k)
       d(k)=v(k) -dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
           &  + sub_v(k)*delt + det_v(k)*delt
    ENDDO

 else 

    k=kts
    
    a(k)=  -dtz(k)*kmdz(k)*rhoinv(k)
    b(k)=one+dtz(k)*(kmdz(k+1) + rhosfc*ust**2/wspd)*rhoinv(k)    &
        &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff                 &
        &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
    c(k)=  -dtz(k)*kmdz(k+1)*rhoinv(k)                            &
        &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff                 &
        &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
    d(k)=v(k)  + dtz(k)*voce*ust**2/wspd                          &
        &  - dtz(k)*rhoinv(k)*s_awv1(k+1)*onoff                   &
        &  + dtz(k)*rhoinv(k)*sd_awv1(k+1)*onoff                  &
        &  + sub_v(k)*delt + det_v(k)*delt

    do k=kts+1,kte-1
       a(k)=  -dtz(k)*kmdz(k)*rhoinv(k)                           &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)*onoff                   &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)*onoff
       b(k)=one+dtz(k)*(kmdz(k)+kmdz(k+1))*rhoinv(k)              &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))*onoff      &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))*onoff
       c(k)=  -dtz(k)*kmdz(k+1)*rhoinv(k)                         &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*onoff                 &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)*onoff
       d(k)=v(k) + dtz(k)*rhoinv(k)*(s_awv1(k)-s_awv1(k+1))*onoff &
         & - dtz(k)*rhoinv(k)*(sd_awv1(k)-sd_awv1(k+1))*onoff     &
         & + sub_v(k)*delt + det_v(k)*delt
    enddo
 endif













    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=v(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte

       dv(k)=(x(k)-v(k))/delt
    ENDDO




 if (bl_mynn_edmf > 1) then

    DO k=kts+1,kte-1
       upcont(k)= s_awthl1(k)- s_aw1(k)*(thl(k)*upwind+thl(k-1)*(one-upwind))
       dncont(k)=sd_awthl1(k)-sd_aw1(k)*(thl(k)*upwind+thl(k-1)*(one-upwind))
    ENDDO
    
    upcont(kte)=zero
    dncont(kte)=zero


    k=kts
    a(1)=zero
    b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
    c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
    d(1)=thl(k)+dtz(k)*rhosfc*flt*rhoinv(k) + tcd(k)*delt              &
         - dtz(k)*(upcont(k+1)+dncont(k+1))                            &
         + diss_heat(k)*delt + sub_thl(k)*delt + det_thl(k)*delt

    DO k=kts+1,kte-1
       a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
       b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
       c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
       d(k)=thl(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
           &  + diss_heat(k)*delt + sub_thl(k)*delt + det_thl(k)*delt
    ENDDO

 else 
    
    k=kts

    a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
    b(k)=1.+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)             &
       &   - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
       &   - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
    c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                       &
       &   - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
       &   - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
    d(k)=thl(k) + dtz(k)*rhosfc*flt*rhoinv(k) + tcd(k)*delt  &

       &   - dtz(k)*rhoinv(k)*s_awthl1(k+1)                  &
       &   + dtz(k)*rhoinv(k)*sd_awthl1(k+1)                 &
       &   + diss_heat(k)*delt + sub_thl(k)*delt + det_thl(k)*delt

    do k=kts+1,kte-1
       a(k)= -dtz(k)*khdz(k)*rhoinv(k)                       &
       &   + p5*dtz(k)*rhoinv(k)*s_aw1(k)                    &
       &   + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
       b(k)=1.+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)          &
       &   + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))       &
       &   + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
       c(k)= -dtz(k)*khdz(k+1)*rhoinv(k)                     &
       &   - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
       &   - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
       d(k)=thl(k) + tcd(k)*delt                             &

       &   + dtz(k)*rhoinv(k)*(s_awthl1(k)-s_awthl1(k+1))    &
       &   - dtz(k)*rhoinv(k)*(sd_awthl1(k)-sd_awthl1(k+1))  &
       &   +     diss_heat(k)*delt                           &
       &   +     sub_thl(k)*delt + det_thl(k)*delt
    enddo
 endif














    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=thl(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       thl(k)=x(k)
    ENDDO

IF (bl_mynn_mixqt > 0) THEN
 
 
 
 
 
 
 if (bl_mynn_edmf > 1) then

    DO k=kts+1,kte-1
       upcont(k)= s_awqt1(k)- s_aw1(k)*(sqw(k)*upwind+sqw(k-1)*(one-upwind))
       dncont(k)=sd_awqt1(k)-sd_aw1(k)*(sqw(k)*upwind+sqw(k-1)*(one-upwind))
    ENDDO
    
    upcont(kte)=zero
    dncont(kte)=zero


    k=kts
    a(1)=zero
    b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
    c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
    d(1)=sqw(k)+dtz(k)*rhosfc*flq*rhoinv(k) + qcd(k)*delt              &
         - dtz(k)*(upcont(k+1)+dncont(k+1))                            &
         + sub_sqv(k)*delt + det_sqv(k)*delt

    DO k=kts+1,kte-1
       a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
       b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
       c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
       d(k)=sqw(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
           &  + sub_sqv(k)*delt + det_sqv(k)*delt
    ENDDO

 else 

    k=kts
    
    a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
    b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)         &
       & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                 &
       & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
    c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                    &
       & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                 &
       & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
    d(k)=sqw(k)  + dtz(k)*rhosfc*flq*rhoinv(k) + qcd(k)*delt &
       &  - dtz(k)*rhoinv(k)*s_awqt1(k+1)                 &
       &  + dtz(k)*rhoinv(k)*sd_awqt1(k+1)

    do k=kts+1,kte-1
       a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                   &
       & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                   &
       & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
       b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)      &
       & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))      &
       & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
       c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                 &
       & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                 &
       & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
       d(k)=sqw(k) + qcd(k)*delt                          &
       & + dtz(k)*rhoinv(k)*(s_awqt1(k)-s_awqt1(k+1))     &
       & - dtz(k)*rhoinv(k)*(sd_awqt1(k)-sd_awqt1(k+1))
    enddo
endif












    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=sqw(kte)


    CALL tridiag2(kte,a,b,c,d,sqw2)





ELSE
    sqw2=sqw
ENDIF

IF (bl_mynn_mixqt == 0) THEN




 IF (bl_mynn_cloudmix > 0 .AND. FLAG_QC) THEN

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqc1(k)- s_aw1(k)*(sqc(k)*upwind+sqc(k-1)*(one-upwind))
         dncont(k)=sd_awqc1(k)-sd_aw1(k)*(sqc(k)*upwind+sqc(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=sqc(k)+dtz(k)*rhosfc*flqc*rhoinv(k) + qcd(k)*delt             &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))                        &
          &    + det_sqc(k)*delt
    
      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=sqc(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
             &  + det_sqc(k)*delt
      ENDDO

    else 

       k=kts
       
       a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
       b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)        &
       &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
       &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
       c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                   &
       &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
       &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
       d(k)=sqc(k)  + dtz(k)*rhosfc*flqc*rhoinv(k) + qcd(k)*delt &
       &  - dtz(k)*rhoinv(k)*s_awqc1(k+1)                   &
       &  + dtz(k)*rhoinv(k)*sd_awqc1(k+1)                  &
       &  + det_sqc(k)*delt

       do k=kts+1,kte-1
          a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                  &
          & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                  &
          & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
          b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)     &
          & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))     &
          & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
          c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                &
          & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                &
          & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
          d(k)=sqc(k) + qcd(k)*delt                         &
          & + dtz(k)*rhoinv(k)*(s_awqc1(k)-s_awqc1(k+1))    &
          & - dtz(k)*rhoinv(k)*(sd_awqc1(k)-sd_awqc1(k+1))  &
          & + det_sqc(k)*delt
       enddo
    endif
    

    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=sqc(kte)


    CALL tridiag2(kte,a,b,c,d,sqc2)





  ELSE
    
    sqc2=sqc
  ENDIF
ENDIF

IF (bl_mynn_mixqt == 0) THEN
  
  
  
  

   k=kts
   
   qvflux = flqv
   if (qvflux < zero) then
      
      qvflux = max(qvflux, (min(0.9*sqv(kts) - 1e-8, zero)/dtz(kts)))
   endif

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqv1(k)- s_aw1(k)*(sqv(k)*upwind+sqv(k-1)*(one-upwind))
         dncont(k)=sd_awqv1(k)-sd_aw1(k)*(sqv(k)*upwind+sqv(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=sqv(k)+dtz(k)*rhosfc*qvflux*rhoinv(k) + qcd(k)*delt             &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))			   &    
          &    + sub_sqv(k)*delt + det_sqv(k)*delt

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=sqv(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k)) &
             &   + sub_sqv(k)*delt + det_sqv(k)*delt
      ENDDO

   else 

      k=kts
      
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)        &
      & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                   &
      & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                   &
      & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                   &
      & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=sqv(k)  + dtz(k)*rhosfc*qvflux*rhoinv(k) + qcd(k)*delt &
      & - dtz(k)*rhoinv(k)*s_awqv1(k+1)                    &
      & + dtz(k)*rhoinv(k)*sd_awqv1(k+1)                   &
      & + sub_sqv(k)*delt + det_sqv(k)*delt

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                  &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                  &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)     &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))     &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=sqv(k) + qcd(k)*delt                         &
         & + dtz(k)*rhoinv(k)*(s_awqv1(k)-s_awqv1(k+1))    &
         & - dtz(k)*rhoinv(k)*(sd_awqv1(k)-sd_awqv1(k+1))  &
         & + sub_sqv(k)*delt + det_sqv(k)*delt
      enddo
   endif
   














    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=sqv(kte)


    CALL tridiag2(kte,a,b,c,d,sqv2)





ELSE
    sqv2=sqv
ENDIF




IF (bl_mynn_cloudmix > 0 .AND. FLAG_QI) THEN

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= zero 
         dncont(k)=sd_awqi1(k)-sd_aw1(k)*(sqi(k)*upwind+sqi(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=sqi(k)                                         &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=sqi(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
      
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)      &

      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                 &

      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=sqi(k)                                        &

      &  + dtz(k)*rhoinv(k)*sd_awqi1(k+1)

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                &

         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)   &

         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)              &

         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=sqi(k)                                     &

         & - dtz(k)*rhoinv(k)*(sd_awqi1(k)-sd_awqi1(k+1))
      enddo
   endif















    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=sqi(kte)


    CALL tridiag2(kte,a,b,c,d,sqi2)





ELSE
   sqi2=sqi
ENDIF





IF (bl_mynn_cloudmix > 0 .AND. .false.) THEN

    k=kts

    a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
    b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)
    c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)
    d(k)=sqs(k)

    DO k=kts+1,kte-1
       a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
       b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
       c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)
       d(k)=sqs(k)
    ENDDO


    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=sqs(kte)


    CALL tridiag2(kte,a,b,c,d,sqs2)





ELSE
   sqs2=sqs
ENDIF




IF (bl_mynn_cloudmix > 0 .AND. FLAG_QNI .AND. &
      bl_mynn_mixnumcon > 0) THEN

   DO k=kts,kte
      qni2(k)=max(qni(k), zero)
      
      if (sqi(k) .gt. 1e-12)qni2(k)=max(qni2(k), ni_min)
   ENDDO

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqni1(k)- s_aw1(k)*(qni2(k)*upwind+qni2(k-1)*(one-upwind))
         dncont(k)=sd_awqni1(k)-sd_aw1(k)*(qni2(k)*upwind+qni2(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=qni2(k)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=qni2(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
      
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)        &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                   &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=qni2(k)                                         &
      &  - dtz(k)*rhoinv(k)*s_awqni1(k+1)                  &
      &  + dtz(k)*rhoinv(k)*sd_awqni1(k+1)

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                  &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                  &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)     &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))     &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=qni2(k)                                      &
         & + dtz(k)*rhoinv(k)*(s_awqni1(k)-s_awqni1(k+1))  &
         & - dtz(k)*rhoinv(k)*(sd_awqni1(k)-sd_awqni1(k+1))
      enddo
   endif
   

    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=qni2(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       qni2(k)=x(k)
       
       if (sqi2(k) .gt. 1e-12)qni2(k)=max(qni2(k), ni_min)
    ENDDO

ELSE
    qni2=qni
ENDIF





IF (bl_mynn_cloudmix > 0 .AND. FLAG_QNC .AND. &
    bl_mynn_mixnumcon > 0) THEN

   DO k=kts,kte
      qnc2(k)=max(qnc(k),zero)
      
      if (sqc(k) .gt. 1e-12)qnc2(k)=max(qnc2(k), nc_min)
   ENDDO

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqnc1(k)- s_aw1(k)*(qnc2(k)*upwind+qnc2(k-1)*(one-upwind))
         dncont(k)=sd_awqnc1(k)-sd_aw1(k)*(qnc2(k)*upwind+qnc2(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=qnc2(k)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=qnc2(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
    
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)        &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                   &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                  &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=qnc2(k)                                         &
      &  - dtz(k)*rhoinv(k)*s_awqnc1(k+1)                  &
      &  + dtz(k)*rhoinv(k)*sd_awqnc1(k+1)

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                  &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                  &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)     &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))     &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=qnc2(k)                                      &
         & + dtz(k)*rhoinv(k)*(s_awqnc1(k)-s_awqnc1(k+1))  &
         & - dtz(k)*rhoinv(k)*(sd_awqnc1(k)-sd_awqnc1(k+1))
      enddo
   endif


    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=qnc2(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       qnc2(k)=x(k)
       
       if (sqc2(k) .gt. 1e-12)qnc2(k)=max(qnc2(k), nc_min)
    ENDDO

ELSE
    qnc2=qnc
ENDIF




IF (FLAG_QNWFA .AND. bl_mynn_mixaerosols > 0) THEN

   do k=kts,kte
      qnwfa2(k)=max(qnwfa(k),zero)
   enddo

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqnwfa1(k)- s_aw1(k)*(qnwfa2(k)*upwind+qnwfa2(k-1)*(one-upwind))
         dncont(k)=sd_awqnwfa1(k)-sd_aw1(k)*(qnwfa2(k)*upwind+qnwfa2(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero


      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=qnwfa2(k)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=qnwfa2(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
      
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)            &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                      &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                       &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                      &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=qnwfa2(k)                                           &
      &  - dtz(k)*rhoinv(k)*s_awqnwfa1(k+1)                    &
      &  + dtz(k)*rhoinv(k)*sd_awqnwfa1(k+1)

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                      &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                      &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)         &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))         &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                    &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                    &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=qnwfa2(k)                                        &
         & + dtz(k)*rhoinv(k)*(s_awqnwfa1(k)-s_awqnwfa1(k+1))  &
         & - dtz(k)*rhoinv(k)*(sd_awqnwfa1(k)-sd_awqnwfa1(k+1))
      enddo
   endif
      

    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=qnwfa2(kte)


    call tridiag2(kte,a,b,c,d,x)


    do k=kts,kte
       
       qnwfa2(k)=max(x(k),zero)
    enddo

    
    do k=kts,kte
       
       if ((xland-1.5).ge.0)then   
          wt       = min(one, max(zero, zw(k)-100._kind_phys)/max(200._kind_phys, pblh)) 
          wfa_max2 = (one - wt)*6.e8_kind_phys + wt*wfa_max
       else                        
          wfa_max2 = wfa_max
       endif
       aero_min = wfa_min * exp(-zw(k)/wfa_ht)
       aero_max = wfa_max2* exp(-zw(k)/wfa_ht)
       qnwfa2(k)= min(max(aero_min, qnwfa2(k)), aero_max)
    enddo
else
    
    qnwfa2=qnwfa
endif




IF (FLAG_QNIFA .AND. bl_mynn_mixaerosols > 0) THEN

   do k=kts,kte
      qnifa2(k)=max(qnifa(k),zero)
   enddo

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqnifa1(k)- s_aw1(k)*(qnifa2(k)*upwind+qnifa2(k-1)*(one-upwind))
         dncont(k)=sd_awqnifa1(k)-sd_aw1(k)*(qnifa2(k)*upwind+qnifa2(k-1)*(one-upwind))
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero

      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=qnifa2(k)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=qnifa2(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
    
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)            &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                      &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                       &
      &  - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                      &
      &  - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
      d(k)=qnifa2(k)                                           &
      &  - dtz(k)*rhoinv(k)*s_awqnifa1(k+1)                    &
      &  + dtz(k)*rhoinv(k)*sd_awqnifa1(k+1)

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                      &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)                      &
         & + p5*dtz(k)*rhoinv(k)*sd_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)         &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))         &
         & + p5*dtz(k)*rhoinv(k)*(sd_aw1(k)-sd_aw1(k+1))
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                    &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)                    &
         & - p5*dtz(k)*rhoinv(k)*sd_aw1(k+1)
         d(k)=qnifa2(k)                                        &
         & + dtz(k)*rhoinv(k)*(s_awqnifa1(k)-s_awqnifa1(k+1))  &
         & - dtz(k)*rhoinv(k)*(sd_awqnifa1(k)-sd_awqnifa1(k+1))
      enddo
   endif
   

    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=qnifa2(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       qnifa2(k)=max(x(k),zero)
    ENDDO

    
    DO k=kts,kte
       aero_min = ifa_min * exp(-zw(k)/ifa_ht)
       aero_max = ifa_max * exp(-zw(k)/ifa_ht)
       qnifa2(k)= min(max(aero_min, qnifa2(k)), aero_max)
    ENDDO
    
ELSE
    
    qnifa2=qnifa
ENDIF




IF (FLAG_QNBCA .AND. bl_mynn_mixaerosols > 0) THEN

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awqnbca1(k)- s_aw1(k)*(qnbca(k)*upwind+qnbca(k-1)*(one-upwind))
         dncont(k)=zero 
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero

      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=qnbca(k)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=qnbca(k)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
   
      k=kts
      a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k) + khdz(k+1))*rhoinv(k)          &
      & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*nonloc
      c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                       &
      & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*nonloc
      d(k)=qnbca(k)  - dtz(k)*rhoinv(k)*s_awqnbca1(k+1)*nonloc

      do k=kts+1,kte-1
         a(k)=  -dtz(k)*khdz(k)*rhoinv(k)                      &
         & + p5*dtz(k)*rhoinv(k)*s_aw1(k)*nonloc
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)         &
         & + p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))*nonloc
         c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)                    &
         & - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)*nonloc
         d(k)=qnbca(k) + dtz(k)*rhoinv(k)*(s_awqnbca1(k)-s_awqnbca1(k+1))*nonloc
      enddo
   endif
   

    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=qnbca(kte)


   CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       qnbca2(k)=max(zero, x(k))
    ENDDO

ELSE
    
    qnbca2=qnbca
ENDIF




IF (FLAG_OZONE) THEN
    k=kts


    a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
    b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k)
    c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)
    d(k)=ozone(k)

    DO k=kts+1,kte-1
       a(k)=  -dtz(k)*khdz(k)*rhoinv(k)
       b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
       c(k)=  -dtz(k)*khdz(k+1)*rhoinv(k)
       d(k)=ozone(k)
    ENDDO


    a(kte)=zero
    b(kte)=one
    c(kte)=zero
    d(kte)=ozone(kte)


    CALL tridiag2(kte,a,b,c,d,x)


    DO k=kts,kte
       
       dozone(k)=(max(x(k),zero) - ozone(k))/delt
    ENDDO
ELSE
    dozone(:)=zero
ENDIF






   IF (bl_mynn_mixqt > 0) THEN 
      DO k=kts,kte
         
         th_new = thl(k) + xlvcp/exner(k)*sqc(k) &
           &             + xlscp/exner(k)*sqi(k)

         t  = th_new*exner(k)
         qsat = qsat_blend(t,p(k)) 
         
         
         
         
         

         IF (sqc(k) > zero .or. sqi(k) > zero) THEN 
            sqv2(k) = MIN(sqw2(k),qsat)
            portion_qc = sqc(k)/(sqc(k) + sqi(k))
            portion_qi = sqi(k)/(sqc(k) + sqi(k))
            condensate = MAX(sqw2(k) - qsat, zero)
            sqc2(k) = condensate*portion_qc
            sqi2(k) = condensate*portion_qi
         ELSE                     
            sqv2(k) = sqw2(k)     
            sqi2(k) = zero         
            sqc2(k) = zero
         ENDIF
      ENDDO
   ENDIF


    
    
    
    DO k=kts,kte
       Dqv(k)=(sqv2(k) - sqv(k))/delt
       
    ENDDO

    IF (bl_mynn_cloudmix > 0) THEN
      
      
      
      
      IF (FLAG_QC) THEN
         DO k=kts,kte
            Dqc(k)=(sqc2(k) - sqc(k))/delt
            
         ENDDO
      ELSE
         DO k=kts,kte
           Dqc(k) = 0.
         ENDDO
      ENDIF

      
      
      
      IF (FLAG_QNC .AND. bl_mynn_mixnumcon > 0) THEN
         DO k=kts,kte
           Dqnc(k) = (qnc2(k)-qnc(k))/delt
           
         ENDDO 
      ELSE
         DO k=kts,kte
           Dqnc(k) = 0.
         ENDDO
      ENDIF

      
      
      
      IF (FLAG_QI) THEN
         DO k=kts,kte
           Dqi(k)=(sqi2(k) - sqi(k))/delt
           
         ENDDO
      ELSE
         DO k=kts,kte
           Dqi(k) = 0.
         ENDDO
      ENDIF

      
      
      
      IF (.false.) THEN 
         DO k=kts,kte
           Dqs(k)=(sqs2(k) - sqs(k))/delt
         ENDDO
      ELSE
         DO k=kts,kte
           Dqs(k) = 0.
         ENDDO
      ENDIF

      
      
      
      IF (FLAG_QNI .AND. bl_mynn_mixnumcon > 0) THEN
         DO k=kts,kte
           Dqni(k)=(qni2(k)-qni(k))/delt
           
         ENDDO
      ELSE
         DO k=kts,kte
           Dqni(k)=0.
         ENDDO
      ENDIF
    ELSE 
      
      DO k=kts,kte
         Dqc(k) =0.
         Dqnc(k)=0.
         Dqi(k) =0.
         Dqni(k)=0.
         Dqs(k) =0.
      ENDDO
    ENDIF

    
    CALL moisture_check(kte, delt, delp, exner,        &
                        sqv2, sqc2, sqi2, sqs2, thl,   &
                        dqv, dqc, dqi, dqs, dth        )

    
    
    
    DO k=kts,kte
       IF(Dozone(k)*delt + ozone(k) < 0.) THEN
         Dozone(k)=-ozone(k)*0.99/delt
       ENDIF
    ENDDO

    
    
    
    IF (FLAG_QI) THEN
      DO k=kts,kte
         Dth(k)=(thl(k) + xlvcp/exner(k)*sqc(k)          &
           &            + xlscp/exner(k)*(sqi(k))        & 
           &            - th(k))/delt
         
         
         
         
         
         
         
         
      ENDDO
    ELSE
      DO k=kts,kte
         Dth(k)=(thl(k)+xlvcp/exner(k)*sqc2(k) - th(k))/delt
         
         
         
         
      ENDDO
    ENDIF

    
    
    
    IF (FLAG_QNWFA .AND. FLAG_QNIFA .AND. &
        bl_mynn_mixaerosols > 0) THEN
       DO k=kts,kte
          
          
          
          Dqnwfa(k)=(qnwfa2(k) - qnwfa(k))/delt
          
          
          
          Dqnifa(k)=(qnifa2(k) - qnifa(k))/delt
       ENDDO
    ELSE
       DO k=kts,kte
          Dqnwfa(k)=0.
          Dqnifa(k)=0.
       ENDDO
    ENDIF

    
    
    
    IF (FLAG_QNBCA .AND. bl_mynn_mixaerosols > 0) THEN
       DO k=kts,kte
          Dqnbca(k)=(qnbca2(k) - qnbca(k))/delt
       ENDDO
    ELSE
       DO k=kts,kte
          Dqnbca(k)=0.
       ENDDO
    ENDIF

    
    
    
    
    
    

    if (debug_code) then
       problem = .false.
       do k=kts,kte
          wsp  = sqrt(u(k)**2 + v(k)**2)
          wsp2 = sqrt((u(k)+du(k)*delt)**2 + (v(k)+du(k)*delt)**2)
          th2  = th(k) + Dth(k)*delt
          tk2  = th2*exner(k)
          if (wsp2 > 200. .or. tk2 > 360. .or. tk2 < 160.) then
             problem = .true.
             print*,"After tendencies problem at: i=",i," k=",k
             print*," wsp=",wsp," updated wsp=",wsp2
             print*," T=",th(k)*exner(k)," updated T:",tk2
             print*," du=",du(k)*delt," dv=",dv(k)*delt," dth=",dth(k)*delt
             print*," km=",kmdz(k)*dz(k)," kh=",khdz(k)*dz(k)
             print*," u*=",ust," wspd=",wspd
             print*," rhosfc=",rhosfc," delp=",delp(k)
             print*," LH=",flq*rhosfc*1004.," HFX=",flt*rhosfc*1004.
             print*," flq=",flq," flt=",flt," exner=",exner(k)
             print*," drag term=",ust**2/wspd*dtz(k)*rhosfc/rho(kts)
             kproblem = k
          endif
       enddo
       if (problem) then
          print*,"==thl:",thl(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qv:",sqv2(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qc:",sqc2(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"===qi:",sqi2(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"====u:",u(max(kproblem-3,1):min(kproblem+3,kte))
          print*,"====v:",v(max(kproblem-3,1):min(kproblem+3,kte))
       endif
    endif


  END SUBROUTINE mynn_tendencies


  SUBROUTINE moisture_check(kte, delt, dp, exner,   &
                            qv, qc, qi, qs, th,     &
                            dqv, dqc, dqi, dqs, dth )

  
  
  
  
  
  
  
  
  
  
  
  
    use module_bl_mynnedmf_common, only: xlvcp,xlscp,zero,two,p5,kind_phys
    
    implicit none
    integer,         intent(in)     :: kte
    real(kind_phys), intent(in)     :: delt
    real(kind_phys), dimension(kte), intent(in)     :: dp, exner
    real(kind_phys), dimension(kte), intent(inout)  :: qv, qc, qi, qs, th
    real(kind_phys), dimension(kte), intent(inout)  :: dqv, dqc, dqi, dqs, dth
    integer:: k
    real(kind_phys)::  dqc2, dqi2, dqs2, dqv2, sum, aa, dum
    real(kind_phys), parameter :: qvmin = 1e-20,       &
                                  qcmin = zero,         &
                                  qimin = zero

    do k = kte, 1, -1  
       dqc2 = max(zero, qcmin-qc(k)) 
       dqi2 = max(zero, qimin-qi(k)) 
       dqs2 = max(zero, qimin-qs(k)) 

       
       dqc(k) = dqc(k) +  dqc2/delt
       dqi(k) = dqi(k) +  dqi2/delt
       dqs(k) = dqs(k) +  dqs2/delt
       dqv(k) = dqv(k) - (dqc2+dqi2+dqs2)/delt
       dth(k) = dth(k) + xlvcp/exner(k)*(dqc2/delt) + &
                         xlscp/exner(k)*((dqi2+dqs2)/delt)
       
       qc(k)  = qc(k)  +  dqc2
       qi(k)  = qi(k)  +  dqi2
       qs(k)  = qs(k)  +  dqs2
       qv(k)  = qv(k)  -  dqc2 - dqi2 - dqs2
       th(k)  = th(k)  +  xlvcp/exner(k)*dqc2 + &
                          xlscp/exner(k)*(dqi2+dqs2)

       
       dqv2   = max(zero, qvmin-qv(k)) 
       dqv(k) = dqv(k) + dqv2/delt
       qv(k)  = qv(k)  + dqv2
       if ( k .ne. 1 ) then
          
          qv(k-1)   = qv(k-1)  - dqv2*dp(k)/dp(k-1)
          dqv(k-1)  = dqv(k-1) - dqv2*dp(k)/dp(k-1)/delt
       endif
       qv(k) = max(qv(k),qvmin)
       qc(k) = max(qc(k),qcmin)
       qi(k) = max(qi(k),qimin)
       qs(k) = max(qs(k),qimin)
    end do


    
    if( dqv2 .gt. 1.e-20 ) then
        sum = zero
        do k = 1, kte
           if( qv(k) .gt. two*qvmin ) sum = sum + qv(k)*dp(k)
        enddo
        aa = dqv2*dp(1)/max(1.e-20_kind_phys,sum)
        if( aa .lt. p5 ) then
            do k = 1, kte
               if( qv(k) .gt. two*qvmin ) then
                   dum    = aa*qv(k)
                   qv(k)  = qv(k) - dum
                   dqv(k) = dqv(k) - dum/delt
               endif
            enddo
        else
        

        endif
    endif

    return

  END SUBROUTINE moisture_check



SUBROUTINE mynn_mix_chem(kts,kte,i,     &
     delt,dz,pblh,                      &
     nchem, ndvel,                      &
     chem1, vd1, settle1,               &
     rho,                               &
     flt, tcd, qcd,                     &
     dfh,                               &
     s_aw1, s_awchem1,                  &
     emis_ant_no, frp,                  &
     enh_mix,                           &
     bl_mynn_edmf, upwind               )


  use module_bl_mynnedmf_common, only: zero,one,two,p5,kind_phys
  
integer, intent(in) :: kts,kte,i,bl_mynn_edmf
real(kind_phys), dimension(kts:kte), intent(in) :: dfh,dz,tcd,qcd
real(kind_phys), dimension(kts:kte), intent(in) :: rho
real(kind_phys), intent(in)    :: flt
real(kind_phys), intent(in)    :: delt,pblh,upwind
integer, intent(in) :: nchem, ndvel
real(kind_phys), dimension( kts:kte+1), intent(in) :: s_aw1
real(kind_phys), dimension( kts:kte, nchem ), intent(inout) :: chem1
real(kind_phys), dimension( kts:kte, nchem ), intent(in) :: settle1
real(kind_phys), dimension( kts:kte+1,nchem), intent(in) :: s_awchem1
real(kind_phys), dimension( ndvel ), intent(in) :: vd1
real(kind_phys), intent(in) :: emis_ant_no,frp
logical, intent(in) :: enh_mix

real(kind_phys), dimension(kts:kte) :: dtz,upcont,dncont
real(kind_phys), dimension(kts:kte) :: a,b,c,d,x
real(kind_phys):: rhs,dztop
real(kind_phys):: t,dzk
real(kind_phys):: hght 
real(kind_phys):: khdz_old, khdz_back
integer :: k,kk,kmaxfire                         
integer :: ic  
    
integer, SAVE :: icall

real(kind_phys), dimension(kts:kte) :: rhoinv
real(kind_phys), dimension(kts:kte+1) :: rhoz,khdz
real(kind_phys), parameter :: NO_threshold    = 0.1      
real(kind_phys), parameter :: frp_threshold   = 10.0     
real(kind_phys), parameter :: pblh_threshold  = 100.0

dztop=.5*(dz(kte)+dz(kte-1))

DO k=kts,kte
   dtz(k)=delt/dz(k)
ENDDO



rhoz(kts)  =rho(kts)
rhoinv(kts)=one/rho(kts)
khdz(kts)  =rhoz(kts)*dfh(kts)

DO k=kts+1,kte
   rhoz(k)  =(rho(k)*dz(k-1) + rho(k-1)*dz(k))/(dz(k-1)+dz(k))
   rhoz(k)  =  MAX(rhoz(k),1E-4_kind_phys)
   rhoinv(k)=one/MAX(rho(k),1E-4_kind_phys)
   dzk      = p5  *( dz(k)+dz(k-1) )
   khdz(k)  = rhoz(k)*dfh(k)
ENDDO
rhoz(kte+1)=rhoz(kte)
khdz(kte+1)=rhoz(kte+1)*dfh(kte)

if (bl_mynn_edmf == 1) then
   
   DO k=kts+1,kte-1
      khdz(k) = MAX(khdz(k),  p5*s_aw1(k))
      khdz(k) = MAX(khdz(k), -p5*(s_aw1(k)-s_aw1(k+1)))
   ENDDO
endif


IF ( enh_mix ) THEN
   DO k=kts+1,kte-1
      khdz_old  = khdz(k)
      khdz_back = pblh * 0.15_kind_phys / dz(k)
      
      IF ( pblh < pblh_threshold ) THEN
         IF ( emis_ant_no > NO_threshold ) THEN
            khdz(k) = MAX(1.1*khdz(k),sqrt((emis_ant_no / NO_threshold)) / dz(k) * rhoz(k)) 

         ENDIF
         IF ( frp > frp_threshold ) THEN
            kmaxfire = ceiling(log(frp))
            khdz(k) = MAX(1.1*khdz(k), (one - k/(kmaxfire*two)) * ((log(frp))**2 - two*log(frp)) / dz(k)*rhoz(k)) 

         ENDIF
      ENDIF
   ENDDO
ENDIF





DO ic = 1,nchem

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awchem1(k,ic)- s_aw1(k)*(chem1(k,ic)*upwind+chem1(k-1,ic)*(one-upwind))
         dncont(k)= zero 
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero

      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=chem1(k,ic)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=chem1(k,ic)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 
      
      k=kts
      a(k)=   -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k) - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
      c(k)=   -dtz(k)*khdz(k+1)*rhoinv(k)           - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
      d(k)=chem1(k,ic) & 
           & - dtz(k)*vd1(ic)*chem1(k,ic) &
           & - dtz(k)*rhoinv(k)*s_awchem1(k+1,ic)

      DO k=kts+1,kte-1
         a(k)=   -dtz(k)*khdz(k)*rhoinv(k)     + p5*dtz(k)*rhoinv(k)*s_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k) + &
            &    p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))
         c(k)=   -dtz(k)*khdz(k+1)*rhoinv(k) - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
         d(k)=chem1(k,ic) + dtz(k)*rhoinv(k)*(s_awchem1(k,ic)-s_awchem1(k+1,ic))
      ENDDO
   endif

   
   a(kte)=zero
   b(kte)=one
   c(kte)=zero
   d(kte)=chem1(kte,ic)

   CALL tridiag3(kte,a,b,c,d,x)

   DO k=kts,kte
      chem1(k,ic)=x(k) + settle1(k,ic)
   ENDDO

ENDDO

END SUBROUTINE mynn_mix_chem



SUBROUTINE mynn_mix_scalars(            &
     kts,kte,i,                         &
     delt,dz,                           &
     nscalars, scalars,                 &
     rho,                               &
     flt, tcd, qcd,                     &
     dfh,                               &
     s_aw1, s_awscalars1,               &
     bl_mynn_edmf, upwind               )


  use module_bl_mynnedmf_common, only: zero,one,p5,kind_phys
  
integer, intent(in) :: kts,kte,i,bl_mynn_edmf
real(kind_phys), dimension(kts:kte), intent(in) :: dfh,dz,tcd,qcd
real(kind_phys), dimension(kts:kte), intent(in) :: rho
real(kind_phys), intent(in)    :: flt
real(kind_phys), intent(in)    :: delt,upwind
integer, intent(in) :: nscalars
real(kind_phys), dimension( kts:kte+1), intent(in) :: s_aw1
real(kind_phys), dimension( kts:kte, nscalars ), intent(inout) :: scalars
real(kind_phys), dimension( kts:kte+1,nscalars), intent(in) :: s_awscalars1

real(kind_phys), dimension(kts:kte) :: dtz,upcont,dncont
real(kind_phys), dimension(kts:kte) :: a,b,c,d,x
real(kind_phys):: rhs,dztop
real(kind_phys):: dzk 
integer :: k,ns  

real(kind_phys), dimension(kts:kte) :: rhoinv
real(kind_phys), dimension(kts:kte+1) :: rhoz,khdz

dztop=p5*(dz(kte)+dz(kte-1))
    
DO k=kts,kte
   dtz(k)=delt/dz(k)
ENDDO



rhoz(kts)  =rho(kts)
rhoinv(kts)=one/rho(kts)
khdz(kts)  =rhoz(kts)*dfh(kts)

DO k=kts+1,kte
   rhoz(k)  =(rho(k)*dz(k-1) + rho(k-1)*dz(k))/(dz(k-1)+dz(k))
   rhoz(k)  =   MAX(rhoz(k),1E-4_kind_phys)
   rhoinv(k)=one/MAX(rho(k),1E-4_kind_phys)
   dzk      = p5*( dz(k)+dz(k-1) )
   khdz(k)  = rhoz(k)*dfh(k)
ENDDO
rhoz(kte+1)=rhoz(kte)
khdz(kte+1)=rhoz(kte+1)*dfh(kte)

if (bl_mynn_edmf == 1) then
   
   DO k=kts+1,kte-1
      khdz(k) = MAX(khdz(k),  p5*s_aw1(k))
      khdz(k) = MAX(khdz(k), -p5*(s_aw1(k)-s_aw1(k+1)))
   ENDDO
endif





DO ns = 1,nscalars

   if (bl_mynn_edmf > 1) then

      DO k=kts+1,kte-1
         upcont(k)= s_awscalars1(k,ns)- s_aw1(k)*(scalars(k,ns)*upwind+scalars(k-1,ns)*(one-upwind))
         dncont(k)=zero 
      ENDDO
      
      upcont(kte)=zero
      dncont(kte)=zero

      k=kts
      a(1)=zero
      b(1)=one + dtz(k)*khdz(k+1)*rhoinv(k)
      c(1)=    - dtz(k)*khdz(k+1)*rhoinv(k)
      d(1)=scalars(k,ns)                                       &
          &    - dtz(k)*(upcont(k+1)+dncont(k+1))

      DO k=kts+1,kte-1
         a(k)=   - dtz(k)*khdz(k)*rhoinv(k)
         b(k)=one+ dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k)
         c(k)=   - dtz(k)*khdz(k+1)*rhoinv(k)
         d(k)=scalars(k,ns)-dtz(k)*(upcont(k+1)-upcont(k)+dncont(k+1)-dncont(k))
      ENDDO

   else 

      k=kts
      a(k)=   -dtz(k)*khdz(k)*rhoinv(k)
      b(k)=one+dtz(k)*(khdz(k+1)+khdz(k))*rhoinv(k) - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
      c(k)=   -dtz(k)*khdz(k+1)*rhoinv(k)           - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
      d(k)=scalars(k,ns) - dtz(k)*rhoinv(k)*s_awscalars1(k+1,ns)

      DO k=kts+1,kte-1
         a(k)=   -dtz(k)*khdz(k)*rhoinv(k)     + p5*dtz(k)*rhoinv(k)*s_aw1(k)
         b(k)=one+dtz(k)*(khdz(k)+khdz(k+1))*rhoinv(k) + &
             &    p5*dtz(k)*rhoinv(k)*(s_aw1(k)-s_aw1(k+1))
         c(k)=   -dtz(k)*khdz(k+1)*rhoinv(k) - p5*dtz(k)*rhoinv(k)*s_aw1(k+1)
         d(k)=scalars(k,ns) + dtz(k)*rhoinv(k)*(s_awscalars1(k,ns)-s_awscalars1(k+1,ns))
      ENDDO
   endif

   
   a(kte)=zero
   b(kte)=one
   c(kte)=zero
   d(kte)=scalars(kte,ns)

   CALL tridiag3(kte,a,b,c,d,x)

   DO k=kts,kte
      scalars(k,ns)=max(1e-12_kind_phys, x(k))
   ENDDO
ENDDO

END SUBROUTINE mynn_mix_scalars
  


  SUBROUTINE retrieve_exchange_coeffs(kts,kte,dfm,dfh,dz,km1,kh1)


    use module_bl_mynnedmf_common, only: zero,p5,kind_phys
    
    integer , intent(in) :: kts,kte

    real(kind_phys), dimension(kts:kte), intent(in)  :: dz,dfm,dfh
    real(kind_phys), dimension(kts:kte), intent(out) :: km1, kh1


    integer :: k
    real(kind_phys):: dzk

    km1(kts)=zero
    kh1(kts)=zero

    do k=kts+1,kte
       dzk   = p5 *( dz(k)+dz(k-1) )
       km1(k)=dfm(k)*dzk
       kh1(k)=dfh(k)*dzk
    enddo




  END SUBROUTINE retrieve_exchange_coeffs



  SUBROUTINE tridiag(n,a,b,c,d)







    use module_bl_mynnedmf_common, only: zero,one,kind_phys
    
    integer, intent(in):: n
    real(kind_phys), dimension(n), intent(in) :: a,b
    real(kind_phys), dimension(n), intent(inout) :: c,d
    
    integer :: i
    real(kind_phys):: p
    real(kind_phys), dimension(n) :: q
    
    c(n)=zero
    q(1)=-c(1)/b(1)
    d(1)=d(1)/b(1)
    
    DO i=2,n
       p=one/(b(i)+a(i)*q(i-1))
       q(i)=-c(i)*p
       d(i)=(d(i)-a(i)*d(i-1))*p
    ENDDO
    
    DO i=n-1,1,-1
       d(i)=d(i)+q(i)*d(i+1)
    ENDDO

  END SUBROUTINE tridiag



      subroutine tridiag2(kte,a,b,c,d,x)







        use module_bl_mynnedmf_common, only: zero,one,kind_phys
        
        integer,intent(in) :: kte
        real(kind_phys), dimension(kte), intent(in) :: a,b,c,d
        real(kind_phys), dimension(kte), intent(out):: x
        real(kind_phys), dimension(kte)  :: cp1,dp1
        real(kind_phys):: m
        integer :: k

        
        cp1(1) = c(1)/b(1)
        dp1(1) = d(1)/b(1)
        
        do k = 2,kte
           m = b(k)-cp1(k-1)*a(k)
           cp1(k) = c(k)/m
           dp1(k) = (d(k)-dp1(k-1)*a(k))/m
        enddo
        
        x(kte) = dp1(kte)
        
        do k = kte-1, 1, -1
           x(k) = dp1(k)-cp1(k)*x(k+1)
        end do

    end subroutine tridiag2


       subroutine tridiag3(kte,a,b,c,d,x)












        use module_bl_mynnedmf_common, only: zero,one,kind_phys
         
        integer,intent(in)   :: kte
        integer, parameter   :: kts=1
        real(kind_phys), dimension(kte) :: a,b,c,d
        real(kind_phys), dimension(kte), intent(out) :: x
        integer :: k

        do k=kte-1,kts,-1
           d(k)=d(k)-c(k)*d(k+1)/b(k+1)
           b(k)=b(k)-c(k)*a(k+1)/b(k+1)
        enddo

        do k=kts+1,kte
           d(k)=d(k)-a(k)*d(k-1)/b(k-1)
        enddo

        do k=kts,kte
           x(k)=d(k)/b(k)
        enddo

        return
        end subroutine tridiag3




























SUBROUTINE GET_PBLH(KTS,KTE,pblh,thv1,qke1,ust,zw1,dz1,landsea,kpbl)


  use module_bl_mynnedmf_common, only: zero,one,ten,hundred,p5,kind_phys
  
integer,intent(in) :: KTS,KTE


real(kind_phys), intent(out) :: pblh
real(kind_phys), intent(in) :: landsea,ust
real(kind_phys), dimension(kts:kte), intent(in) :: thv1, qke1, dz1
real(kind_phys), dimension(kts:kte+1), intent(in) :: zw1

real(kind_phys):: pblh_stable,qtke,qtkem1,wt,maxqke,TKEeps,minthv,cpblh
real(kind_phys):: delt_thv   
real(kind_phys), parameter :: sbl_lim  = 200. 
real(kind_phys), parameter :: sbl_damp = 170. 
integer :: i,j,k,kthv,ktke,kpbl
integer, parameter :: stable_method=1 


kpbl = 2


k = kts+1
kthv = 1
minthv = 9.E9_kind_phys
do while (zw1(k) .le. 200.)
   if (minthv > thv1(k)) then
      minthv = thv1(k)
      kthv = k
   endif
   k = k+1
enddo


pblh=zero
k = kthv+1
if ((landsea-1.5) .ge. zero) then
   
   delt_thv = 0.75_kind_phys
else
   
   delt_thv = 1.25_kind_phys
endif

pblh=zero
k = kthv+1
do k=kts+1,kte-1
   if (thv1(k) .ge. (minthv + delt_thv)) then
      pblh = zw1(k) - dz1(k-1)* &
      & min((thv1(k)-(minthv + delt_thv))/ &
      & max(thv1(k)-thv1(k-1),1E-6_kind_phys),one)
   endif
   if (k .EQ. kte-1) pblh = zw1(kts+1) 
   if (pblh .gt. zero) exit
enddo


if (stable_method == 0) then
   
   
   
   
   ktke   = 1
   maxqke = max(qke1(kts),zero)
   
   
   tkeeps = maxqke/40._kind_phys
   tkeeps = max(tkeeps,0.01_kind_phys) 
   pblh_stable=zero

   k = ktke+1
   do k=kts+1,kte-1
      qtke  =max(p5*qke1(k)  , zero)
      qtkem1=max(p5*qke1(k-1), zero)
      if (qtke .le. tkeeps) then
         pblh_stable = zw1(k) - dz1(k-1)* &
         & min((tkeeps-qtke)/max(qtkem1-qtke, 1e-6_kind_phys), one)
         
         pblh_stable = max(pblh_stable,zw1(kts+1))
         
      endif
      
      if (k .eq. kte-1) pblh_stable = zw1(kts+1) 
      if (pblh_stable .ne. zero) exit
   enddo
   
   
   
   
   
   pblh_stable = min(pblh_stable,pblh+350._kind_phys)
   pblh_stable = max(pblh_stable,max(pblh-350._kind_phys, ten))
   
   if (maxqke <= tkeeps) pblh_stable = pblh
else
   
   wt          = p5*TANH((pblh - hundred)/hundred) + p5
   cpblh       = 400._kind_phys*(one-wt) + 700._kind_phys*wt
   pblh_stable = max(ust*cpblh, ten)
   
   
   pblh_stable = max(pblh_stable, pblh-sbl_lim)
   pblh_stable = min(pblh_stable, pblh+sbl_lim)
endif


wt=p5*TANH((pblh - sbl_lim)/sbl_damp) + p5
pblh=pblh_stable*(one-wt) + pblh*wt


do k=kts+1,kte-1
   if ( zw1(k) >= pblh) then
      kpbl = k-1
      exit
   endif
enddo


END SUBROUTINE GET_PBLH

  



















  SUBROUTINE DMP_mf(i,j,                           &
                 & kts,kte,dt,zw1,dz1,pres1,rho1,  &
                 & momentum_opt,                   &
                 & tke_opt,                        &
                 & scalar_opt,                     &
                 & aerosol_opt, numcon_opt,        &
                 & bl_mynn_closure,                &
                 & u1,v1,w1,th1,thl1,thv1,tk1,     &
                 & qt1,qv1,qc1,qke1,qsq1,          &
                 & qnc1,qni1,qnwfa1,qnifa1,qnbca1, &
                 & ex1,vt1,vq1,sgm1,               &
                 & ust,flt,fltv,flq,flqv,          &
                 & pblh,kpbl,dx,landsea,ts,        &
            
                 & edmf_a1,edmf_w1,                &
                 & edmf_qt1,edmf_thl1,             &
                 & edmf_ent1,edmf_qc1,             &
                 & edmf_qv1,edmf_u1,edmf_v1,       &
            
                 & s_aw1,s_awthl1,s_awqt1,         &
                 & s_awqv1,s_awqc1,                &
                 & s_awu1,s_awv1,s_awqke1,s_awqsq1,&
                 & s_awqnc1,s_awqni1,              &
                 & s_awqnwfa1,s_awqnifa1,          &
                 & s_awqnbca1,                     &
                 & sub_thl1,sub_sqv1,              &
                 & sub_u1,sub_v1,                  &
                 & det_thl1,det_sqv1,det_sqc1,     &
                 & det_u1,det_v1,                  &
            
                 & nchem,chem1,s_awchem1,          &
                 & mix_chem,                       &
            
                 & nscalars,scalars,s_awscalars1,  &                 
            
                 & qc_bl1,cldfra_bl1,              &
                 & qc_bl1_old,cldfra_bl1_old,      &
            
                 & F_QC,F_QI,                      &
                 & F_QNC,F_QNI,                    &
                 & F_QNWFA,F_QNIFA,F_QNBCA,        &
                 & Psig_shcu,                      &
            
                 & maxwidth,ktop,maxmf,ztop,       &
                 & excess_h,excess_q,              &
            
                 & spp_pbl,pattern_spp_pbl1,       &
                 & tkeprod_up,el1                  )

    use module_bl_mynnedmf_common, only: gtr,p608,grav,xlvcp,ep_2,  &
         ep_3,r_v,cp,cpv,tv0,b1,                                    &
         zero,one,two,three,four,five,thirty,hundred,               &
         p2,p3,p333,p4,p5,kind_phys



 integer, intent(in) ::                                            &
      kts,                kte,                  kpbl,              &
      momentum_opt,       tke_opt,              scalar_opt,        &
      aerosol_opt,        numcon_opt,                              &
      spp_pbl,            i,                    j
 real(kind_phys), intent(in):: bl_mynn_closure
 real(kind_phys), dimension(kts:kte), intent(in)  ::               &
      pattern_spp_pbl1

 real(kind_phys), dimension(kts:kte), intent(in)  ::               &
      &u1,v1,w1,th1,thl1,tk1,qt1,qv1,qc1,                          &
      &ex1,dz1,thv1,pres1,rho1,qke1,qsq1,qnc1,qni1,                &
      &qnwfa1,qnifa1,qnbca1,el1
 real(kind_phys),dimension(kts:kte+1), intent(in) ::               &
      &zw1
 real(kind_phys), intent(in)                      ::               &
      &flt,fltv,flq,flqv,Psig_shcu,                                &
      &landsea,ts,dx,dt,ust,pblh
 logical, optional :: F_QC,F_QI,F_QNC,F_QNI,F_QNWFA,F_QNIFA,F_QNBCA

 
 real(kind_phys),dimension(kts:kte), intent(inout) ::              &
      &edmf_a1,edmf_w1,edmf_qt1,edmf_thl1,edmf_ent1,edmf_qc1,      &
      &edmf_u1,edmf_v1,edmf_qv1
 
  real(kind_phys),dimension(kts:kte) :: edmf_th1
 
 integer,         intent(inout) :: ktop
 real(kind_phys), intent(inout) :: maxmf,ztop,maxwidth,            &
      &excess_h,excess_q
 
 real(kind_phys),dimension(kts:kte+1), intent(inout) ::            &
      &s_aw1,s_awthl1,s_awqt1,s_awqv1,s_awqc1,s_awqnc1,s_awqni1,   &
      &s_awqnwfa1,s_awqnifa1,s_awqnbca1,s_awu1,s_awv1,             &
      &s_awqke1,s_awqsq1
 real(kind_phys),dimension(kts:kte+1) :: s_aw2

 real(kind_phys),dimension(kts:kte), intent(inout) ::              &
      &qc_bl1,cldfra_bl1,qc_bl1_old,cldfra_bl1_old,tkeprod_up
 integer, parameter :: nup=8, debug_mf=0
 real(kind_phys)    :: nup2

 
 
 
 real(kind_phys),dimension(kts:kte+1,1:NUP) ::                     &
      &UPW,UPTHL,UPQT,UPQC,UPQV,                                   &
      &UPA,UPU,UPV,UPTHV,UPQKE,UPQSQ,UPQNC,                        &
      &UPQNI,UPQNWFA,UPQNIFA,UPQNBCA
 
 real(kind_phys),dimension(kts:kte,1:NUP) :: ENT
 
 integer :: k,ip,k50
 real(kind_phys):: fltv2,wstar,qstar,thstar,sigmaW,sigmaQT,        &
      &sigmaTH,z0,pwmin,pwmax,wmin,wmax,wlv,Psig_w,maxw,maxqc,wpbl
 real(kind_phys):: B,QTn,THLn,THVn,QCn,Un,Vn,QKEn,QSQn,QNCn,QNIn,  &
      &  QNWFAn,QNIFAn,QNBCAn,upak,                                &
      &  Wn2,Wn,EntEXP,EntEXM,EntW,BCOEFF,THVkm1,THVk,Pk,rho_int

 
 real(kind_phys), parameter ::                                     &
      &Wa=2./3.,      Wb=0.002,      Wc=1.5 

 
 real(kind_phys), parameter :: Atot = 0.10 
 real(kind_phys), parameter :: lmax = 1000.
 real(kind_phys), parameter :: lmin = 300. 
 real(kind_phys), parameter :: dlmin = 0.  
 real(kind_phys)            :: minwidth    
 real(kind_phys)            :: dl          
 real(kind_phys), parameter :: dcut = 1.2  
 real(kind_phys)::  d     
      
      
 real(kind_phys):: cn,c,l,n,an2,hux,wspd_pbl,cloud_base,           &
      maxwidth_dx,maxwidth_pbl,maxwidth_cld,maxwidth_flx

 
 integer, intent(in) :: nchem
 real(kind_phys),dimension(kts:kte,   nchem)       :: chem1
 real(kind_phys),dimension(kts:kte+1, nchem)       :: s_awchem1
 real(kind_phys),dimension(nchem)                  :: chemn
 real(kind_phys),dimension(kts:kte+1,1:NUP, nchem) :: UPCHEM
 integer :: ic,cb_check
 real(kind_phys),dimension(kts:kte,   nchem)       :: edmf_chem
 logical, intent(in) :: mix_chem
 
 integer, intent(in) :: nscalars
 real(kind_phys),dimension(kts:kte,   nscalars)    :: scalars
 real(kind_phys),dimension(kts:kte+1, nscalars)    :: s_awscalars1
 real(kind_phys),dimension(nscalars)               :: scalarsn
 real(kind_phys),dimension(kts:kte+1,1:NUP,nscalars) :: upscalars
 
 integer,dimension(1:NUP) :: ktop_plume
 logical :: superadiabatic

 
 real(kind_phys),dimension(kts:kte), intent(inout) :: vt1, vq1, sgm1
 real(kind_phys):: sigq,xl,rsl,cpm,a,qmq,Aup,Q1,diffqt,qsat_tk,     &
         Fng,qww,alpha,beta,bb,f,pt,t,q2p,b9,satvp,rhgrid,entfac,   &
         cf_strat,qc_strat,cf_mf,qc_mf,qc_mf_min,pct_mf,wt2,        &
         dqwdz,tauc,mf_at_cb,qcfac,sigqfac,cffac
 real(kind_phys), parameter :: cf_thresh = 0.5 

 
 real(kind_phys),dimension(kts:kte) :: exneri,dzi,rhoi,qti
 
 real(kind_phys):: THp, QTp, QCp, QCs, esat, qsl
 real(kind_phys):: csigma,acfac,ac_wsp

 
 integer :: overshoot
 real(kind_phys):: bvf, Frz, dzp

 
 
 real(kind_phys):: adjustment, flx1, flt2
 real(kind_phys), parameter :: fluxportion=0.75 
                                     
                                     
 
 
 logical,parameter:: env_subs   = .false.
 real(kind_phys),dimension(kts:kte) :: sub_thl1,sub_sqv1,         &
         sub_u1,sub_v1,det_thl1,det_sqv1,det_sqc1,det_u1,det_v1,  &  
         envm_a,envm_w,envm_thl,envm_sqv,envm_sqc,                &
         envm_u,envm_v                  
 real(kind_phys),dimension(kts:kte+1) ::  envi_a,envi_w 
 real(kind_phys):: temp,sublim,qc_ent,qv_ent,qt_ent,thl_ent,detrate, &
         detrateUV,oow,exc_fac,aratio,detturb,qc_grid,qc_sgs,        &
         qc_plume,exc_heat,exc_moist,tk_int,tvs,dthvdz,zagl
 real(kind_phys), parameter :: Cdet   = 1./45.
 real(kind_phys), parameter :: dzpmax = 300. 
 
 
 
 
 real(kind_phys), parameter :: Csub=0.25

 
 real(kind_phys), parameter :: pgfac = 0.00  
 real(kind_phys):: Uk,Ukm1,Vk,Vkm1,dxsa

 
 real(kind_phys):: qkebl

 
 
      

 ktop      =0    
 ztop      =zero
 maxmf     =zero
 maxwidth  =zero
 excess_h  =zero
 excess_q  =zero
 
 upw       =zero
 upthl     =zero
 upthv     =zero
 upqt      =zero
 upa       =zero
 upu       =zero
 upv       =zero
 upqc      =zero
 upqv      =zero
 upqke     =zero
 upqsq     =zero
 upqnc     =zero
 upqni     =zero
 upqnwfa   =zero
 upqnifa   =zero
 upqnbca   =zero
 if ( mix_chem ) then
    upchem(kts:kte+1,1:nup,1:nchem)=zero
 endif
 if ( scalar_opt > 0 ) then
    upscalars(kts:kte+1,1:nup,1:nscalars)=zero
 endif
 ent       =0.001_kind_phys
 
 edmf_a1   =zero
 edmf_w1   =zero
 edmf_qt1  =zero
 edmf_thl1 =zero
 edmf_ent1 =zero
 edmf_qc1  =zero
 edmf_qv1  =zero
 edmf_u1   =zero
 edmf_v1   =zero
 if ( mix_chem ) then
    edmf_chem(kts:kte,1:nchem) = zero
 endif
 
 s_aw1     =zero
 s_awthl1  =zero
 s_awqt1   =zero
 s_awqv1   =zero
 s_awqc1   =zero
 s_awu1    =zero
 s_awv1    =zero
 s_awqke1  =zero
 s_awqsq1  =zero
 s_awqnc1  =zero
 s_awqni1  =zero
 s_awqnwfa1=zero
 s_awqnifa1=zero
 s_awqnbca1=zero
 if ( mix_chem ) then
    s_awchem1(kts:kte+1,1:nchem) = zero
 endif
 if ( scalar_opt > 0 ) then
    s_awscalars1(kts:kte+1,1:nscalars) = zero
 endif
 
 sub_thl1 = zero
 sub_sqv1 = zero
 sub_u1   = zero
 sub_v1   = zero
 det_thl1 = zero
 det_sqv1 = zero
 det_sqc1 = zero
 det_u1   = zero
 det_v1   = zero
 nup2     = nup 
 tkeprod_up = zero

 if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
    print*,"===incoming forcing in mf component:"
    print*,"  flt=",flt," fltv=",fltv
    print*,"  Psig_shcu=",Psig_shcu," wspd=",sqrt(max(u1(kts)**2 + v1(kts)**2, 0.01_kind_phys))
 endif
      
 
 
 maxw        = zero
 cloud_base  = 9000.0_kind_phys
 qkebl       = zero
 do k=1,kte-1
    zagl = zw1(k) + p5*dz1(k)
    if (zagl > (pblh + 500.)) exit

    wpbl = w1(k)
    if (w1(k) < zero)wpbl = two*w1(k)
    maxw = max(maxw,abs(wpbl))

    
    if (zagl <= 50.)k50=k

    
    if (k <= kpbl) then
       qkebl = qkebl + qke1(k)
    endif

    
    qc_sgs = max(qc1(k), qc_bl1(k))
    if ((qc_sgs > 1E-5) .and. (cldfra_bl1(k) .ge. p5) .and. cloud_base == 9000.0) then
       cloud_base = zw1(k) 
    endif
 enddo
 qkebl = qkebl / real(kpbl, kind=kind_phys)
 
 
 maxw = max(zero, maxw - one)
 Psig_w = max(zero, one - maxw)
 Psig_w = min(Psig_w, Psig_shcu)

 
 fltv2 = fltv
 if (Psig_w < 1e-2_kind_phys .and. fltv > zero) fltv2 = -1.*fltv

 if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
    print*,"===criteria for small w in pbl:"
    print*,"maxw=",maxw," Psig_w=",Psig_w," k50=",k50
 endif
      
 
 
 superadiabatic = .false.
 if ((landsea-1.5).ge.zero) then
    hux = -0.0024  
 else
    hux = -0.0030  
 endif
 tvs    = ts*(one+p608*qv1(kts))
 if ((thv1(kts)-tvs)/(p5*dz1(kts)) < hux) then
    
    superadiabatic = .true.
    do k=2,max(2,k50)
       hux = -0.0018  
       dthvdz = (thv1(k)-thv1(k-1))/(p5*(dz1(k)+dz1(k-1)))
       if (dthvdz < hux) then
          superadiabatic = .true.
       else
          superadiabatic = .false.
          exit
       endif
    enddo
 endif

 if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
    print*," superadiatic=",superadiabatic," hux=",hux
 endif

 
 
 
 
 
 
 
 
 
 maxwidth_dx = min(dx*dcut, lmax)
 
 maxwidth_pbl = min(1.1_kind_phys*pblh, lmax) 
 
 if ((landsea-1.5) .lt. zero) then  
    maxwidth_cld = min(lmax, max(0.5_kind_phys*cloud_base, 300._kind_phys))
 else                               
    maxwidth_cld = min(lmax, max(0.8_kind_phys*cloud_base, 300._kind_phys))
 endif
 
 wspd_pbl=sqrt(max(u1(kts)**2 + v1(kts)**2, 0.01_kind_phys))
 
 
 if ((landsea-1.5) .lt. zero) then  
    maxwidth_flx = MAX(MIN(1000.*(0.6*tanh((fltv - 0.040)/0.04) + p5),1000._kind_phys), zero)
 else                             
    maxwidth_flx = MAX(MIN(1000.*(0.6*tanh((fltv - 0.007)/0.02) + p5),1000._kind_phys), zero)
    
 endif
 maxwidth = MIN(maxwidth_dx, maxwidth_pbl)
 maxwidth = MIN(maxwidth,    maxwidth_cld)
 maxwidth = MIN(maxwidth,    maxwidth_flx)      
 minwidth = lmin

 if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then  
    print*,"===limiting factors on plume width:"
    print*,"maxwidth_dx= ",maxwidth_dx," dx=",dx
    print*,"maxwidth_pbl=",maxwidth_pbl," pblh=",pblh
    print*,"maxwidth_cld=",maxwidth_cld," cldbase=",cloud_base
    print*,"maxwidth_flx=",maxwidth_flx," fltv=",fltv
    print*,"===final check for activation criteria:"
    print*,"fltv2=",fltv2," superadiatic=",superadiabatic
    print*,"maxwidth=",maxwidth," minwidth=",minwidth
 endif

 if (maxwidth .le. minwidth) then 
    nup2 = 0
    maxwidth = zero
 endif

 
 if ( flt > zero .AND. fltv2 > 0.008 .AND. (maxwidth > minwidth) .AND. superadiabatic) then

    
    cn = zero
    d  =-1.9_kind_phys  
    dl = (maxwidth - minwidth)/real(nup-1,kind=kind_phys)
    do ip=1,NUP
       
       l = minwidth + dl*real(ip-1,kind=kind_phys)
       cn = cn + l**d * (l*l)/(dx*dx) * dl  
    enddo
    C = Atot/cn   

    
    if ((landsea-1.5) .lt. zero) then  
       acfac = p5*tanh((fltv2 - 0.02_kind_phys)/0.05_kind_phys) + p5
    else
       acfac = p5*tanh((fltv2 - 0.012_kind_phys)/0.03_kind_phys) + p5
    endif
      
    
    
    ac_wsp = one - min((max(wspd_pbl - 13.0_kind_phys, zero))/12._kind_phys, one)
    acfac  = min(acfac, ac_wsp)

    
    An2 = zero
    do ip=1,nup
       
       l  = minwidth + dl*real(ip-1,kind=kind_phys)
       N  = C*l**d                           
       UPA(1,ip) = N*l*l/(dx*dx) * dl        

       UPA(1,ip) = UPA(1,ip)*acfac
       An2 = An2 + UPA(1,ip)                 
       
       
    end do

    if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
       print*,"===limiting factors on fractional area coverage:"
       print*,"total araa frac=",An2," atot=",atot
       print*,"upa=",UPA(1,1:nup)
       print*,"acfac fluxes=",acfac," fltv=",fltv
       print*,"acfac wspd=",ac_wsp," wspd_pbl=",wspd_pbl
    endif
    
    
    z0    =50._kind_phys
    pwmin =0.1_kind_phys    
    pwmax =0.4_kind_phys    

    wstar =max(1.E-2_kind_phys,(gtr*fltv2*pblh)**(p333))
    qstar =max(flq,1e-5_kind_phys)/wstar
    thstar=flt/wstar

    csigma=1.34_kind_phys

    if (env_subs) then
       exc_fac = zero
    else
       if ((landsea-1.5).GE.zero) then
          
          
          
          
          
          
          exc_fac = 9.28_kind_phys
       else
          
          exc_fac = 0.58_kind_phys
       endif
    endif
    
    exc_fac = exc_fac * ac_wsp

    
    sigmaW =csigma*wstar*(z0/pblh)**(p333)*(one - 0.8_kind_phys*z0/pblh)
    sigmaQT=csigma*qstar*(z0/pblh)**(p333)
    sigmaTH=csigma*thstar*(z0/pblh)**(p333)

    
    
    wmin=MIN(sigmaW*pwmin,0.1_kind_phys)
    wmax=MIN(sigmaW*pwmax,0.5_kind_phys)

    if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
       print*,"===excess components:"
       print*,"wstar=",wstar,"qstar=",qstar,"thstar=",thstar
       print*,"csigma=",csigma,"exc_fac=",exc_fac,"wmin=",wmin
       print*,"sigmaw=",sigmaw,"sigmaqt=",sigmaqt,"sigmath=",sigmath
    endif

    
    do ip=1,NUP
       wlv=wmin+(wmax-wmin)/real(NUP2,kind=kind_phys)*real(ip-1,kind=kind_phys)
       UPA(2,ip)    =UPA(1,ip)
       
       UPW(2,ip)    =wmin + real(ip,kind=kind_phys)/real(NUP)*(wmax-wmin)
       UPU(2,ip)    =(u1(kts)    *dz1(kts+1)+u1(kts+1)    *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPV(2,ip)    =(v1(kts)    *dz1(kts+1)+v1(kts+1)    *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQC(2,ip)   =zero
       

       exc_heat     =exc_fac*UPW(2,ip)*sigmaTH/sigmaW
       UPTHV(2,ip)  =(thv1(kts)  *dz1(kts+1)+thv1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1)) &
           &        + exc_heat
       UPTHL(2,ip)  =(thl1(kts)  *dz1(kts+1)+thl1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1)) &
           &        + exc_heat
       
       exc_moist    =exc_fac*UPW(2,ip)*sigmaQT/sigmaW
       UPQT(2,ip)   =(qt1(kts)   *dz1(kts+1)+qt1(kts+1)   *dz1(kts))/(dz1(kts)+dz1(kts+1))&
            &       + exc_moist
       UPQKE(2,ip)  =(qke1(kts)  *dz1(kts+1)+qke1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQSQ(2,ip)  =(qsq1(kts)  *dz1(kts+1)+qsq1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQNC(2,ip)  =(qnc1(kts)  *dz1(kts+1)+qnc1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQNI(2,ip)  =(qni1(kts)  *dz1(kts+1)+qni1(kts+1)  *dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQNWFA(2,ip)=(qnwfa1(kts)*dz1(kts+1)+qnwfa1(kts+1)*dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQNIFA(2,ip)=(qnifa1(kts)*dz1(kts+1)+qnifa1(kts+1)*dz1(kts))/(dz1(kts)+dz1(kts+1))
       UPQNBCA(2,ip)=(qnbca1(kts)*dz1(kts+1)+qnbca1(kts+1)*dz1(kts))/(dz1(kts)+dz1(kts+1))
    enddo

    if ( mix_chem ) then
      do ip=1,NUP
        do ic = 1,nchem
          UPCHEM(2,ip,ic)=(chem1(kts,ic)*dz1(kts+1)+chem1(kts+1,ic)*dz1(kts))/(dz1(kts)+dz1(kts+1))
        enddo
      enddo
    endif
    if ( scalar_opt > 0 ) then
      do ip=1,NUP
        do ic = 1,nscalars
          upscalars(2,ip,ic)=(scalars(kts,ic)*dz1(kts+1)+scalars(kts+1,ic)*dz1(kts))/(dz1(kts)+dz1(kts+1))
        enddo
      enddo
   endif
   
    
    envm_thl(kts:kte)=thl1(kts:kte)
    envm_sqv(kts:kte)=qv1(kts:kte)
    envm_sqc(kts:kte)=qc1(kts:kte)
    envm_u(kts:kte)  =u1(kts:kte)
    envm_v(kts:kte)  =v1(kts:kte)
    
    rhoi(kts)   = rho1(kts)
    dzi(kts)    = p5*dz1(kts)
    exneri(kts) = ex1(kts)
    do k=kts+1,kte
       rhoi(k)  = (rho1(k-1)*dz1(k)+rho1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))
       dzi(k)   = (dz1(k-1)*dz1(k)+dz1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))
       exneri(k)= (ex1(k-1)*dz1(k)+ex1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))
    enddo
   
    
    dxsa = one - MIN(MAX((12000.0_kind_phys-dx)/(12000.0_kind_phys-3000.0_kind_phys), zero), one)

    ktop_plume = 0
    
    do ip=1,NUP
       QCn = zero
       overshoot = 0 
       l  = minwidth + dl*real(ip-1,kind=kind_phys)    
       do k=kts+1,kte-2
          zagl = zw1(k) + p5*dz1(k)
          
          
          wmin   = p3 + l*0.0005_kind_phys
          
          
          
          
          
          
          
          entfac = 0.21_kind_phys * min(1.57_kind_phys, max(1.30_kind_phys, sqrt(qkebl)))
          
          
          wt2    = min(one, max(zero, zagl - pblh)/500._kind_phys) 
          entfac = entfac*(one-wt2) + wt2*0.33_kind_phys
          ENT(k,ip) = entfac/(MIN(MAX(UPW(K-1,ip),wmin),one)*l)
          
          
          
          
          

          
          ENT(k,ip) = max(ENT(k,ip),0.0003_kind_phys)
          

          
          IF(zw1(k) >= MIN(pblh+1500., 4000.))THEN
            ENT(k,ip)=ENT(k,ip) + (zw1(k)-MIN(pblh+1500.,4000.))*5.0E-6
          ENDIF

          
          ENT(k,ip) = ENT(k,ip) * (one - pattern_spp_pbl1(k))

          ENT(k,ip) = min(ENT(k,ip), 0.9_kind_phys/dz1(k))

          
          Uk     =(u1(k)*dz1(k+1)+u1(k+1)*dz1(k))/(dz1(k+1)+dz1(k))
          Ukm1   =(u1(k-1)*dz1(k)+u1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))
          Vk     =(v1(k)*dz1(k+1)+v1(k+1)*dz1(k))/(dz1(k+1)+dz1(k))
          Vkm1   =(v1(k-1)*dz1(k)+v1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))

          
          EntExp =ENT(k,ip)*dz1(k)
          EntExm =EntExp*p333    
          QTn    =UPQT(k,IP)   *(one-EntExp) + qt1(k)*EntExp
          THLn   =UPTHL(k,IP)  *(one-EntExp) + thl1(k)*EntExp
          Un     =UPU(k,IP)    *(one-EntExm) + u1(k)*EntExm + dxsa*pgfac*(Uk - Ukm1)
          Vn     =UPV(k,IP)    *(one-EntExm) + v1(k)*EntExm + dxsa*pgfac*(Vk - Vkm1)
          QKEn   =UPQKE(k,IP)  *(one-EntExp) + qke1(k)*EntExp
          QSQn   =UPQSQ(k,IP)  *(one-EntExp) + qsq1(k)*EntExp
          QNCn   =UPQNC(k,IP)  *(one-EntExp) + qnc1(k)*EntExp
          QNIn   =UPQNI(k,IP)  *(one-EntExp) + qni1(k)*EntExp
          QNWFAn =UPQNWFA(k,IP)*(one-EntExp) + qnwfa1(k)*EntExp
          QNIFAn =UPQNIFA(k,IP)*(one-EntExp) + qnifa1(k)*EntExp
          QNBCAn =UPQNBCA(k,IP)*(one-EntExp) + qnbca1(k)*EntExp

          
          
          qc_ent =QCn
          qt_ent =QTn
          thl_ent=THLn

          
          
          
          
          
          
          

          if ( mix_chem ) then
            do ic = 1,nchem
              
              
              
              chemn(ic)=UPCHEM(k,ip,ic)*(one-EntExp) + chem1(k,ic)*EntExp
            enddo
          endif
          if ( scalar_opt > 0 ) then
            do ic = 1,nscalars
              scalarsn(ic)=upscalars(k,ip,ic)*(one-EntExp) + scalars(k,ic)*EntExp
            enddo
          endif
          
          
          Pk    =(pres1(k)*dz1(k+1)+pres1(k+1)*dz1(k))/(dz1(k)+dz1(k+1))
          
          call condensation_edmf(QTn,THLn,Pk,zw1(k+1),THVn,QCn)

          
          THVk  =(thv1(k)*dz1(k+1)+thv1(k+1)*dz1(k))/(dz1(k+1)+dz1(k))
          THVkm1=(thv1(k-1)*dz1(k)+thv1(k)*dz1(k-1))/(dz1(k-1)+dz1(k))


          B=grav*(THVn/THVk - one)
          IF(B>0.)THEN
            BCOEFF = 0.15        
          ELSE
            BCOEFF = 0.2 
          ENDIF

          
          
          
          
          
          
          
          
          IF (UPW(K-1,ip) < 0.2 ) THEN
             Wn = UPW(k,ip) + (-2. * ENT(k,ip) * UPW(k,ip) + BCOEFF*B / MAX(UPW(k,ip),0.2)) * MIN(dz1(k), 250.)
          ELSE
             Wn = UPW(k,ip) + (-2. * ENT(k,ip) * UPW(k,ip) + BCOEFF*B / UPW(k,ip)) * MIN(dz1(k), 250.)
          ENDIF
          
          
          IF(Wn > UPW(k,ip) + MIN(1.25*dz1(k)/200., two) ) THEN
             Wn = UPW(k,ip) + MIN(1.25*dz1(k)/200., two)
          ENDIF
          
          IF(Wn < UPW(k,ip) - MIN(1.25*dz1(k)/200., two) ) THEN
             Wn = UPW(k,ip) - MIN(1.25*dz1(k)/200., two)
          ENDIF
          Wn = MIN(MAX(Wn, zero), 3.0_kind_phys)

          
          
          if (k==kts+1 .and. wn < 1e-8) then
             nup2=0
             exit
          endif

          if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
            if (wn .ge. 3.0) then
              
              print *," **** SUSPICIOUSLY LARGE W:"
              print *,' QCn:',QCn,' ENT=',ENT(k,ip),' Nup2=',Nup2
              print *,'pblh:',pblh,' Wn:',Wn,' UPW(k)=',UPW(k,ip)
              print *,'K=',k,' B=',B,' dz=',dz1(k)
            ENDIF
          ENDIF

          
          IF (Wn <= zero .AND. overshoot == 0) THEN
             overshoot = 1
             IF ( THVk-THVkm1 .GT. zero ) THEN
                bvf = SQRT( gtr*(THVk-THVkm1)/dz1(k) )
                
                Frz = UPW(k,ip)/(bvf*dz1(k))
                
                dzp = dz1(k)*MAX(MIN(Frz,one),zero) 
             ENDIF
          ELSE
             dzp = dz1(k)
          ENDIF

          
          
          
          

          
          
          
          aratio   = MIN(UPA(k,IP)/(one-UPA(k,IP)), p5) 
          detturb  = 0.00008
          oow      = -0.060/MAX(one,(0.5*(Wn+UPW(k,IP))))   
          detrate  = MIN(MAX(oow*(Wn-UPW(k,IP))/dz1(k), detturb), .0002) 
          detrateUV= MIN(MAX(oow*(Wn-UPW(k,IP))/dz1(k), detturb), .0001) 
          envm_thl(k)=envm_thl(k) + (p5*(thl_ent + UPTHL(k,IP)) - thl1(k))*detrate*aratio*MIN(dzp,dzpmax)
          qv_ent = p5*(MAX(qt_ent-qc_ent,zero) + MAX(UPQT(k,IP)-UPQC(k,IP),zero))
          envm_sqv(k)=envm_sqv(k) + (qv_ent-qv1(K))*detrate*aratio*MIN(dzp,dzpmax)
          IF (UPQC(k,IP) > 1E-8) THEN
             IF (qc1(k) > 1E-6) THEN
                qc_grid = qc1(k)
             ELSE
                qc_grid = qc_bl1(K)
             ENDIF
             envm_sqc(k)=envm_sqc(k) + MAX(UPA(k,IP)*p5*(QCn + UPQC(k,IP)) - qc_grid, zero)*detrate*aratio*MIN(dzp,dzpmax)
          ENDIF
          envm_u(k)  =envm_u(k)   + (p5*(Un + UPU(k,IP)) - u1(K))*detrateUV*aratio*MIN(dzp,dzpmax)
          envm_v(k)  =envm_v(k)   + (p5*(Vn + UPV(k,IP)) - v1(K))*detrateUV*aratio*MIN(dzp,dzpmax)

          IF (Wn > 0.) THEN
             
             UPW(k+1,IP)=Wn  
             UPTHV(k+1,IP)=THVn
             UPTHL(k+1,IP)=THLn
             UPQT(k+1,IP)=QTn
             UPQC(k+1,IP)=QCn
             UPU(k+1,IP)=Un
             UPV(k+1,IP)=Vn
             UPQKE(k+1,IP)=QKEn
             UPQSQ(k+1,IP)=QSQn
             UPQNC(k+1,IP)=QNCn
             UPQNI(k+1,IP)=QNIn
             UPQNWFA(k+1,IP)=QNWFAn
             UPQNIFA(K+1,IP)=QNIFAn
             UPQNBCA(k+1,IP)=QNBCAn
             UPA(k+1,IP)=UPA(K,IP)
             if ( mix_chem ) then
               do ic = 1,nchem
                 UPCHEM(k+1,ip,ic) = chemn(ic)
               enddo
             endif
             if ( scalar_opt > 0 ) then
               do ic = 1,nscalars
                 upscalars(k+1,ip,ic) = scalarsn(ic)
               enddo
             endif
             ktop = MAX(ktop,k)
          ELSE
             if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
                print*,"plume #:",ip," ktop=",ktop
                print*,"area=",UPA(1:ktop,ip)
                print*,"w=",UPW(1:ktop,ip)
             endif
             exit  
          END IF
       ENDDO 

       IF (debug_mf == 1 .and. i==idbg .and. j==jdbg) THEN
          IF (MAXVAL(UPW(:,IP)) > 10.0 .OR. MINVAL(UPA(:,IP)) < zero .OR. &
              MAXVAL(UPA(:,IP)) > Atot .OR. NUP2 > 10) THEN
             
             print *,'flq:',flq,' fltv:',fltv2,' Nup2=',Nup2
             print *,'pblh:',pblh,' wstar:',wstar,' ktop=',ktop
             print *,'sigmaW=',sigmaW,' sigmaTH=',sigmaTH,' sigmaQT=',sigmaQT
             
             print *,'u:',u1
             print *,'v:',v1
             print *,'thl:',thl1
             print *,'UPA:',UPA(:,IP)
             print *,'UPW:',UPW(:,IP)
             print *,'UPTHL:',UPTHL(:,IP)
             print *,'UPQT:',UPQT(:,IP)
             print *,'ENT:',ENT(:,ip)
          ENDIF
       ENDIF
       ktop_plume(ip)=k 
       excess_h =max(excess_h,exc_heat)
       excess_q =max(excess_q,exc_moist)
    ENDDO 
 ELSE
    
    NUP2=0.
 END IF 

 ktop=MIN(ktop,KTE-1)
 IF (ktop == 0) THEN
    ztop = zero
 ELSE
    ztop = zw1(ktop)
 ENDIF

 IF (nup2 > 0) THEN
    
    
    DO ip=1,NUP
       DO k=kts,kte-1
          s_aw1(k)   = s_aw1(k)    + rhoi(k)*UPA(K,ip)*UPW(K,ip)*Psig_w
          s_awthl1(k)= s_awthl1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPTHL(K,ip)*Psig_w
          s_awqt1(k) = s_awqt1(k)  + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQT(K,ip)*Psig_w
          
          
          

             qc_plume = UPQC(K,ip)



          s_awqc1(k) = s_awqc1(k)  + rhoi(k)*UPA(K,ip)*UPW(K,ip)*qc_plume*Psig_w
          s_awqv1(k) = s_awqt1(k)  - s_awqc1(k)
       ENDDO
    ENDDO
    
    if (momentum_opt > 0) then
       do ip=1,nup
          do k=kts,kte-1
             s_awu1(k) = s_awu1(k)   + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPU(K,ip)*Psig_w
             s_awv1(k) = s_awv1(k)   + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPV(K,ip)*Psig_w
          enddo
       enddo
    endif
    
    if (tke_opt > 0) then
       do ip=1,nup
          do k=kts,kte-1
             s_awqke1(k)= s_awqke1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQKE(K,ip)*Psig_w
          enddo
       enddo
       
       if (bl_mynn_closure > 2.5) then
          do ip=1,nup
             do k=kts,kte-1
                s_awqsq1(k)= s_awqsq1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQSQ(K,ip)*Psig_w
             enddo
          enddo
       endif
    endif
    
    if ( mix_chem ) then
       do k=kts,kte-1
          do ip=1,nup
             do ic = 1,nchem
                s_awchem1(k,ic) = s_awchem1(k,ic) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPCHEM(K,ip,ic)*Psig_w
             enddo
          enddo
       enddo
    endif
    
    if ( scalar_opt > 0 ) then
       do k=kts,kte-1
          do ip=1,nup
             do ic = 1,nscalars
                s_awscalars1(k,ic) = s_awscalars1(k,ic) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*upscalars(k,ip,ic)*Psig_w
             enddo
          enddo
       enddo
    endif
    if (numcon_opt > 0) then
       do ip=1,nup
          do k=kts,kte-1
             s_awqnc1(k)  = s_awqnc1(k)   + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQNC(K,ip)*Psig_w
             s_awqni1(k)  = s_awqni1(k)   + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQNI(K,ip)*Psig_w
          enddo
       enddo
    endif
    if (aerosol_opt > 0) then
       do ip=1,nup
          do k=kts,kte-1
             s_awqnwfa1(k)= s_awqnwfa1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQNWFA(K,ip)*Psig_w
             s_awqnifa1(k)= s_awqnifa1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQNIFA(K,ip)*Psig_w
             s_awqnbca1(k)= s_awqnbca1(k) + rhoi(k)*UPA(K,ip)*UPW(K,ip)*UPQNBCA(K,ip)*Psig_w
          enddo
       enddo
    endif

   
   
   
   IF (abs(s_aw1(kts+1)) > 1e-8_kind_phys) THEN
      flx1 = max(s_aw1(kts+1)*(thv1(kts)-thv1(kts+1))/dzi(kts+1),1.0e-6_kind_phys)
   ELSE
      flx1 = zero
      
      
   ENDIF
   adjustment=one
   flt2=max(fltv,zero)
   
   
   IF (flx1 > fluxportion*flt2/dz1(kts) .AND. flx1>zero) THEN
      adjustment = max(0.01, fluxportion*flt2/dz1(kts)/flx1)
      s_aw1      = s_aw1*adjustment
      s_awthl1   = s_awthl1*adjustment
      s_awqt1    = s_awqt1*adjustment
      s_awqc1    = s_awqc1*adjustment
      s_awqv1    = s_awqv1*adjustment
      if (numcon_opt > 0) then
         s_awqnc1   = s_awqnc1*adjustment
         s_awqni1   = s_awqni1*adjustment
      endif
      if (aerosol_opt > 0) then
         s_awqnwfa1 = s_awqnwfa1*adjustment
         s_awqnifa1 = s_awqnifa1*adjustment
         s_awqnbca1 = s_awqnbca1*adjustment
      endif
      IF (momentum_opt > 0) THEN
         s_awu1  = s_awu1*adjustment
         s_awv1  = s_awv1*adjustment
      ENDIF
      IF (tke_opt > 0) THEN
         s_awqke1= s_awqke1*adjustment
         if (bl_mynn_closure>2.5) s_awqsq1= s_awqsq1*adjustment
      ENDIF
      IF ( mix_chem ) THEN
         s_awchem1 = s_awchem1*adjustment
      ENDIF
      IF ( scalar_opt > 0 ) THEN
         s_awscalars1 = s_awscalars1*adjustment
      ENDIF
      UPA = UPA*adjustment
   ENDIF
   if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then
      print*,"Flux adjustment:"
      print*,"adjustment=",adjustment," fluxportion=",fluxportion
      print*,"flt2=",flt2," flx1=",flx1
   endif
      
   
   
   k=kts 
   do ip=1,nup
      edmf_a1(k)  =edmf_a1(k)  +upa(k+1,ip)
      edmf_w1(k)  =edmf_w1(k)  +upa(k+1,ip)*p5*UPW(k+1,ip)
      edmf_qt1(k) =edmf_qt1(k) +upa(k+1,ip)*UPQT(k+1,ip)
      edmf_thl1(k)=edmf_thl1(k)+upa(k+1,ip)*UPTHL(k+1,ip)
      edmf_ent1(k)=edmf_ent1(k)+upa(k+1,ip)*ENT(k+1,ip)
      edmf_qc1(k) =edmf_qc1(k) +upa(k+1,ip)*UPQC(k+1,ip)
   enddo
   do ip=1,nup
      do k=kts+1,ktop_plume(ip)-1  
         upak        =(upa(k+1,ip)*dzi(k) + upa(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         edmf_a1(k)  =edmf_a1(k)  +upak
         edmf_w1(k)  =edmf_w1(k)  +upak*(upw(k+1,ip)*dzi(k)  + upw(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         edmf_qt1(k) =edmf_qt1(k) +upak*(upqt(k+1,ip)*dzi(k) + upqt(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         edmf_thl1(k)=edmf_thl1(k)+upak*(upthl(k+1,ip)*dzi(k)+ upthl(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         edmf_ent1(k)=edmf_ent1(k)+upak*(ent(k+1,ip)*dzi(k)  + ent(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         if (abs(upqc(k,ip)) < 1e-10_kind_phys .and. upqc(k+1,ip) > zero) then
            
            edmf_qc1(k) =edmf_qc1(k) +upak*upqc(k+1,ip)
         else
            edmf_qc1(k) =edmf_qc1(k) +upak*(upqc(k+1,ip)*dzi(k) + upqc(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
         endif
      enddo
   enddo
   
   do ip=1,nup
      k=ktop_plume(ip)
      edmf_a1(k)  =edmf_a1(k)  +upa(k,ip)
      edmf_w1(k)  =edmf_w1(k)  +upa(k,ip)*upw(k,ip)
      edmf_qt1(k) =edmf_qt1(k) +upa(k,ip)*upqt(k,ip)
      edmf_thl1(k)=edmf_thl1(k)+upa(k,ip)*upthl(k,ip)
      edmf_ent1(k)=edmf_ent1(k)+upa(k,ip)*ent(k,ip)
      edmf_qc1(k) =edmf_qc1(k) +upa(k,ip)*upqc(k,ip)
   enddo
   do k=kts,kte-1
      
      if (edmf_a1(k)>0.) then
         edmf_w1(k)  =edmf_w1(k)/edmf_a1(k)
         edmf_qt1(k) =edmf_qt1(k)/edmf_a1(k)
         edmf_thl1(k)=edmf_thl1(k)/edmf_a1(k)
         edmf_ent1(k)=edmf_ent1(k)/edmf_a1(k)
         edmf_qc1(k) =edmf_qc1(k)/edmf_a1(k)
         edmf_a1(k)  =edmf_a1(k)*Psig_w
      endif
   enddo 
   do k=kts,kte-1
      
      if(edmf_a1(k)*edmf_w1(k) > maxmf) maxmf = edmf_a1(k)*edmf_w1(k)
      
      
      tkeprod_up(k)=(abs(edmf_w1(k))**3)*edmf_a1(k)/(b1*max(p5*(el1(k)+el1(k+1)),0.1)) 
   enddo

   
   if ( mix_chem ) then
      do k=kts,kte-1
        do ip=1,nup
          do ic = 1,nchem
            upak 	    =(upa(k+1,ip)*dzi(k) + upa(k,ip)*dzi(k+1))/(dzi(k+1)+dzi(k))
            edmf_chem(k,ic) = edmf_chem(k,ic) + upak*(upchem(k+1,ip,ic)*dzi(k) + upchem(k,ip,ic)*dzi(k+1))/(dzi(k+1)+dzi(k))
          enddo
        enddo
      enddo
      do k=kts,kte-1
         if (edmf_a1(k)>0.) then
            do ic = 1,nchem
               edmf_chem(k,ic) = edmf_chem(k,ic)/edmf_a1(k)
            enddo
         endif
      enddo
   endif
   
   
   
   IF (env_subs) THEN
      DO k=kts+1,kte-1
         
         
         
         
         envi_w(k) = p333*(edmf_w1(k-1)+edmf_w1(k)+edmf_w1(k+1))
         envi_a(k) = p333*(edmf_a1(k-1)+edmf_a1(k)+edmf_a1(k+1))*adjustment
      ENDDO
      
      envi_w(kts) = edmf_w1(kts)
      envi_a(kts) = edmf_a1(kts)
      
      envi_w(kte) = zero
      envi_a(kte) = edmf_a1(kte)
      
      envi_w(kte+1) = zero
      envi_a(kte+1) = edmf_a1(kte)
      
      
      
      IF (envi_w(kts) > 0.9*dz1(kts)/dt) THEN
         sublim = 0.9*dz1(kts)/dt/envi_w(kts)
      ELSE
         sublim = one
      ENDIF
      
      DO k=kts,kte
         temp=envi_a(k)
         envi_a(k)=one-temp
         envi_w(k)=csub*sublim*envi_w(k)*temp/(1.-temp)
      ENDDO
      
      
      sub_thl1(kts)= 0.5*envi_w(kts)*envi_a(kts)*                               &
                     (rho1(kts+1)*thl1(kts+1)-rho1(kts)*thl1(kts))/dzi(kts)/rhoi(kts)
      sub_sqv1(kts)= 0.5*envi_w(kts)*envi_a(kts)*                               &
                     (rho1(kts+1)*qv1(kts+1)-rho1(kts)*qv1(kts))/dzi(kts)/rhoi(kts)
      DO k=kts+1,kte-1
         dzi(k)     = 0.5*(dz1(k)+dz1(k+1))
         sub_thl1(k)= 0.5*(envi_w(k)+envi_w(k-1))*0.5*(envi_a(k)+envi_a(k-1)) * &
                      (rho1(k+1)*thl1(k+1)-rho1(k)*thl1(k))/dzi(k)/rhoi(k)
         sub_sqv1(k)= 0.5*(envi_w(k)+envi_w(k-1))*0.5*(envi_a(k)+envi_a(k-1)) * &
                      (rho1(k+1)*qv1(k+1)-rho1(k)*qv1(k))/dzi(k)/rhoi(k)
      ENDDO

      DO k=kts,KTE-1
         det_thl1(k)=Cdet*(envm_thl(k)-thl1(k))*envi_a(k)*Psig_w
         det_sqv1(k)=Cdet*(envm_sqv(k)-qv1(k))*envi_a(k)*Psig_w
         det_sqc1(k)=Cdet*(envm_sqc(k)-qc1(k))*envi_a(k)*Psig_w
      ENDDO

      IF (momentum_opt > 0) THEN
         sub_u1(kts)=0.5*envi_w(kts)*envi_a(kts)*                               &
                     (rho1(kts+1)*u1(kts+1)-rho1(kts)*u1(kts))/dzi(kts)/rhoi(kts)
         sub_v1(kts)=0.5*envi_w(kts)*envi_a(kts)*                               &
                     (rho1(kts+1)*v1(kts+1)-rho1(kts)*v1(kts))/dzi(kts)/rhoi(kts)
         DO k=kts+1,kte-1
            sub_u1(k)=0.5*(envi_w(k)+envi_w(k-1))*0.5*(envi_a(k)+envi_a(k-1)) * &
                       (rho1(k+1)*u1(k+1)-rho1(k)*u1(k))/dzi(k)/rhoi(k)
            sub_v1(k)=0.5*(envi_w(k)+envi_w(k-1))*0.5*(envi_a(k)+envi_a(k-1)) * &
                       (rho1(k+1)*v1(k+1)-rho1(k)*v1(k))/dzi(k)/rhoi(k)
         ENDDO

         DO k=kts,KTE-1
           det_u1(k) = Cdet*(envm_u(k)-u1(k))*envi_a(k)*Psig_w
           det_v1(k) = Cdet*(envm_v(k)-v1(k))*envi_a(k)*Psig_w
         ENDDO
       ENDIF
   ENDIF 

   
   
   DO k=kts,kte-1
      edmf_th1(k)= edmf_thl1(k) + xlvcp/ex1(k)*edmf_qc1(K)
      qti(k+1)   =(qt1(k)*dz1(k+1)+qt1(k+1)*dz1(k))/(dz1(k+1)+dz1(k))
   ENDDO
   qti(kts)=qt1(kts)
   



   cb_check = 0
   do k=kts+1,kte-2
      if (k > KTOP) exit
      if (edmf_qc1(k) > zero ) then 
         if (cb_check == 0) then
            mf_at_cb = edmf_a1(k)*edmf_w1(k)
            cb_check = 1
          endif
          
          Aup = edmf_a1(k)
          THp = edmf_th1(k)
          QTp = edmf_qt1(k)
          QCp = edmf_qc1(k)
          dqwdz = (qti(k)-qti(k+1))/dz1(k) 
          
          
          
          esat = esat_blend(tk1(k))
          
          qsl=ep_2*esat/max(1.e-7_kind_phys,(pres1(k)-ep_3*esat)) 

          
          xl = xl_blend(tk1(k))               
          qsat_tk = qsat_blend(tk1(k),pres1(k))
                                              
          rsl = xl*qsat_tk / (r_v*tk1(k)**2)  
                                              
          cpm = cp + qt1(k)*cpv               
          a   = one/(one + xl*rsl/cpm)        
          b9  = a*rsl                         

          q2p = xlvcp/ex1(k)
          pt  = thl1(k) +q2p*QCp*Aup 
          bb  = b9*tk1(k)/pt 
                             
                             
                             
                             
                             
          qww   = one+p608*qt1(k)
          alpha = p608*pt
          beta  = pt*xl/(tk1(k)*cp) - 1.61_kind_phys*pt
          

          
          if (a > zero) then
             f = MIN(one/a, four)      
          else
             f = one
          endif

          
          
          

          
          

          
          tauc = 1800._kind_phys
          sigq = five * Aup * edmf_w1(k) * max(4e-3_kind_phys, tauc*max(zero, dqwdz))

          
          
          
          sigq = max(sigq, three * Aup * (QTp - qt1(k)))
          
          

          
          qmq = qt1(k) - qsat_tk                
          if (qmq > zero) sigq = min(qsat_tk*0.015_kind_phys, sigq)
          Q1  = qmq/sigq                        

          if ((landsea-1.5).GE.zero) then   
             
             
             
             
             
             cffac = min(max(zero, mf_at_cb - 0.05_kind_phys)/0.05_kind_phys, one)
             cf_mf = min(max(p5 + 0.36_kind_phys * atan(1.8_kind_phys*(Q1+p2+cffac)),0.01_kind_phys), one)
             cf_mf = max(cf_mf, 1.2_kind_phys * Aup)
             
             
          else                              
             
             
             
             
             cf_mf = min(max(p5 + 0.36_kind_phys * atan(1.8_kind_phys*(Q1+p2)),0.01_kind_phys), one)
             cf_mf = max(cf_mf, 1.8_kind_phys * Aup)
             
          endif

          
          
          
          
          
          
          

          
          
          
          qcfac = 1.20_kind_phys + 1.0_kind_phys*min(max(zero, mf_at_cb - 0.04_kind_phys)/0.05_kind_phys, one)
          if ((landsea-1.5).GE.zero) then  
             
             
             
             
             
             qc_mf     = qcfac * (QCp * Aup) 
          else                             
             if (QCp * Aup > 5e-5) then
                qc_mf     = 1.86_kind_phys * (QCp * Aup) - 2.2e-5_kind_phys
             else
                qc_mf     = 1.20_kind_phys * (QCp * Aup)
             endif
          endif
            
            
            
            
            wt2          = one - min(one, max(zero, (cf_mf - thirty))/(hundred-thirty))  
            qc_mf_min    = wt2*qsat_tk*0.01_kind_phys*cf_mf + (one-wt2)*qsat_tk*0.025_kind_phys*cf_mf
            qc_mf        = max(qc_mf, qc_mf_min)
            
            cf_strat     = cldfra_bl1(k)
            qc_strat     = qc_bl1(k)
            cldfra_bl1(k)= max(cf_mf, cf_strat)
            pct_mf       = cf_mf/cldfra_bl1(k)
            qc_bl1(k)    = qc_mf*pct_mf + (one-pct_mf)*qc_strat

            
            
            
            
            
            
               Q1=max(Q1,-2.25)
            
            
            

            if (Q1 .ge. one) then
               Fng = one
            elseif (Q1 .ge. -1.7 .and. Q1 .lt. one) then
               Fng = EXP(-p4*(Q1-one))
            elseif (Q1 .ge. -2.5 .and. Q1 .lt. -1.7) then
               Fng = three + EXP(-3.8*(Q1+1.7))
            else
               Fng = min(23.9 + EXP(-1.6*(Q1+2.5)), 60.)
            endif

            
            vt1(k) = qww   - (1.5*Aup)*beta*bb*Fng - one
            vq1(k) = alpha + (1.5*Aup)*beta*a*Fng  - tv0
         endif 
      enddo 

ENDIF  


if (ktop > 0) then
   maxqc = maxval(edmf_qc1(1:ktop)) 
   if ( maxqc < 1.E-8) maxmf = -1.*maxmf
endif




if (debug_mf == 1 .and. i==idbg .and. j==jdbg) then








 


















   print*,"mean updraft properties at end of mf component:"
   print*,' edmf_a1',edmf_a1(1:max(1,ktop))
   print*,' edmf_w1',edmf_w1(1:max(1,ktop))
   print*,' edmf_qt1:',edmf_qt1(1:max(1,ktop))
   print*,' edmf_thl1:',edmf_thl1(1:max(1,ktop))
 
ENDIF 



END SUBROUTINE DMP_MF



subroutine condensation_edmf(QT,THL,P,zagl,THV,QC)



  use module_bl_mynnedmf_common, only: p1000mb,xlvcp,rcp,rvovrd,&
       zero,one,p5,kind_phys
  
real(kind_phys),intent(in)   :: QT,THL,P,zagl
real(kind_phys),intent(inout):: THV
real(kind_phys),intent(inout):: QC

integer :: niter,ni
real(kind_phys):: diff,exn,t,th,qs,qcold









  niter=50

  diff=1.e-6_kind_phys

  EXN=(P/p1000mb)**rcp
  QC=zero  
  do ni=1,NITER
     T=EXN*THL + xlvcp/EXN*QC
     QS=qsat_blend(T,P)
     QCOLD=QC
     QC=p5*QC + p5*MAX((QT-QS),zero)
     if (abs(QC-QCOLD)<Diff) exit
  enddo

  T=EXN*THL + xlvcp/EXN*QC
  QS=qsat_blend(T,P)
  QC=max(QT-QS,zero)

  
  if(zagl < 100.)QC=zero

  
  THV=(THL+xlvcp/EXN*QC)*(one+QT*(rvovrd-one)-rvovrd*QC)







  
  
  

end subroutine condensation_edmf


subroutine condensation_ddmf(qt,thl,p,zagl,thv,qc,qi,debug_dd)



  use module_bl_mynnedmf_common, only: p1000mb,xlvcp,xlscp,rcp, &
       rvovrd,p608,zero,one,p5,kind_phys
  
real(kind_phys),intent(in)   :: qt,thl,p,zagl
real(kind_phys),intent(inout):: thv,qc,qi

integer :: niter,ni,debug_dd
real(kind_phys):: diff,exn,t,th,qs,qx,qxold,frac_ice,frac_liq,thvin


  niter=50

  diff=1.e-6

  if ((qi+qc) .gt. zero) then
     frac_ice = qi/(qi+qc)
     frac_liq = one - frac_ice
  else
     frac_ice = zero
     frac_liq = zero
  endif

  if (debug_dd == 1) then
     thvin = thv
     print*,"------- in consensation_ddmf-------------"
     print*,"input qc=",qc," qi=",qi," frac_liq=",frac_liq
  endif

  exn=(p/p1000mb)**rcp
  do ni=1,niter
     t=exn*thl + xlvcp/exn*qc + xlscp/exn*qi
     qs=qsat_blend(t,p)
     qxold=qc+qi
     qx=qc+qi
     qx=p5*qx + p5*max((qt-qs),zero)
     qc=frac_liq*qx
     qi=frac_ice*qx
     if (abs(qx-qxold)<diff) exit
  enddo

  t=exn*thl + xlvcp/exn*qc + xlscp/exn*qi
  qs=qsat_blend(t,p)
  qx=max(qt-qs,zero)
  qc=frac_liq*qx
  qi=frac_ice*qx

  

  
  thv=(thl + xlvcp/exn*qc + xlscp/exn*qi)*(one + p608*(qt-qx) - qx)







  
  
  


  

  if (debug_dd == 1) then
     print*,"output qc=",qc," qi=",qi," frac_liq=",frac_liq
     print*,"input thv=",thvin," out thv=",thv," diff=",thv-thvin
     print*,"------- exiting consensation_ddmf----------"
  endif

end subroutine condensation_ddmf



subroutine condensation_edmf_r(QT,THL,P,zagl,THV,QC)




  use module_bl_mynnedmf_common, only: p1000mb,rcp,rvovrd,xlvcp,&
       one,zero,kind_phys
  
real(kind_phys),intent(in)   :: QT,THV,P,zagl
real(kind_phys),intent(inout):: THL, QC

integer :: niter,ni
real(kind_phys):: diff,exn,t,th,qs,qcold


  niter=50

  diff=2.e-5_kind_phys

  EXN=(P/p1000mb)**rcp
  
  T = THV*EXN
  
  

  QC=zero

  do ni=1,NITER
     QCOLD = QC
     T = EXN*THV/(one+QT*(rvovrd-one)-rvovrd*QC)
     QS=qsat_blend(T,P)
     QC= MAX((QT-QS), zero)
     if (abs(QC-QCOLD)<Diff) exit
  enddo
  THL = (T - xlvcp*QC)/EXN

end subroutine condensation_edmf_r




















subroutine ddmp_mf(kts,kte,dt,dx,zw,dz,p,            &
              &u,v,th,thl,thv,tk,qt,qv,qc,qi,        &
              &qnc,qni,qnwfa,qnifa,                  &
              &qke,rho,exner,                        &
              &qc_bl1,qi_bl1,cldfra_bl1,             &
              &ust,flt,flq,fltv,pblh,kpbl,           &
              &edmf_a_dd,edmf_w_dd, edmf_qt_dd,      &
              &edmf_thl_dd,edmf_ent_dd,edmf_qc_dd,   &
              &sd_aw,sd_awthl,sd_awqt,               &
              &sd_awqv,sd_awqc,sd_awqi,              &
              &sd_awqnc,sd_awqni,                    &
              &sd_awqnwfa,sd_awqnifa,                &
              &sd_awu,sd_awv,sd_awqke,               &
              &tkeprod_dn,el,                        &
              &rthraten,psig,                        &
              &maxmf_dd,maxwidth_dd                  )

  use module_bl_mynnedmf_common, only: cp,grav,p608, &
       xlvcp,b1,one,two,zero,eight,p1,p2,p333,p5,    &
       p666,kind_phys

        integer, intent(in) :: kts,kte,kpbl
        real(kind_phys), dimension(kts:kte), intent(in) ::            &
            u,v,th,thl,tk,qt,qv,qc,qi,thv,p,qke,rho,exner,            &
            qnc,qni,qnwfa,qnifa,dz,qc_bl1,qi_bl1,cldfra_bl1,el
        real(kind_phys), dimension(kts:kte), intent(in) :: rthraten
        
        real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
        real(kind_phys), intent(in)  :: flt,flq,fltv
        real(kind_phys), intent(in)  :: dt,dx,ust,pblh,psig
   
        real(kind_phys), dimension(kts:kte), intent(inout) ::         &
        edmf_a_dd,   edmf_w_dd,   edmf_qt_dd,  edmf_thl_dd,           &
        edmf_ent_dd, edmf_qc_dd,  tkeprod_dn
   
        real(kind_phys), dimension(kts:kte+1) ::                      &
            sd_aw, sd_awthl, sd_awu, sd_awv,                          &
            sd_awqt, sd_awqc, sd_awqv, sd_awqi, sd_awqnc, sd_awqni,   &
            sd_awqnwfa, sd_awqnifa, sd_awqke, sd_aw2
   
        real(kind_phys), intent(inout) :: maxmf_dd,maxwidth_dd

   
        integer, parameter::                &
            & ndd            = 5,           &  
            & kmin           = 3               
        real(kind_phys),parameter ::        &
            & minddd         = 100.,        &  
            & maxddd         = 500.,        &  
            & zmin           = 50.,         &  
            & dz200          = 200.,        &  
            & cooling_thresh = -0.00010        
                              
                              
                              
        real(kind_phys)::                   &
            & maxdd2,                       &  
            &    ddd,                       &  
            &     dl,                       &  
            &    add,                       &  
            & minexc,                       &  
            & maxexc                           
  
        integer,         dimension(1:ndd) :: dd_initk
  
        real(kind_phys), dimension(kts:kte+1,1:ndd) ::                &
            downw,downthl,downqt,downqc,downa,downu,downv,downthv,    &
            downqi,downqnc,downqni,downqnwfa,downqnifa
  
        real(kind_phys), dimension(kts:kte+1,1:ndd) :: ent
  
        real(kind_phys), dimension(kts:kte):: massflux
        integer :: k,i,ki,qltop,qlbase
        real(kind_phys):: qstar,thstar,sigmaw,sigmaqt,                &
            sigmath,z0,pwmin,pwmax,wmin,wmax,went,mindownw
        real(kind_phys):: qtn,thln,thvn,qcn,un,vn,qken,wn2,wn,        &
            qin,qncn,qnin,qnwfan,qnifan,                              &
            thvk,pk,entexp,entw,beta_dm,entexm,rho_int
        real(kind_phys):: jump_thv,jump_qt,jump_thetal,               &
            refthl,refthv,refthlv,refqt,refqc,refqi,refqnc,refqni,    &
            refqnwfa,refqnifa,refu,refv,refqke,refqt2,reftk,refp,     &
            qx_k,qx_km1,refthvm1,cftop,crate,ac_wsp,wspd_pbl,         &
            maxcooling
        real(kind_phys):: dthvx,tmp1,rcldb,ent_eff

  
        real(kind_phys):: radflux, f0, wstar_rad, dz_ent
        logical :: cloudflg
        logical :: singlelayer             
        real(kind_phys):: sigq,xl,rsl,cpm,a,diffqt,dp,                &
            fng,qww,alpha,beta,bb,f,pt,t,q2p,b9,satvp,rhgrid,         &
            def_th,def_qt,frac_liq,frac_ice,qs1,qs2

  
        real(kind_phys),parameter ::                                  &
            &wa=1., wb=1.5, z00=100.
        real(kind_phys) :: buoy,bcoeff,ecoeff

  
        integer, parameter :: debug_dd=0   

  
  
   downw      =zero
   downthl    =zero
   downthv    =zero
   downqt     =zero
   downa      =zero
   downu      =zero
   downv      =zero
   downqc     =zero
   downqi     =zero
   downqnc    =zero
   downqni    =zero
   downqnwfa  =zero
   downqnifa  =zero
   ent        =zero
   dd_initk   =0

   edmf_a_dd  =zero
   edmf_w_dd  =zero
   edmf_qt_dd =zero
   edmf_thl_dd=zero
   edmf_ent_dd=zero
   edmf_qc_dd =zero

   sd_aw      =zero
   sd_awthl   =zero
   sd_awqt    =zero
   sd_awqv    =zero
   sd_awqc    =zero
   sd_awqi    =zero
   sd_awqnc   =zero
   sd_awqni   =zero
   sd_awqnwfa =zero
   sd_awqnifa =zero
   sd_awu     =zero
   sd_awv     =zero
   sd_awqke   =zero

   
   cloudflg   =.false.
   singlelayer=.false.
   qltop      =1     
   qlbase     =1     
   crate      =zero
   do k = max(kmin,kpbl-5),min(kpbl+20,kte-1)
      maxcooling = minval(rthraten(k-1:k+1))
      
      if ((cldfra_bl1(k).gt.p5 .and. cldfra_bl1(k+1).lt.p5) .and. &  
          (maxcooling  .lt. cooling_thresh)) then                    
         
         cloudflg =.true.
         if (maxcooling < crate) then
            qltop    =k               
            cftop    =cldfra_bl1(k)
            crate    =maxcooling      
            if (cldfra_bl1(k-1).lt.p2)singlelayer=.true.
         endif
      endif
   enddo

   
   maxdd2  = min(maxddd, 1.2_kind_phys*dx)                                       
   maxdd2  = min(maxdd2, -100._kind_phys*crate*3600._kind_phys + 200._kind_phys) 
   maxdd2  = min(maxdd2, 80._kind_phys + p666*zw(qltop+1))                       
   maxwidth_dd = maxdd2
   dl      = max(maxdd2-minddd, zero)/real(ndd-1)
   
   
   if (cloudflg .and. ( dl .gt. zero)) then
      do k = qltop, kts, -1
         qx_k =max(qt(k), qc_bl1(k) +qi_bl1(k))     
         if (qx_k .gt. 1e-6_kind_phys) then
            qlbase = k 
         endif
      enddo

      do i=1,ndd
         
         dd_initk(i) = qltop
      enddo

      
      f0 = zero
      do k = max(kmin,qltop-2), qltop+1
         radflux = rthraten(k) * exner(k)     
         dp      = p5 * (( p(k) - p(k+1) ) + ( p(k-1) - p(k) ))
         radflux = radflux * cp / grav * dp   
         if ( radflux < zero ) f0 = abs(radflux) + f0
      enddo
      f0 = min(max(f0, one), 200._kind_phys)  

      
      
      
      
      
      add      = min( f0*0.003, 0.3_kind_phys)
      
      wspd_pbl = sqrt(max(u(kts)**2 + v(kts)**2, 0.01_kind_phys))
      ac_wsp   = one - min(max(zero, wspd_pbl - 10._kind_phys)/15._kind_phys, one)
      add      = add * min(ac_wsp, psig)

      
      dz_ent      = p5 * (dz(qltop+1) + dz(qltop))
      jump_thv    = (thv(qltop+1) - thv(qltop)) / dz_ent * dz200
      jump_qt     = (qt(qltop+1)  - qt(qltop))	/ dz_ent * dz200
      jump_thetal = (thl(qltop+1) - thl(qltop))	/ dz_ent * dz200

      ki = qltop  
      if (singlelayer) then 
         refthl  = thl(ki)
         refthlv = refthl*(1.+p608*qv(ki))
         refthv  = thv(ki)
         refthvm1= thv(ki-1)
         reftk   = tk(ki)
         refqt   = qt(ki)
         refqc   = qc(ki)
         refqi   = qi(ki)
         refqnc  = qnc(ki)
         refqni  = qni(ki)
         refqnwfa= qnwfa(ki)
         refqnifa= qnifa(ki)
         refu    = u(ki)
         refv    = v(ki)
         refqke  = qke(ki)
         refp    = p(ki)
      else                  
         refthl  = (thl(ki-1)*dz(ki)   + thl(ki)*dz(ki-1))  /(dz(ki)+dz(ki-1))
         refthlv = refthl*(1.+p608*(qv(ki-1)*dz(ki) + qv(ki)*dz(ki-1))/(dz(ki)+dz(ki-1)))
         refthv  = (thv(ki-1)*dz(ki)   + thv(ki)*dz(ki-1))  /(dz(ki)+dz(ki-1))
         refthvm1= (thv(ki-2)*dz(ki-1) + thv(ki-1)*dz(ki-2))/(dz(ki-1)+dz(ki-2)) 
         reftk   = (tk(ki-1)*dz(ki)    + tk(ki)*dz(ki-1))   /(dz(ki)+dz(ki-1))
         refqt   = (qt(ki-1)*dz(ki)    + qt(ki)*dz(ki-1))   /(dz(ki)+dz(ki-1))
         refqc   = (qc(ki-1)*dz(ki)    + qc(ki)*dz(ki-1))   /(dz(ki)+dz(ki-1))
         refqi   = (qi(ki-1)*dz(ki)    + qi(ki)*dz(ki-1))   /(dz(ki)+dz(ki-1))
         refqnc  = (qnc(ki-1)*dz(ki)   + qnc(ki)*dz(ki-1))  /(dz(ki)+dz(ki-1))
         refqni  = (qni(ki-1)*dz(ki)   + qni(ki)*dz(ki-1))  /(dz(ki)+dz(ki-1))
         refqnwfa= (qnwfa(ki-1)*dz(ki) + qnwfa(ki)*dz(ki-1))/(dz(ki)+dz(ki-1))
         refqnifa= (qnifa(ki-1)*dz(ki) + qnifa(ki)*dz(ki-1))/(dz(ki)+dz(ki-1))
         refu    = (u(ki-1)*dz(ki)     + u(ki)*dz(ki-1))    /(dz(ki)+dz(ki-1))
         refv    = (v(ki-1)*dz(ki)     + v(ki)*dz(ki-1))    /(dz(ki)+dz(ki-1))
         refqke  = (qke(ki-1)*dz(ki)   + qke(ki)*dz(ki-1))  /(dz(ki)+dz(ki-1))
         refp    = (p(ki-1)*dz(ki)     + p(ki)*dz(ki-1))    /(dz(ki)+dz(ki-1))
      endif

      
      
      wstar_rad = 1.25_kind_phys * ( grav*dz200 * f0 / (refthl * rho(ki) * cp) ) **p333
      wstar_rad = min(max(wstar_rad, p1), 1.5_kind_phys)      

      
      dthvx     = (thl(ki+2) + th(ki+2)*p608*qt(ki+2)) - &
                  (thl(ki)   + th(ki)  *p608*qt(ki))
      dthvx     = dthvx * dz200/dz_ent  
      dthvx     = max(dthvx, p1)
      rcldb     = max(refqc+refqi, qc_bl1(ki)+qi_bl1(ki))  
      tmp1      = xlvcp * rcldb/(exner(ki)*dthvx)
      
      
      ent_eff   = max(zero, min(one, p2 + p2*eight*tmp1))

      went      = p333 * ent_eff * wstar_rad
      qstar     = went * jump_qt / wstar_rad
      thstar    = -f0/rho(ki)/cp/wstar_rad - went*jump_thv/wstar_rad

      sigmaw    = 0.2 * wstar_rad   
      sigmaqt   = 50. * abs(qstar)  
      sigmath   = 1.5 * abs(thstar) 

      pwmin     = -1. 
      pwmax     = -3.
      wmin      = max(min(sigmaw*pwmin, -0.1), -0.2)
      wmax      = max(min(sigmaw*pwmax, -0.2), -0.5)

      
      minexc = min(-0.06     , 0.5*(refthvm1 - refthv)) 
      maxexc = min(minexc-0.1, -0.6          )
      maxexc = max(maxexc    , thv(1) - refthlv)        
      def_th = max(min(0.05*(-0.3)*sigmath/sigmaw, minexc), maxexc)

      
      qs1    = qsat_blend(reftk       ,refp)
      qs2    = qsat_blend(reftk+def_th,refp)
      refqt2 = refqt*qs2/qs1
      
      def_qt = min(qs2 - qs1,  -0.33*(refqc+refqi)) 
      

      if (refqc+refqi > zero) then
         frac_liq = refqc/(refqc+refqi)
         frac_ice = one - frac_liq
      else
         frac_liq = zero
         frac_ice = zero
      endif

      if (debug_dd .eq. 1) then
         print*,"-------------------------------------"
         print*,"found conditions for downdraft mixing"
         print*,"qltop=",qltop," qlbase=",qlbase
         print*,"q*=",qstar," theta*=",thstar," went=",went
         print*,"f0=",f0," jump_thv=",jump_thv,"w*rad=",wstar_rad
         print*,"u*=",ust," jump_qt=",jump_qt," fltv=",fltv
         print*,"grav=",grav," pblh=",pblh," thv(1)=",thv(1)
         print*,"p(1)=",p(1)," jump_thetal=",jump_thetal
         print*,"sigmaw=",sigmaw," wmin=",wmin," wmax=",wmax
      endif

      
      
      do i=1,ndd
         ki = dd_initk(i)

         
         downw(ki,i)=wmin-(abs(wmax)-abs(wmin))/real(ndd-1)*(i-1)


         
         downa(ki,i)     = add/real(ndd)
         downu(ki,i)     = refu
         downv(ki,i)     = refv
         downqc(ki,i)    = max(refqc + def_qt*frac_liq, zero)
         downqi(ki,i)    = max(refqi + def_qt*frac_ice, zero)
         downqnc(ki,i)   = refqnc
         downqni(ki,i)   = refqni
         downqnwfa(ki,i) = refqnwfa
         downqnifa(ki,i) = refqnifa
         downqt(ki,i)    = refqt  + def_qt
         downthv(ki,i)   = refthv + def_th*real(i)/real(ndd)
         downthl(ki,i)   = refthl + def_th*real(i)/real(ndd)

         




      enddo

      if (debug_dd .eq. 1) then
         print*,"=====initialized downdraft properties===="
         print*,"downw=",downw(ki,:)
         print*,"downa=",downa(ki,:)
         print*,"downthv=",downthv(ki,:)
         print*,"def_qt=",def_qt," def_th=",def_th," tot area=",add
         print*,"begin integration of downdrafts (smallest to largest):"
      endif

      do i=1,ndd
         ddd  = minddd + dl*real(i-1) 
         
         thvn = downthv(ki,i)
         qcn  = downqc(ki,i)
         qin  = downqi(ki,i)
         do k=dd_initk(i)-1,kts+1,-1
            
            wmin = p1 + ddd*0.0005
            
            
            
            ent(k,i) = 0.18/(min(max(abs(downw(k+1,i)),wmin),0.6)*ddd)
            
            
            ent(k,i) = max(ent(k,i),0.0003)
            ent(k,i) = max(ent(k,i),0.06/zw(k))
            ent(k,i) = min(ent(k,i),0.9/dz(k)) 

            
            
            
            entexp  =ent(k,i)*dz(k)        
            entexm  =ent(k,i)*dz(k)*p5     

            qtn     =downqt(k+1,i)   *(one-entexp) + qt(k)*entexp
            thln    =downthl(k+1,i)  *(one-entexp) + thl(k)*entexp
            qncn    =downqnc(k+1,i)  *(one-entexp) + qnc(k)*entexp
            qnin    =downqni(k+1,i)  *(one-entexp) + qni(k)*entexp
            qnwfan  =downqnwfa(k+1,i)*(one-entexp) + qnwfa(k)*entexp
            qnifan  =downqnifa(k+1,i)*(one-entexp) + qnifa(k)*entexp
            un      =downu(k+1,i)    *(one-entexm) + u(k)*entexm
            vn      =downv(k+1,i)    *(one-entexm) + v(k)*entexm
            





            
            pk  =(p(k-1)*dz(k)+p(k)*dz(k-1))/(dz(k)+dz(k-1))
            call condensation_ddmf(qtn,thln,pk,zw(k),thvn,qcn,qin,debug_dd)

            
            thvk =(thv(k-1)*dz(k)+thv(k)*dz(k-1))/(dz(k)+dz(k-1))
            buoy =grav*(thvn/thvk - one)
            
            if (buoy > zero) then
               bcoeff =  0.2  
               ecoeff = -2.0
            else
               
               bcoeff = 0.2*(one - exp(-zw(k)/(0.15_kind_phys*pblh)))
               ecoeff = -2.0_kind_phys - exp(-zw(k)/(0.15_kind_phys*pblh))
               buoy   = buoy*(one - exp(-zw(k)/(0.15_kind_phys*pblh)))
            endif
            if (k==kts+2) buoy=max(buoy,0.01)
            if (k==kts+1) buoy=max(buoy,0.05)











            mindownw = min(downw(k+1,i),-0.2_kind_phys)
            wn = downw(k+1,i) + (ecoeff*ent(k,i)*downw(k+1,i)        &
                              - bcoeff*buoy/mindownw)*min(dz(k), 250.)

            
            
            if (wn < downw(k+1,i) - min(1.25*dz(k)/200., 2.0))then
                wn = downw(k+1,i) - min(1.25*dz(k)/200., 2.0)
            endif
            
            if (wn > downw(k+1,i) + min(1.25*dz(k)/200., 2.0))then
                wn = downw(k+1,i) + min(1.25*dz(k)/200., 2.0)
            endif
            wn = max(min(wn,zero), -2.0_kind_phys)

            if (debug_dd .eq.1) then
               print *, "downdraft #:", i," diameter:",ddd,"==========="
               print *, "  k       =",      k,      " z    =", zw(k)
               print *, "  ent     =",ent(k,i),     " bouy =", buoy
               print *, "  downthv =",   thvn,      " thvk =", thvk
               print *, "  downthl =",   thln,      " thl  =", thl(k)
               print *, "  downqt  =",   qtn ,      " qt   =", qt(k)
               print *, "  downqc  =",   qcn ,      " qc   =", qc(k)
               print *, "  downw+1 =",downw(k+1,i), " wn2  =", wn
            endif

            if (wn .lt. 0.) then 
               downw(k,i)     = wn 
               downthv(k,i)   = thvn
               downthl(k,i)   = thln
               downqt(k,i)    = qtn
               downqc(k,i)    = qcn
               downqi(k,i)    = qin
               downqnc(k,i)   = qncn
               downqni(k,i)   = qnin
               downqnwfa(k,i) = qnwfan
               downqnifa(k,i) = qnifan
               downu(k,i)     = un
               downv(k,i)     = vn
               downa(k,i)     = downa(k+1,i)
            else
               









               exit
            endif
         enddo
      enddo

      downw(1,:) = zero 
      downa(1,:) = zero

      
      
      
      do k=qltop,kts,-1
         do i=1,ndd
            edmf_a_dd(k)  =edmf_a_dd(k)  +downa(k,i)
            edmf_w_dd(k)  =edmf_w_dd(k)  +downa(k,i)*downw(k,i)
            edmf_qt_dd(k) =edmf_qt_dd(k) +downa(k,i)*downqt(k,i)
            edmf_thl_dd(k)=edmf_thl_dd(k)+downa(k,i)*downthl(k,i)
            edmf_ent_dd(k)=edmf_ent_dd(k)+downa(k,i)*ent(k,i)
            edmf_qc_dd(k) =edmf_qc_dd(k) +downa(k,i)*downqc(k,i)
         enddo

         if (edmf_a_dd(k) > 0.) then
            edmf_w_dd(k)  =edmf_w_dd(k)  /edmf_a_dd(k)
            edmf_qt_dd(k) =edmf_qt_dd(k) /edmf_a_dd(k)
            edmf_thl_dd(k)=edmf_thl_dd(k)/edmf_a_dd(k)
            edmf_ent_dd(k)=edmf_ent_dd(k)/edmf_a_dd(k)
            edmf_qc_dd(k) =edmf_qc_dd(k) /edmf_a_dd(k)
         endif
         
         tkeprod_dn(k)=(abs(edmf_w_dd(k))**3)*edmf_a_dd(k)/(b1*max(el(k),p5))
         tkeprod_dn(k)=min(0.0004_kind_phys, tkeprod_dn(k))
      enddo
      
      
      tkeprod_dn(qltop+1)=went**3*edmf_a_dd(qltop)/(b1*max(el(qltop+1),p5))
      massflux = edmf_a_dd * edmf_w_dd
      maxmf_dd = abs(minval(massflux))

      if (debug_dd .eq. 2) then
         do k=kts,kte
            if (tkeprod_dn(k) > 0.0004_kind_phys .or. edmf_w_dd(k) < -two) then
               print*,"-------------------------------------"
               print*,"found strange behavior in downdrafts at",k
               print*,"qltop=",qltop," qlbase=",qlbase," a(top)=",edmf_a_dd(qltop)
               print*,"q*=",qstar," theta*=",thstar," went=",went
               print*,"f0=",f0," jump_thv=",jump_thv,"w*rad=",wstar_rad
               print*,"u*=",ust," jump_qt=",jump_qt," fltv=",fltv
               print*,"grav=",grav," pblh=",pblh," thv(1)=",thv(1)
               print*,"p(1)=",p(1)," jump_thl=",jump_thetal," ent_eff=",ent_eff
               print*,"sigmaw=",sigmaw," wmin=",wmin," wmax=",wmax
               print*,"tke prod=",tkeprod_dn(k)," w=",edmf_w_dd(k)," a(k)=",edmf_a_dd(k)
            endif
         enddo
      endif
      
      
      do k=kts,qltop
         rho_int = (rho(k)*dz(k+1)+rho(k+1)*dz(k))/(dz(k+1)+dz(k))
         do i=1,ndd
            sd_aw(k)   =sd_aw(k)   +rho_int*downa(k,i)*abs(downw(k,i))
            sd_awthl(k)=sd_awthl(k)+rho_int*downa(k,i)*downw(k,i)*downthl(k,i)
            sd_awqt(k) =sd_awqt(k) +rho_int*downa(k,i)*downw(k,i)*downqt(k,i)
            sd_awqc(k) =sd_awqc(k) +rho_int*downa(k,i)*downw(k,i)*downqc(k,i)
            sd_awqi(k) =sd_awqi(k) +rho_int*downa(k,i)*downw(k,i)*downqi(k,i)
            sd_awqnc(k)=sd_awqnc(k)+rho_int*downa(k,i)*downw(k,i)*downqnc(k,i)
            sd_awqni(k)=sd_awqni(k)+rho_int*downa(k,i)*downw(k,i)*downqni(k,i)
            sd_awqnwfa(k)=sd_awqnwfa(k)+rho_int*downa(k,i)*downw(k,i)*downqnwfa(k,i)
            sd_awqnifa(k)=sd_awqnifa(k)+rho_int*downa(k,i)*downw(k,i)*downqnifa(k,i)
            sd_awu(k)  =sd_awu(k)  +rho_int*downa(k,i)*downw(k,i)*downu(k,i)
            sd_awv(k)  =sd_awv(k)  +rho_int*downa(k,i)*downw(k,i)*downv(k,i)
         enddo
         sd_awqv(k) = sd_awqt(k)  - sd_awqc(k) - sd_awqi(k)
      enddo

   endif 

end subroutine ddmp_mf


SUBROUTINE SCALE_AWARE(dx,pblh,Psig_bl,Psig_shcu)

    
    
    
    
    
    
    
    

  use module_bl_mynnedmf_common, only: zero,one,ten,kind_phys
  
    real(kind_phys), intent(in)  :: dx,pblh
    real(kind_phys), intent(out) :: Psig_bl,Psig_shcu
    real(kind_phys)              :: dxdh

    Psig_bl=one
    Psig_shcu=one
    dxdh=MAX(2.5_kind_phys*dx,ten)/MIN(PBLH,3000._kind_phys)
    
    
    
    
    
     
    
    Psig_bl= ((dxdh**2) + 0.106_kind_phys*(dxdh**0.667_kind_phys))/((dxdh**2) + &
         0.066_kind_phys*(dxdh**0.667_kind_phys) + 0.071_kind_phys)

    
    dxdh=MAX(2.5_kind_phys*dx,ten)/MIN(PBLH+500._kind_phys,3500._kind_phys)
    
    
    

    
    
    


    
    

    
    


    
    

    
    


    
    
    
    Psig_shcu= ((dxdh**2) + 0.145_kind_phys*(dxdh**0.667_kind_phys))/((dxdh**2) + &
         0.172_kind_phys*(dxdh**0.667_kind_phys) + 0.170_kind_phys)


    

    
    

    
    
    If(Psig_bl > one) Psig_bl=one
    If(Psig_bl < zero) Psig_bl=zero

    If(Psig_shcu > one) Psig_shcu=one
    If(Psig_shcu < zero) Psig_shcu=zero

  END SUBROUTINE SCALE_AWARE









  FUNCTION esat_blend(t) 

      use module_bl_mynnedmf_common, only: t0c,tice,one,kind_phys
      
      real(kind_phys), intent(in):: t
      real(kind_phys):: esat_blend,XC,ESL,ESI,chi
      
      real(kind_phys), parameter:: J0= .611583699E03
      real(kind_phys), parameter:: J1= .444606896E02
      real(kind_phys), parameter:: J2= .143177157E01
      real(kind_phys), parameter:: J3= .264224321E-1
      real(kind_phys), parameter:: J4= .299291081E-3
      real(kind_phys), parameter:: J5= .203154182E-5
      real(kind_phys), parameter:: J6= .702620698E-8
      real(kind_phys), parameter:: J7= .379534310E-11
      real(kind_phys), parameter:: J8=-.321582393E-13
      
      real(kind_phys), parameter:: K0= .609868993E03
      real(kind_phys), parameter:: K1= .499320233E02
      real(kind_phys), parameter:: K2= .184672631E01
      real(kind_phys), parameter:: K3= .402737184E-1
      real(kind_phys), parameter:: K4= .565392987E-3
      real(kind_phys), parameter:: K5= .521693933E-5
      real(kind_phys), parameter:: K6= .307839583E-7
      real(kind_phys), parameter:: K7= .105785160E-9
      real(kind_phys), parameter:: K8= .161444444E-12

      XC=MAX(-80._kind_phys, t - t0c) 




      IF (t .GE. (t0c-6._kind_phys)) THEN
          esat_blend = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8))))))) 
      ELSE IF (t .LE. tice) THEN
          esat_blend = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
      ELSE
          ESL = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
          ESI = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
          chi = ((t0c-6._kind_phys) - t)/((t0c-6._kind_phys) - tice)
          esat_blend = (one-chi)*ESL  + chi*ESI
      END IF

  END FUNCTION esat_blend







  FUNCTION qsat_blend(t, P)

      use module_bl_mynnedmf_common, only: t0c,tice,one,kind_phys

      real(kind_phys), intent(in):: t, P
      real(kind_phys):: qsat_blend,XC,ESL,ESI,RSLF,RSIF,chi
      
      real(kind_phys), parameter:: J0= .611583699E03
      real(kind_phys), parameter:: J1= .444606896E02
      real(kind_phys), parameter:: J2= .143177157E01
      real(kind_phys), parameter:: J3= .264224321E-1
      real(kind_phys), parameter:: J4= .299291081E-3
      real(kind_phys), parameter:: J5= .203154182E-5
      real(kind_phys), parameter:: J6= .702620698E-8
      real(kind_phys), parameter:: J7= .379534310E-11
      real(kind_phys), parameter:: J8=-.321582393E-13
      
      real(kind_phys), parameter:: K0= .609868993E03
      real(kind_phys), parameter:: K1= .499320233E02
      real(kind_phys), parameter:: K2= .184672631E01
      real(kind_phys), parameter:: K3= .402737184E-1
      real(kind_phys), parameter:: K4= .565392987E-3
      real(kind_phys), parameter:: K5= .521693933E-5
      real(kind_phys), parameter:: K6= .307839583E-7
      real(kind_phys), parameter:: K7= .105785160E-9
      real(kind_phys), parameter:: K8= .161444444E-12

      XC=MAX(-80._kind_phys,t - t0c)

      IF (t .GE. (t0c-6._kind_phys)) THEN
          ESL  = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
          ESL  = min(ESL, P*0.15_kind_phys) 
          qsat_blend = 0.622_kind_phys*ESL/max(P-ESL, 1e-5_kind_phys) 
      ELSE IF (t .LE. tice) THEN
          ESI  = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
          ESI  = min(ESI, P*0.15_kind_phys)
          qsat_blend = 0.622*ESI/max(P-ESI, 1e-5_kind_phys)
      ELSE
          ESL  = J0+XC*(J1+XC*(J2+XC*(J3+XC*(J4+XC*(J5+XC*(J6+XC*(J7+XC*J8)))))))
          ESL  = min(ESL, P*0.15_kind_phys)
          ESI  = K0+XC*(K1+XC*(K2+XC*(K3+XC*(K4+XC*(K5+XC*(K6+XC*(K7+XC*K8)))))))
          ESI  = min(ESI, P*0.15_kind_phys)
          RSLF = 0.622_kind_phys*ESL/max(P-ESL, 1e-5_kind_phys)
          RSIF = 0.622_kind_phys*ESI/max(P-ESI, 1e-5_kind_phys)

          chi  = ((t0c-6._kind_phys) - t)/((t0c-6._kind_phys) - tice) 
          qsat_blend = (one-chi)*RSLF + chi*RSIF
      END IF

  END FUNCTION qsat_blend








  FUNCTION xl_blend(t)

    use module_bl_mynnedmf_common, only: xlv,xls,cpv,cliq,cice,t0c,tice,&
         one,kind_phys

      real(kind_phys), intent(in):: t
      real(kind_phys):: xl_blend,xlvt,xlst,chi
      

      IF (t .GE. t0c) THEN
          xl_blend = xlv + (cpv-cliq)*(t-t0c)  
      ELSE IF (t .LE. tice) THEN
          xl_blend = xls + (cpv-cice)*(t-t0c)  
      ELSE
          xlvt = xlv + (cpv-cliq)*(t-t0c)  
          xlst = xls + (cpv-cice)*(t-t0c)  

          chi  = (t0c - t)/(t0c - tice)
          xl_blend = (one-chi)*xlvt + chi*xlst     
      END IF

  END FUNCTION xl_blend



  FUNCTION phim(zet)
     
     
     
     
     
     
      use module_bl_mynnedmf_common, only: cphm_unst,zero,one,two,p333,p5,p666,kind_phys

      real(kind_phys), intent(in):: zet
      real(kind_phys):: dummy_0,dummy_1,dummy_11,dummy_2,dummy_22,dummy_3,dummy_33,dummy_4,dummy_44,dummy_psi
      real(kind_phys), parameter :: am_st=6.1, bm_st=2.5, rbm_st=1./bm_st
      real(kind_phys), parameter :: ah_st=5.3, bh_st=1.1, rbh_st=1./bh_st
      real(kind_phys), parameter :: am_unst=10., ah_unst=34.
      real(kind_phys):: phi_m,phim

      if ( zet >= zero ) then
         dummy_0 = one+zet**bm_st
         dummy_1 = zet+dummy_0**(rbm_st)
         dummy_11= one+dummy_0**(rbm_st-1)*zet**(bm_st-one)
         dummy_2 = (-am_st/dummy_1)*dummy_11
         phi_m   = one-zet*dummy_2
      else
         dummy_0 = (one-cphm_unst*zet)**0.25
         phi_m   = one/dummy_0
         dummy_psi = two*log(p5*(one+dummy_0))+log(p5*(one+dummy_0**2))-two*atan(dummy_0)+1.570796

         dummy_0 = (one-am_unst*zet)          
         dummy_1 = dummy_0**p333          
         dummy_11=-p333*am_unst*dummy_0**(-p666) 
         dummy_2 = p333*(dummy_1**2+dummy_1+one)      
         dummy_22= p333*dummy_11*(two*dummy_1+one)    
         dummy_3 = 0.57735*(two*dummy_1+one)  
         dummy_33= 1.1547*dummy_11            
         dummy_4 = 1.5*log(dummy_2)-1.73205*atan(dummy_3)+1.813799364 
         dummy_44= (1.5/dummy_2)*dummy_22-1.73205*dummy_33/(one+dummy_3**2)

         dummy_0 = zet**2
         dummy_1 = one/(one+dummy_0) 
         dummy_11= two*zet         
         dummy_2 = ((one-phi_m)/zet+dummy_11*dummy_4+dummy_0*dummy_44)*dummy_1
         dummy_22= -dummy_11*(dummy_psi+dummy_0*dummy_4)*dummy_1**2

         phi_m = one-zet*(dummy_2+dummy_22)
      end if

      
      phim = phi_m

  END FUNCTION phim


  FUNCTION phih(zet)
    
    
    
    
    
    
      use module_bl_mynnedmf_common, only: cphh_unst,zero,one,two,p333,p5,p666,kind_phys

      real(kind_phys), intent(in):: zet
      real(kind_phys):: dummy_0,dummy_1,dummy_11,dummy_2,dummy_22,dummy_3,dummy_33,dummy_4,dummy_44,dummy_psi
      real(kind_phys), parameter :: am_st=6.1, bm_st=2.5, rbm_st=1./bm_st
      real(kind_phys), parameter :: ah_st=5.3, bh_st=1.1, rbh_st=1./bh_st
      real(kind_phys), parameter :: am_unst=10., ah_unst=34.
      real(kind_phys):: phh,phih

      if ( zet >= zero ) then
         dummy_0 = one+zet**bh_st
         dummy_1 = zet+dummy_0**(rbh_st)
         dummy_11= one+dummy_0**(rbh_st-one)*zet**(bh_st-one)
         dummy_2 = (-ah_st/dummy_1)*dummy_11
         phih    = one-zet*dummy_2
      else
         dummy_0 = (one-cphh_unst*zet)**p5
         phh     = one/dummy_0
         dummy_psi = two*log(p5*(one+dummy_0))

         dummy_0 = (one-ah_unst*zet)         
         dummy_1 = dummy_0**p333         
         dummy_11=-p333*ah_unst*dummy_0**(-p666) 
         dummy_2 = p333*(dummy_1**2+dummy_1+one)      
         dummy_22= p333*dummy_11*(two*dummy_1+one)    
         dummy_3 = 0.57735*(two*dummy_1+one) 
         dummy_33= 1.1547*dummy_11           
         dummy_4 = 1.5*log(dummy_2)-1.73205*atan(dummy_3)+1.813799364 
         dummy_44= (1.5/dummy_2)*dummy_22-1.73205*dummy_33/(one+dummy_3**2)

         dummy_0 = zet**2
         dummy_1 = one/(one+dummy_0)         
         dummy_11= two*zet                   
         dummy_2 = ((one-phh)/zet+dummy_11*dummy_4+dummy_0*dummy_44)*dummy_1
         dummy_22= -dummy_11*(dummy_psi+dummy_0*dummy_4)*dummy_1**2

         phih = one-zet*(dummy_2+dummy_22)
      end if

end function phih

 subroutine topdown_cloudrad(kts,kte,                         &
               &dz1,zw,fltv,u1,v1,xland,kpbl,pblh,            &
               &sqc,sqi,sqw,thl,th1,ex1,pres1,rho1,thv,       &
               &cldfra_bl1,qc_bl,qi_bl,rthraten,              &
               &tkeprod_dn,psig,                              &
               &maxtkeprod,cldtop_cooling,ent_eff)

   use module_bl_mynnedmf_common, only: cp,ep_2,xlv,r_d,p608, &
        xlvcp,grav,zero,one,two,three,eight,p1,p2,p25,p3,     &
        p333,p5,kind_phys
   
    
    integer,         intent(in) :: kte,kts
    real(kind_phys), dimension(kts:kte), intent(in) :: dz1,sqc,sqi,sqw,&
          thl,th1,ex1,pres1,rho1,thv,cldfra_bl1,qc_bl,qi_bl
    real(kind_phys), dimension(kts:kte), intent(in) :: rthraten
    real(kind_phys), dimension(kts:kte+1), intent(in) :: zw
    real(kind_phys), intent(in) :: pblh,fltv,u1,v1,psig
    real(kind_phys), intent(in) :: xland
    integer        , intent(in) :: kpbl
    
    real(kind_phys), dimension(kts:kte), intent(inout) :: tkeprod_dn
    real(kind_phys), intent(inout) :: maxtkeprod
    real(kind_phys), intent(inout) :: cldtop_cooling
    real(kind_phys), intent(inout) :: ent_eff
    
    real(kind_phys),parameter:: dz500   = 500.  
    real(kind_phys),parameter:: dz200   = 200.  
    real(kind_phys),parameter:: cooling_thresh = -0.000069  
                                                
                                                
                                                
    real(kind_phys), dimension(kts:kte) :: zfac,zfacent,tkeprodorig
    real(kind_phys) :: dthvx,tmp1,zfacent_max,zagl
    real(kind_phys) :: temps,templ,zl1,zb1,dz_ent,wstar_rad,maxcooling
    real(kind_phys) :: f0,radflux,went,rcldb,rvls,minrad,zminrad,dp
    real(kind_phys) :: wspd_pbl,ac_wsp
    real(kind_phys), parameter :: pfac =2.0, zfmin = 0.01, phifac=8.0
    integer :: k,kk,kminrad
    logical :: cloudflg
    logical,parameter:: debug=.false.

    cloudflg       = .false.
    minrad         = zero
    kminrad        = kpbl
    zminrad        = pblh
    
    tkeprodorig(kts:kte)=tkeprod_dn(kts:kte)
    
    
    do k = 1,min(kpbl+10,kte-1)
       maxcooling = minval(rthraten(max(kts,k-1):k+1))
       
       if ((cldfra_bl1(k).gt.p5 .and. cldfra_bl1(k+1).lt.p5) .and. &  
           (maxcooling  .lt. cooling_thresh)) then                    
          
          cloudflg =.true.
          if (maxcooling < minrad) then
             minrad    =maxcooling      
             kminrad  = k
             zminrad  = zw(k) + p5*dz1(k) 
          endif
       endif
    enddo

    if (cloudflg) then
       zl1     = p25*dz1(kts)
       k       = kminrad

       templ   = thl(k)*ex1(k)
       
       rvls    = 100.*6.112*EXP(17.67*(templ-273.16)/(templ-29.65))*(ep_2/pres1(k+1))
       temps   = templ + (sqw(k)-rvls)/(cp/xlv  +  ep_2*xlv*rvls/(r_d*templ**2))
       rvls    = 100.*6.112*EXP(17.67*(temps-273.15)/(temps-29.65))*(ep_2/pres1(k+1))
       rcldb   = max(sqw(k)-rvls,zero)         
       rcldb   = max(rcldb, qc_bl(k)+qi_bl(k)) 

       
       dthvx   = (thl(k+2) + th1(k+2)*p608*sqw(k+2)) - &
                 (thl(k)   + th1(k)  *p608*sqw(k))
       
       dz_ent  = (dz1(k  )*dz1(k+1)+dz1(k+1)*dz1(k  ))/(dz1(k+1)+dz1(k)) + &
                 (dz1(k+1)*dz1(k+2)+dz1(k+2)*dz1(k+1))/(dz1(k+1)+dz1(k+2)) 
       dthvx   = dthvx * dz200/dz_ent  
       dthvx   = max(dthvx, p1)
       tmp1    = xlvcp * rcldb/(ex1(k)*dthvx)
       
       
       ent_eff = min(two, p2 + p2*eight*tmp1)

       f0      = zero
       do kk = max(kts,kminrad-1),kminrad+1
          radflux = rthraten(kk)*ex1(kk)             
          dp      = rho1(k)*grav*dz1(k)
          radflux = radflux * cp / grav * dp         
          if (radflux < zero ) then
             f0   = abs(radflux)+f0                  
          endif
       enddo

       
       if ((xland-1.5).ge.zero) then     
          f0     = min(f0, 150.0_kind_phys)
       else                              
          f0     = min(p25*f0, 30._kind_phys)
       endif
       cldtop_cooling = abs(minrad) * 3600.   
       
       
       wstar_rad = 1.25_kind_phys * ( grav/thv(k) * dz200 * f0 / (rho1(k) * cp) ) **p333
       wstar_rad = min(max(wstar_rad, p1), three)
       dthvx     = max(thv(k+1)-thv(k), p1)
       went      = p333*ent_eff*wstar_rad     

       
       
       zb1       = max(zl1, zminrad - wstar_rad*800._kind_phys)
       do k = kts,kminrad+3
          
          zagl       = zw(k) + p5*dz1(k)
          if (k==kts .and. kminrad==kts) zagl = 0.4_kind_phys * dz1(kts)
          
          zfac(k)    = min(max((one - max(zero,zagl-zb1)/(zminrad-zl1)), zfmin), one)
          zfacent(k) = max((zminrad-zagl)/zminrad, zero)*(one-zfac(k))**3
       enddo
       zfacent_max = maxval(zfacent(kts:kminrad+3))
       zfacent     = zfacent/max(zfacent_max, 1e-5_kind_phys)  

       
       do k = kts,kminrad+3
          tkeprod_dn(k) = max(tkeprod_dn(k), p3*ent_eff*wstar_rad**3/dz500*zfacent(k))
          tkeprod_dn(k) = max(tkeprod_dn(k), zero)
          tkeprod_dn(k) = min(0.0004_kind_phys, tkeprod_dn(k))
       enddo
       
       tkeprod_dn(kminrad)=max(tkeprod_dn(kminrad), p333*tkeprod_dn(max(kts,kminrad-1)))
       
       tkeprod_dn(kminrad+1)=max(tkeprod_dn(kminrad+1), min(p5,ent_eff)*tkeprod_dn(max(kts,kminrad)))
       
       
       wspd_pbl   = sqrt(max(u1**2 + v1**2, 0.01_kind_phys))
       ac_wsp     = one - min(max(zero, wspd_pbl - 10._kind_phys)/15._kind_phys, one)
       tkeprod_dn = tkeprod_dn * min(ac_wsp, psig)
    endif 

    
    maxtkeprod = maxval(tkeprod_dn)*3600.
    
    if (cloudflg .and. debug) then
       
       print *, '-----------------------------------------------------------------'
       if ((xland-1.5).ge.zero) then
          print *, ' over water, kminrad=',kminrad
       else
          print *, ' over land, kminrad=',kminrad
       endif
       print *, ' ctop cooling (K/hr)=',minrad*3600.,' rad cooling (W/m^2)=',f0
       print *, ' ent_eff=',ent_eff,' went=',went,' w*rad=',wstar_rad
       print *, '  k    cldfra   dT/dt(hr)  dTKE/dt_orig dTKE/dt(hr)  zfacent'
       print *, '-----------------------------------------------------------------'
       do k = kminrad+3,kts,-1
          print '(I4,5F11.5)', k, cldfra_bl1(k), rthraten(k)*3600., tkeprodorig(k)*3600., tkeprod_dn(k)*3600.,  zfacent(k) 
       enddo
    endif
    
 END SUBROUTINE topdown_cloudrad




END MODULE module_bl_mynnedmf
