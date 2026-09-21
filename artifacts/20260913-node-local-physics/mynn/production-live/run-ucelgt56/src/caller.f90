program mynn_bridge_driver
  use, intrinsic :: ieee_arithmetic
  use ccpp_kind_types, only: kind_phys
  use module_bl_mynnedmf_driver, only: mynnedmf_driver
  implicit none
  integer :: unit, status
  character(len=512) :: output
  real(kind_phys) :: rublten_before(1:4,1:33,1:6)
  integer, parameter :: ids = 1
  integer, parameter :: ide = 4
  integer, parameter :: jds = 1
  integer, parameter :: jde = 6
  integer, parameter :: kds = 1
  integer, parameter :: kde = 33
  integer, parameter :: ims = 1
  integer, parameter :: ime = 4
  integer, parameter :: jms = 1
  integer, parameter :: jme = 6
  integer, parameter :: kms = 1
  integer, parameter :: kme = 33
  integer, parameter :: its = 2
  integer, parameter :: ite = 3
  integer, parameter :: jts = 2
  integer, parameter :: jte = 5
  integer, parameter :: kts = 1
  integer, parameter :: kte = 32
  logical :: flag_qc
  logical :: flag_qi
  logical :: flag_qs
  logical :: flag_qnc
  logical :: flag_qni
  logical :: flag_qnifa
  logical :: flag_qnwfa
  logical :: flag_qnbca
  integer :: initflag
  logical :: restart
  logical :: cycling
  real(kind_phys) :: delt
  real(kind_phys) :: dxc
  real(kind_phys) :: xland(ims:ime,jms:jme)
  real(kind_phys) :: ps(ims:ime,jms:jme)
  real(kind_phys) :: ts(ims:ime,jms:jme)
  real(kind_phys) :: qsfc(ims:ime,jms:jme)
  real(kind_phys) :: ust(ims:ime,jms:jme)
  real(kind_phys) :: ch(ims:ime,jms:jme)
  real(kind_phys) :: hfx(ims:ime,jms:jme)
  real(kind_phys) :: qfx(ims:ime,jms:jme)
  real(kind_phys) :: wspd(ims:ime,jms:jme)
  real(kind_phys) :: znt(ims:ime,jms:jme)
  real(kind_phys) :: uoce(ims:ime,jms:jme)
  real(kind_phys) :: voce(ims:ime,jms:jme)
  real(kind_phys) :: dz(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: u(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: v(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: w(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: th(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: t3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: p(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: exner(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rho(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qv(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qc(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qi(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qs(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qnc(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qni(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qnifa(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qnwfa(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qnbca(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rthraten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: pblh(ims:ime,jms:jme)
  integer :: kpbl(ims:ime,jms:jme)
  real(kind_phys) :: maxwidth_dd(ims:ime,jms:jme)
  real(kind_phys) :: cldfra_bl(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qc_bl(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qi_bl(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: maxwidth(ims:ime,jms:jme)
  real(kind_phys) :: maxmf(ims:ime,jms:jme)
  real(kind_phys) :: ztop_plume(ims:ime,jms:jme)
  real(kind_phys) :: excess_h(ims:ime,jms:jme)
  real(kind_phys) :: excess_q(ims:ime,jms:jme)
  real(kind_phys) :: maxmf_dd(ims:ime,jms:jme)
  real(kind_phys) :: maxtkeprod(ims:ime,jms:jme)
  real(kind_phys) :: cldtop_cooling(ims:ime,jms:jme)
  real(kind_phys) :: ent_eff(ims:ime,jms:jme)
  real(kind_phys) :: qke(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qke_adv(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: tsq(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qsq(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: cov(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: el_pbl(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rublten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rvblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rthblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqvblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqcblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqiblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqsblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqncblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqniblten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqnifablten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqnwfablten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: rqnbcablten(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_a(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_w(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_qt(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_thl(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_ent(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: edmf_qc(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: sub_thl3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: sub_sqv3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: det_thl3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: det_sqv3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: exch_h(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: exch_m(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: dqke(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qwt(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qshear(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qbuoy(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: qdiss(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: sh3d(ims:ime,kms:kme,jms:jme)
  real(kind_phys) :: sm3d(ims:ime,kms:kme,jms:jme)
  integer :: spp_pbl
  real(kind_phys) :: pattern_spp_pbl(ims:ime,kms:kme,jms:jme)
  logical :: bl_mynn_tkeadvect
  integer :: tke_budget
  integer :: bl_mynn_cloudpdf
  integer :: bl_mynn_mixlength
  real(kind_phys) :: bl_mynn_closure
  integer :: bl_mynn_edmf
  integer :: bl_mynn_edmf_mom
  integer :: bl_mynn_edmf_tke
  integer :: bl_mynn_output
  integer :: bl_mynn_mixscalars
  integer :: bl_mynn_mixaerosols
  integer :: bl_mynn_mixnumcon
  integer :: bl_mynn_cloudmix
  integer :: bl_mynn_mixqt
  integer :: bl_mynn_edmf_dd
  integer :: bl_mynn_ess
  call get_command_argument(1, output, status=status)
  if (status /= 0 .or. len_trim(output) == 0) error stop "need output path"
  flag_qc = .false.
  flag_qi = .false.
  flag_qs = .false.
  flag_qnc = .false.
  flag_qni = .false.
  flag_qnifa = .false.
  flag_qnwfa = .false.
  flag_qnbca = .false.
  initflag = 1
  restart = .false.
  cycling = .false.
  delt = 30.0_kind_phys
  dxc = 1000.0_kind_phys
  spp_pbl = 0
  bl_mynn_tkeadvect = .false.
  tke_budget = 0
  bl_mynn_cloudpdf = 0
  bl_mynn_mixlength = 0
  bl_mynn_closure = 2.5_kind_phys
  bl_mynn_edmf = 0
  bl_mynn_edmf_mom = 0
  bl_mynn_edmf_tke = 0
  bl_mynn_output = 0
  bl_mynn_mixscalars = 0
  bl_mynn_mixaerosols = 0
  bl_mynn_mixnumcon = 0
  bl_mynn_cloudmix = 0
  bl_mynn_mixqt = 0
  bl_mynn_edmf_dd = 0
  bl_mynn_ess = 0
  xland = 1.0_kind_phys
  ps = 100000.0_kind_phys
  ts = 290.0_kind_phys
  qsfc = 0.010_kind_phys
  ust = 0.30_kind_phys
  ch = 0.010_kind_phys
  hfx = 30.0_kind_phys
  qfx = 0.0_kind_phys
  wspd = 5.0_kind_phys
  znt = 0.10_kind_phys
  uoce = 0.0_kind_phys
  voce = 0.0_kind_phys
  dz = 100.0_kind_phys
  u = 5.0_kind_phys
  v = 1.0_kind_phys
  w = 0.0_kind_phys
  th = 300.0_kind_phys
  t3d = 290.0_kind_phys
  p = 90000.0_kind_phys
  exner = 0.97_kind_phys
  rho = 1.0_kind_phys
  qv = 0.010_kind_phys
  qc = 0.0_kind_phys
  qi = 0.0_kind_phys
  qs = 0.0_kind_phys
  qnc = 0.0_kind_phys
  qni = 0.0_kind_phys
  qnifa = 0.0_kind_phys
  qnwfa = 0.0_kind_phys
  qnbca = 0.0_kind_phys
  rthraten = 0.0_kind_phys
  pblh = 100.0_kind_phys
  kpbl = 1
  maxwidth_dd = 0.0_kind_phys
  cldfra_bl = 0.0_kind_phys
  qc_bl = 0.0_kind_phys
  qi_bl = 0.0_kind_phys
  maxwidth = 0.0_kind_phys
  maxmf = 0.0_kind_phys
  ztop_plume = 0.0_kind_phys
  excess_h = 0.0_kind_phys
  excess_q = 0.0_kind_phys
  maxmf_dd = 0.0_kind_phys
  maxtkeprod = 0.0_kind_phys
  cldtop_cooling = 0.0_kind_phys
  ent_eff = 0.0_kind_phys
  qke = 0.10_kind_phys
  qke_adv = 0.0_kind_phys
  tsq = 0.0_kind_phys
  qsq = 0.0_kind_phys
  cov = 0.0_kind_phys
  el_pbl = 0.0_kind_phys
  rublten = 0.0_kind_phys
  rvblten = 0.0_kind_phys
  rthblten = 0.0_kind_phys
  rqvblten = 0.0_kind_phys
  rqcblten = 0.0_kind_phys
  rqiblten = 0.0_kind_phys
  rqsblten = 0.0_kind_phys
  rqncblten = 0.0_kind_phys
  rqniblten = 0.0_kind_phys
  rqnifablten = 0.0_kind_phys
  rqnwfablten = 0.0_kind_phys
  rqnbcablten = 0.0_kind_phys
  edmf_a = 0.0_kind_phys
  edmf_w = 0.0_kind_phys
  edmf_qt = 0.0_kind_phys
  edmf_thl = 0.0_kind_phys
  edmf_ent = 0.0_kind_phys
  edmf_qc = 0.0_kind_phys
  sub_thl3d = 0.0_kind_phys
  sub_sqv3d = 0.0_kind_phys
  det_thl3d = 0.0_kind_phys
  det_sqv3d = 0.0_kind_phys
  exch_h = 0.0_kind_phys
  exch_m = 0.0_kind_phys
  dqke = 0.0_kind_phys
  qwt = 0.0_kind_phys
  qshear = 0.0_kind_phys
  qbuoy = 0.0_kind_phys
  qdiss = 0.0_kind_phys
  sh3d = 0.0_kind_phys
  sm3d = 0.0_kind_phys
  pattern_spp_pbl = 0.0_kind_phys
  rublten_before = rublten
  call mynnedmf_driver(ids=ids, &
       ide=ide, &
       jds=jds, &
       jde=jde, &
       kds=kds, &
       kde=kde, &
       ims=ims, &
       ime=ime, &
       jms=jms, &
       jme=jme, &
       kms=kms, &
       kme=kme, &
       its=its, &
       ite=ite, &
       jts=jts, &
       jte=jte, &
       kts=kts, &
       kte=kte, &
       flag_qc=flag_qc, &
       flag_qi=flag_qi, &
       flag_qs=flag_qs, &
       flag_qnc=flag_qnc, &
       flag_qni=flag_qni, &
       flag_qnifa=flag_qnifa, &
       flag_qnwfa=flag_qnwfa, &
       flag_qnbca=flag_qnbca, &
       initflag=initflag, &
       restart=restart, &
       cycling=cycling, &
       delt=delt, &
       dxc=dxc, &
       xland=xland, &
       ps=ps, &
       ts=ts, &
       qsfc=qsfc, &
       ust=ust, &
       ch=ch, &
       hfx=hfx, &
       qfx=qfx, &
       wspd=wspd, &
       znt=znt, &
       uoce=uoce, &
       voce=voce, &
       dz=dz, &
       u=u, &
       v=v, &
       w=w, &
       th=th, &
       t3d=t3d, &
       p=p, &
       exner=exner, &
       rho=rho, &
       qv=qv, &
       qc=qc, &
       qi=qi, &
       qs=qs, &
       qnc=qnc, &
       qni=qni, &
       qnifa=qnifa, &
       qnwfa=qnwfa, &
       qnbca=qnbca, &
       rthraten=rthraten, &
       pblh=pblh, &
       kpbl=kpbl, &
       maxwidth_dd=maxwidth_dd, &
       cldfra_bl=cldfra_bl, &
       qc_bl=qc_bl, &
       qi_bl=qi_bl, &
       maxwidth=maxwidth, &
       maxmf=maxmf, &
       ztop_plume=ztop_plume, &
       excess_h=excess_h, &
       excess_q=excess_q, &
       maxmf_dd=maxmf_dd, &
       maxtkeprod=maxtkeprod, &
       cldtop_cooling=cldtop_cooling, &
       ent_eff=ent_eff, &
       qke=qke, &
       qke_adv=qke_adv, &
       tsq=tsq, &
       qsq=qsq, &
       cov=cov, &
       el_pbl=el_pbl, &
       rublten=rublten, &
       rvblten=rvblten, &
       rthblten=rthblten, &
       rqvblten=rqvblten, &
       rqcblten=rqcblten, &
       rqiblten=rqiblten, &
       rqsblten=rqsblten, &
       rqncblten=rqncblten, &
       rqniblten=rqniblten, &
       rqnifablten=rqnifablten, &
       rqnwfablten=rqnwfablten, &
       rqnbcablten=rqnbcablten, &
       edmf_a=edmf_a, &
       edmf_w=edmf_w, &
       edmf_qt=edmf_qt, &
       edmf_thl=edmf_thl, &
       edmf_ent=edmf_ent, &
       edmf_qc=edmf_qc, &
       sub_thl3d=sub_thl3d, &
       sub_sqv3d=sub_sqv3d, &
       det_thl3d=det_thl3d, &
       det_sqv3d=det_sqv3d, &
       exch_h=exch_h, &
       exch_m=exch_m, &
       dqke=dqke, &
       qwt=qwt, &
       qshear=qshear, &
       qbuoy=qbuoy, &
       qdiss=qdiss, &
       sh3d=sh3d, &
       sm3d=sm3d, &
       spp_pbl=spp_pbl, &
       pattern_spp_pbl=pattern_spp_pbl, &
       bl_mynn_tkeadvect=bl_mynn_tkeadvect, &
       tke_budget=tke_budget, &
       bl_mynn_cloudpdf=bl_mynn_cloudpdf, &
       bl_mynn_mixlength=bl_mynn_mixlength, &
       bl_mynn_closure=bl_mynn_closure, &
       bl_mynn_edmf=bl_mynn_edmf, &
       bl_mynn_edmf_mom=bl_mynn_edmf_mom, &
       bl_mynn_edmf_tke=bl_mynn_edmf_tke, &
       bl_mynn_output=bl_mynn_output, &
       bl_mynn_mixscalars=bl_mynn_mixscalars, &
       bl_mynn_mixaerosols=bl_mynn_mixaerosols, &
       bl_mynn_mixnumcon=bl_mynn_mixnumcon, &
       bl_mynn_cloudmix=bl_mynn_cloudmix, &
       bl_mynn_mixqt=bl_mynn_mixqt, &
       bl_mynn_edmf_dd=bl_mynn_edmf_dd, &
       bl_mynn_ess=bl_mynn_ess)
  if (all(rublten(2:3,1:32,2:3) == rublten_before(2:3,1:32,2:3))) error stop "MYNN made no tendency change"
  if (any(rublten(1,:,:) /= rublten_before(1,:,:))) error stop "tile i-low sentinel changed"
  if (any(rublten(4,:,:) /= rublten_before(4,:,:))) error stop "tile i-high sentinel changed"
  if (any(rublten(:,:,1) /= rublten_before(:,:,1))) error stop "tile j-low sentinel changed"
  if (any(rublten(:,:,6) /= rublten_before(:,:,6))) error stop "tile j-high sentinel changed"
  if (.not. all(ieee_is_finite(xland))) error stop "nonfinite xland"
  if (.not. all(ieee_is_finite(ps))) error stop "nonfinite ps"
  if (.not. all(ieee_is_finite(ts))) error stop "nonfinite ts"
  if (.not. all(ieee_is_finite(qsfc))) error stop "nonfinite qsfc"
  if (.not. all(ieee_is_finite(ust))) error stop "nonfinite ust"
  if (.not. all(ieee_is_finite(ch))) error stop "nonfinite ch"
  if (.not. all(ieee_is_finite(hfx))) error stop "nonfinite hfx"
  if (.not. all(ieee_is_finite(qfx))) error stop "nonfinite qfx"
  if (.not. all(ieee_is_finite(wspd))) error stop "nonfinite wspd"
  if (.not. all(ieee_is_finite(znt))) error stop "nonfinite znt"
  if (.not. all(ieee_is_finite(uoce))) error stop "nonfinite uoce"
  if (.not. all(ieee_is_finite(voce))) error stop "nonfinite voce"
  if (.not. all(ieee_is_finite(dz))) error stop "nonfinite dz"
  if (.not. all(ieee_is_finite(u))) error stop "nonfinite u"
  if (.not. all(ieee_is_finite(v))) error stop "nonfinite v"
  if (.not. all(ieee_is_finite(w))) error stop "nonfinite w"
  if (.not. all(ieee_is_finite(th))) error stop "nonfinite th"
  if (.not. all(ieee_is_finite(t3d))) error stop "nonfinite t3d"
  if (.not. all(ieee_is_finite(p))) error stop "nonfinite p"
  if (.not. all(ieee_is_finite(exner))) error stop "nonfinite exner"
  if (.not. all(ieee_is_finite(rho))) error stop "nonfinite rho"
  if (.not. all(ieee_is_finite(qv))) error stop "nonfinite qv"
  if (.not. all(ieee_is_finite(qc))) error stop "nonfinite qc"
  if (.not. all(ieee_is_finite(qi))) error stop "nonfinite qi"
  if (.not. all(ieee_is_finite(qs))) error stop "nonfinite qs"
  if (.not. all(ieee_is_finite(qnc))) error stop "nonfinite qnc"
  if (.not. all(ieee_is_finite(qni))) error stop "nonfinite qni"
  if (.not. all(ieee_is_finite(qnifa))) error stop "nonfinite qnifa"
  if (.not. all(ieee_is_finite(qnwfa))) error stop "nonfinite qnwfa"
  if (.not. all(ieee_is_finite(qnbca))) error stop "nonfinite qnbca"
  if (.not. all(ieee_is_finite(rthraten))) error stop "nonfinite rthraten"
  if (.not. all(ieee_is_finite(pblh))) error stop "nonfinite pblh"
  if (.not. all(ieee_is_finite(maxwidth_dd))) error stop "nonfinite maxwidth_dd"
  if (.not. all(ieee_is_finite(cldfra_bl))) error stop "nonfinite cldfra_bl"
  if (.not. all(ieee_is_finite(qc_bl))) error stop "nonfinite qc_bl"
  if (.not. all(ieee_is_finite(qi_bl))) error stop "nonfinite qi_bl"
  if (.not. all(ieee_is_finite(maxwidth))) error stop "nonfinite maxwidth"
  if (.not. all(ieee_is_finite(maxmf))) error stop "nonfinite maxmf"
  if (.not. all(ieee_is_finite(ztop_plume))) error stop "nonfinite ztop_plume"
  if (.not. all(ieee_is_finite(excess_h))) error stop "nonfinite excess_h"
  if (.not. all(ieee_is_finite(excess_q))) error stop "nonfinite excess_q"
  if (.not. all(ieee_is_finite(maxmf_dd))) error stop "nonfinite maxmf_dd"
  if (.not. all(ieee_is_finite(maxtkeprod))) error stop "nonfinite maxtkeprod"
  if (.not. all(ieee_is_finite(cldtop_cooling))) error stop "nonfinite cldtop_cooling"
  if (.not. all(ieee_is_finite(ent_eff))) error stop "nonfinite ent_eff"
  if (.not. all(ieee_is_finite(qke))) error stop "nonfinite qke"
  if (.not. all(ieee_is_finite(qke_adv))) error stop "nonfinite qke_adv"
  if (.not. all(ieee_is_finite(tsq))) error stop "nonfinite tsq"
  if (.not. all(ieee_is_finite(qsq))) error stop "nonfinite qsq"
  if (.not. all(ieee_is_finite(cov))) error stop "nonfinite cov"
  if (.not. all(ieee_is_finite(el_pbl))) error stop "nonfinite el_pbl"
  if (.not. all(ieee_is_finite(rublten))) error stop "nonfinite rublten"
  if (.not. all(ieee_is_finite(rvblten))) error stop "nonfinite rvblten"
  if (.not. all(ieee_is_finite(rthblten))) error stop "nonfinite rthblten"
  if (.not. all(ieee_is_finite(rqvblten))) error stop "nonfinite rqvblten"
  if (.not. all(ieee_is_finite(rqcblten))) error stop "nonfinite rqcblten"
  if (.not. all(ieee_is_finite(rqiblten))) error stop "nonfinite rqiblten"
  if (.not. all(ieee_is_finite(rqsblten))) error stop "nonfinite rqsblten"
  if (.not. all(ieee_is_finite(rqncblten))) error stop "nonfinite rqncblten"
  if (.not. all(ieee_is_finite(rqniblten))) error stop "nonfinite rqniblten"
  if (.not. all(ieee_is_finite(rqnifablten))) error stop "nonfinite rqnifablten"
  if (.not. all(ieee_is_finite(rqnwfablten))) error stop "nonfinite rqnwfablten"
  if (.not. all(ieee_is_finite(rqnbcablten))) error stop "nonfinite rqnbcablten"
  if (.not. all(ieee_is_finite(edmf_a))) error stop "nonfinite edmf_a"
  if (.not. all(ieee_is_finite(edmf_w))) error stop "nonfinite edmf_w"
  if (.not. all(ieee_is_finite(edmf_qt))) error stop "nonfinite edmf_qt"
  if (.not. all(ieee_is_finite(edmf_thl))) error stop "nonfinite edmf_thl"
  if (.not. all(ieee_is_finite(edmf_ent))) error stop "nonfinite edmf_ent"
  if (.not. all(ieee_is_finite(edmf_qc))) error stop "nonfinite edmf_qc"
  if (.not. all(ieee_is_finite(sub_thl3d))) error stop "nonfinite sub_thl3d"
  if (.not. all(ieee_is_finite(sub_sqv3d))) error stop "nonfinite sub_sqv3d"
  if (.not. all(ieee_is_finite(det_thl3d))) error stop "nonfinite det_thl3d"
  if (.not. all(ieee_is_finite(det_sqv3d))) error stop "nonfinite det_sqv3d"
  if (.not. all(ieee_is_finite(exch_h))) error stop "nonfinite exch_h"
  if (.not. all(ieee_is_finite(exch_m))) error stop "nonfinite exch_m"
  if (.not. all(ieee_is_finite(dqke))) error stop "nonfinite dqke"
  if (.not. all(ieee_is_finite(qwt))) error stop "nonfinite qwt"
  if (.not. all(ieee_is_finite(qshear))) error stop "nonfinite qshear"
  if (.not. all(ieee_is_finite(qbuoy))) error stop "nonfinite qbuoy"
  if (.not. all(ieee_is_finite(qdiss))) error stop "nonfinite qdiss"
  if (.not. all(ieee_is_finite(sh3d))) error stop "nonfinite sh3d"
  if (.not. all(ieee_is_finite(sm3d))) error stop "nonfinite sm3d"
  if (.not. all(ieee_is_finite(pattern_spp_pbl))) error stop "nonfinite pattern_spp_pbl"
  open(newunit=unit, file=trim(output), status="replace", access="stream", form="unformatted", action="write")
  write(unit) ids, &
       ide, &
       jds, &
       jde, &
       kds, &
       kde, &
       ims, &
       ime, &
       jms, &
       jme, &
       kms, &
       kme, &
       its, &
       ite, &
       jts, &
       jte, &
       kts, &
       kte, &
       flag_qc, &
       flag_qi, &
       flag_qs, &
       flag_qnc, &
       flag_qni, &
       flag_qnifa, &
       flag_qnwfa, &
       flag_qnbca, &
       initflag, &
       restart, &
       cycling, &
       delt, &
       dxc, &
       xland, &
       ps, &
       ts, &
       qsfc, &
       ust, &
       ch, &
       hfx, &
       qfx, &
       wspd, &
       znt, &
       uoce, &
       voce, &
       dz, &
       u, &
       v, &
       w, &
       th, &
       t3d, &
       p, &
       exner, &
       rho, &
       qv, &
       qc, &
       qi, &
       qs, &
       qnc, &
       qni, &
       qnifa, &
       qnwfa, &
       qnbca, &
       rthraten, &
       pblh, &
       kpbl, &
       maxwidth_dd, &
       cldfra_bl, &
       qc_bl, &
       qi_bl, &
       maxwidth, &
       maxmf, &
       ztop_plume, &
       excess_h, &
       excess_q, &
       maxmf_dd, &
       maxtkeprod, &
       cldtop_cooling, &
       ent_eff, &
       qke, &
       qke_adv, &
       tsq, &
       qsq, &
       cov, &
       el_pbl, &
       rublten, &
       rvblten, &
       rthblten, &
       rqvblten, &
       rqcblten, &
       rqiblten, &
       rqsblten, &
       rqncblten, &
       rqniblten, &
       rqnifablten, &
       rqnwfablten, &
       rqnbcablten, &
       edmf_a, &
       edmf_w, &
       edmf_qt, &
       edmf_thl, &
       edmf_ent, &
       edmf_qc, &
       sub_thl3d, &
       sub_sqv3d, &
       det_thl3d, &
       det_sqv3d, &
       exch_h, &
       exch_m, &
       dqke, &
       qwt, &
       qshear, &
       qbuoy, &
       qdiss, &
       sh3d, &
       sm3d, &
       spp_pbl, &
       pattern_spp_pbl, &
       bl_mynn_tkeadvect, &
       tke_budget, &
       bl_mynn_cloudpdf, &
       bl_mynn_mixlength, &
       bl_mynn_closure, &
       bl_mynn_edmf, &
       bl_mynn_edmf_mom, &
       bl_mynn_edmf_tke, &
       bl_mynn_output, &
       bl_mynn_mixscalars, &
       bl_mynn_mixaerosols, &
       bl_mynn_mixnumcon, &
       bl_mynn_cloudmix, &
       bl_mynn_mixqt, &
       bl_mynn_edmf_dd, &
       bl_mynn_ess
  close(unit)
end program mynn_bridge_driver
