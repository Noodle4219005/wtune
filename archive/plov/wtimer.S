/* wtimer -- per-rank, per-call timing of WRF's physics drivers with zero source change.
 *
 * WHY ASSEMBLY.  A C wrapper would have to redeclare the driver's full argument list --
 * microphysics_driver alone takes 421 arguments with 55 OPTIONAL -- and getting the count
 * wrong silently corrupts the stack frame handed to __real_.  This trampoline never touches
 * the stack layout at all: it jmp's to __real_ with rsp exactly as the caller built it, so
 * the argument count is irrelevant and cannot be got wrong.
 *
 * HOW.  On entry we stamp the TSC, swap the return address on the stack for our own label,
 * and jump.  When __real_ returns it lands on our label, where we stamp again and jump to
 * the caller's real return address.
 *
 * ALIGNMENT.  At __wrap_X entry rsp%16 == 8 (the caller's CALL pushed 8).  We do not move
 * rsp, so __real_X sees the same, which is what its own entry expects.  Its RET pops 8, so
 * at label 9 rsp%16 == 0 -- already aligned for a CALL, which is why the epilogue may call
 * into C freely.  Argument registers are dead by then; only rax/rdx are preserved, and a
 * Fortran SUBROUTINE returns nothing anyway.
 *
 * RE-ENTRANCY.  Each symbol owns its own t0/ret slot, so distinct drivers never collide.
 * The same driver is never nested (no recursion, and the call sites sit outside every
 * OpenMP region -- verified at dyn_em/solve_em.F:3881, nearest !$OMP is the END PARALLEL DO
 * at 3848).  The production tree is dmpar, so there are no threads at all.
 */
    .text

/* \idx selects the per-symbol slot; \name is the Intel-mangled Fortran symbol. */
.macro WTWRAP name, idx
    .globl  __wrap_\name
    .type   __wrap_\name, @function
__wrap_\name:
    movq    %rax, wt_sav_rax(%rip)          /* rdtsc clobbers rax:rdx, and rdx is the   */
    movq    %rdx, wt_sav_rdx(%rip)          /* 3rd integer argument -- must come back   */
    rdtsc
    shlq    $32, %rdx
    orq     %rdx, %rax
    movq    %rax, wt_t0+8*\idx(%rip)
    movq    wt_sav_rdx(%rip), %rdx
    movq    wt_sav_rax(%rip), %rax
    movq    (%rsp), %r11                    /* r11 is caller-saved and never an argument */
    movq    %r11, wt_ret+8*\idx(%rip)
    leaq    900\idx\()f(%rip), %r11
    movq    %r11, (%rsp)                    /* hook the return, stack args untouched     */
    /* Call wt_pre(idx) before handing over. Intel Fortran passes EVERY argument by
     * reference, so all 421 of them are addresses -> INTEGER class -> rdi,rsi,rdx,rcx,r8,r9
     * and then the stack. No xmm register ever holds an argument, so six saves suffice.
     * rsp is moved by 8 only for the duration of the call (entry rsp%16==8, a CALL needs 0)
     * and restored before the jmp, so the stack arguments stay at their original offsets. */
    movq    %rdi, wt_a0(%rip)
    movq    %rsi, wt_a1(%rip)
    movq    %rdx, wt_a2(%rip)
    movq    %rcx, wt_a3(%rip)
    movq    %r8,  wt_a4(%rip)
    movq    %r9,  wt_a5(%rip)
    movq    %rax, wt_sav_rax(%rip)
    movl    $\idx, %edi
    subq    $8, %rsp
    call    wt_pre@PLT
    addq    $8, %rsp
    movq    wt_sav_rax(%rip), %rax
    movq    wt_a0(%rip), %rdi
    movq    wt_a1(%rip), %rsi
    movq    wt_a2(%rip), %rdx
    movq    wt_a3(%rip), %rcx
    movq    wt_a4(%rip), %r8
    movq    wt_a5(%rip), %r9
    jmp     __real_\name
900\idx:
    movq    %rax, wt_sav_rax(%rip)
    movq    %rdx, wt_sav_rdx(%rip)
    rdtsc
    shlq    $32, %rdx
    orq     %rdx, %rax
    subq    wt_t0+8*\idx(%rip), %rax
    movq    %rax, %rsi                      /* arg2 = cycles                             */
    movl    $\idx, %edi                     /* arg1 = slot                               */
    call    wt_record@PLT                   /* rsp is 16-aligned here, see header        */
    movq    wt_sav_rdx(%rip), %rdx
    movq    wt_sav_rax(%rip), %rax
    jmp     *wt_ret+8*\idx(%rip)
    .size   __wrap_\name, .-__wrap_\name
.endm

/* Microphysics is not a pass-through wrapper any more: retargeting its array pointers means
 * rewriting arguments 6..417, which live in the CALLER's frame. So instead of tail-jumping we
 * collect all 402 into wt_argv and let C rebuild the call through wt_call_n. This is a real
 * CALL, so the return-address hook the other five use is not needed here.
 *
 * Alignment: at entry rsp%16==8; sub 8 -> 0, which is what a CALL requires; add 8 and RET
 * leaves the caller's frame exactly as it was, and the caller still cleans its own stack args. */
    .globl  __wrap_module_microphysics_driver_mp_microphysics_driver_
    .type   __wrap_module_microphysics_driver_mp_microphysics_driver_, @function
__wrap_module_microphysics_driver_mp_microphysics_driver_:
    movq    %rdi, wt_argv+0(%rip)
    movq    %rsi, wt_argv+8(%rip)
    movq    %rdx, wt_argv+16(%rip)
    movq    %rcx, wt_argv+24(%rip)
    movq    %r8,  wt_argv+32(%rip)
    movq    %r9,  wt_argv+40(%rip)
    leaq    8(%rsp), %rsi               /* stack arguments begin above the return address */
    leaq    wt_argv+48(%rip), %rdi
    movl    $396, %ecx                  /* 402 - 6 (compiled interface, see gen_argmap.py); DF is 0 on entry per the ABI */
    rep     movsq
    subq    $8, %rsp
    call    wt_micro_call@PLT
    addq    $8, %rsp
    ret
    .size   __wrap_module_microphysics_driver_mp_microphysics_driver_, .-__wrap_module_microphysics_driver_mp_microphysics_driver_

/* Radiation has 422 by-reference arguments.  Like microphysics, collect the
 * caller's register and stack vector so C can retarget array arguments for R1. */
    .globl  __wrap_module_radiation_driver_mp_radiation_driver_
    .type   __wrap_module_radiation_driver_mp_radiation_driver_, @function
__wrap_module_radiation_driver_mp_radiation_driver_:
    movq    %rdi, wt_argv+0(%rip)
    movq    %rsi, wt_argv+8(%rip)
    movq    %rdx, wt_argv+16(%rip)
    movq    %rcx, wt_argv+24(%rip)
    movq    %r8,  wt_argv+32(%rip)
    movq    %r9,  wt_argv+40(%rip)
    leaq    8(%rsp), %rsi
    leaq    wt_argv+48(%rip), %rdi
    movl    $416, %ecx                  /* 422 - 6 */
    rep     movsq
    subq    $8, %rsp
    call    wt_rad_call@PLT
    addq    $8, %rsp
    ret
    .size   __wrap_module_radiation_driver_mp_radiation_driver_, .-__wrap_module_radiation_driver_mp_radiation_driver_
WTWRAP module_radiation_driver_mp_pre_radiation_driver_,   2
WTWRAP module_pbl_driver_mp_pbl_driver_,                   3
WTWRAP module_surface_driver_mp_surface_driver_,           4
WTWRAP module_cumulus_driver_mp_cumulus_driver_,           5

    .bss
    .align 8
wt_t0:      .zero 8*6
wt_ret:     .zero 8*6
wt_sav_rax: .zero 8
wt_sav_rdx: .zero 8
/* Argument-register save slots. Static, not stack, so the stack layout handed to __real_ is
 * untouched. Safe because the wrapped drivers are never nested (first_rk_step_part1 calls
 * five of them in sequence, solve_em calls the sixth) and the call sites sit outside every
 * OpenMP region. */
wt_a0:      .zero 8
wt_a1:      .zero 8
wt_a2:      .zero 8
wt_a3:      .zero 8
wt_a4:      .zero 8
wt_a5:      .zero 8
    .globl wt_argv
wt_argv:    .zero 8*422

    .section .note.GNU-stack,"",@progbits
