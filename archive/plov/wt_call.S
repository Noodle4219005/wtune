/* wt_call_n(fn, argv, n) -- call fn with n pointer arguments taken from argv[].
 *
 * Needed because retargeting microphysics_driver's array pointers means rewriting arguments
 * 6..417, and those live in the CALLER's stack frame: patching them in place would corrupt the
 * caller. So the wrapper builds its own argument vector and this marshals it into a fresh frame.
 *
 * SysV x86-64: Intel Fortran passes every argument by reference, so all 418 are addresses ->
 * INTEGER class -> rdi,rsi,rdx,rcx,r8,r9 then the stack. No xmm register ever carries one, so
 * al (the vararg float count) is zeroed and nothing else is needed.
 */
    .text
    .globl wt_call_n
    .type  wt_call_n, @function
wt_call_n:
    pushq   %rbp
    movq    %rsp, %rbp
    pushq   %rbx
    pushq   %r12
    movq    %rdi, %rbx              /* fn   */
    movq    %rsi, %r12              /* argv */
    movl    %edx, %eax              /* n    */
    andq    $-16, %rsp              /* align first, then subtract multiples of 16, so rsp is */
                                    /* 16-byte aligned at the CALL as the ABI requires       */
    subl    $6, %eax
    jle     2f
    movslq  %eax, %rcx              /* rcx = number of stack arguments */
    movq    %rcx, %rax
    addq    $1, %rax
    andq    $-2, %rax
    shlq    $3, %rax                /* bytes, rounded up to a multiple of 16 */
    subq    %rax, %rsp
    leaq    48(%r12), %r8           /* &argv[6] */
    movq    %rsp, %r9
1:  movq    (%r8), %r10
    movq    %r10, (%r9)
    addq    $8, %r8
    addq    $8, %r9
    decq    %rcx
    jnz     1b
2:  movq    0(%r12), %rdi
    movq    16(%r12), %rdx
    movq    24(%r12), %rcx
    movq    32(%r12), %r8
    movq    40(%r12), %r9
    movq    8(%r12), %rsi           /* rsi last: it held argv */
    xorl    %eax, %eax
    call    *%rbx
    leaq    -16(%rbp), %rsp
    popq    %r12
    popq    %rbx
    popq    %rbp
    ret
    .size   wt_call_n, .-wt_call_n
    .section .note.GNU-stack,"",@progbits
