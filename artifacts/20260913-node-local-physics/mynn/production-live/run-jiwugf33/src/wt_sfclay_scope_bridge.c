/* Generated compiler-managed first-RK scope: 84 raw references.
 * compiled_source_sha256=4668213aaec35d72f7bdbd2473d810b4538370f8aadf5979ef940f315be9c43b
 * declaration_profile_sha256=545d198388690ca64d46e4aa8a0fabf565d4e176bf02dd93f72c2ab9fb0de7cf
 * This audited prototype is not certification of a native WRF binary ABI. */
#include <stdint.h>
#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)
#include "wt_sfclay_live.h"
#endif
#ifdef WT_MYNN_NODE_LOCAL
#include "wt_mynn_node_live.h"
#endif
extern void __real_module_first_rk_step_part1_mp_first_rk_step_part1_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34, void *a35, void *a36, void *a37, void *a38, void *a39, void *a40, void *a41, void *a42, void *a43, void *a44, void *a45, void *a46, void *a47, void *a48, void *a49, void *a50, void *a51, void *a52, void *a53, void *a54, void *a55, void *a56, void *a57, void *a58, void *a59, void *a60, void *a61, void *a62, void *a63, void *a64, void *a65, void *a66, void *a67, void *a68, void *a69, void *a70, void *a71, void *a72, void *a73, void *a74, void *a75, void *a76, void *a77, void *a78, void *a79, void *a80, void *a81, void *a82, void *a83);
void __wrap_module_first_rk_step_part1_mp_first_rk_step_part1_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34, void *a35, void *a36, void *a37, void *a38, void *a39, void *a40, void *a41, void *a42, void *a43, void *a44, void *a45, void *a46, void *a47, void *a48, void *a49, void *a50, void *a51, void *a52, void *a53, void *a54, void *a55, void *a56, void *a57, void *a58, void *a59, void *a60, void *a61, void *a62, void *a63, void *a64, void *a65, void *a66, void *a67, void *a68, void *a69, void *a70, void *a71, void *a72, void *a73, void *a74, void *a75, void *a76, void *a77, void *a78, void *a79, void *a80, void *a81, void *a82, void *a83) {
#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)
    uint64_t token = wt_sfclay_first_rk_begin(a0, a1);
#endif
#ifdef WT_MYNN_NODE_LOCAL
    uint64_t mynn_token = wt_mynn_first_rk_begin(a0, a1);
#endif
    __real_module_first_rk_step_part1_mp_first_rk_step_part1_(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a80, a81, a82, a83);
#ifdef WT_MYNN_NODE_LOCAL
    wt_mynn_first_rk_end(mynn_token);
#endif
#if !defined(WT_MYNN_NODE_LOCAL) || defined(WT_SFCLAY_NODE_LOCAL)
    wt_sfclay_first_rk_end(token);
#endif
}
#ifdef WT_SFCLAY_SCOPE_TEST_HELPERS
extern void module_first_rk_step_part1_mp_first_rk_step_part1_(void *a0, void *a1, void *a2, void *a3, void *a4, void *a5, void *a6, void *a7, void *a8, void *a9, void *a10, void *a11, void *a12, void *a13, void *a14, void *a15, void *a16, void *a17, void *a18, void *a19, void *a20, void *a21, void *a22, void *a23, void *a24, void *a25, void *a26, void *a27, void *a28, void *a29, void *a30, void *a31, void *a32, void *a33, void *a34, void *a35, void *a36, void *a37, void *a38, void *a39, void *a40, void *a41, void *a42, void *a43, void *a44, void *a45, void *a46, void *a47, void *a48, void *a49, void *a50, void *a51, void *a52, void *a53, void *a54, void *a55, void *a56, void *a57, void *a58, void *a59, void *a60, void *a61, void *a62, void *a63, void *a64, void *a65, void *a66, void *a67, void *a68, void *a69, void *a70, void *a71, void *a72, void *a73, void *a74, void *a75, void *a76, void *a77, void *a78, void *a79, void *a80, void *a81, void *a82, void *a83);
void wt_sfclay_scope_test_call(void **args) { module_first_rk_step_part1_mp_first_rk_step_part1_(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11], args[12], args[13], args[14], args[15], args[16], args[17], args[18], args[19], args[20], args[21], args[22], args[23], args[24], args[25], args[26], args[27], args[28], args[29], args[30], args[31], args[32], args[33], args[34], args[35], args[36], args[37], args[38], args[39], args[40], args[41], args[42], args[43], args[44], args[45], args[46], args[47], args[48], args[49], args[50], args[51], args[52], args[53], args[54], args[55], args[56], args[57], args[58], args[59], args[60], args[61], args[62], args[63], args[64], args[65], args[66], args[67], args[68], args[69], args[70], args[71], args[72], args[73], args[74], args[75], args[76], args[77], args[78], args[79], args[80], args[81], args[82], args[83]); }
#endif
