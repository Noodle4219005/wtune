import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PIXI = "/home/u5727520/wrf-stack/wtune-runtime-dev-20260912/pixi.toml"


MOCK = r'''
#include "wt_kf_live.h"
#include <stddef.h>
#include <string.h>
int calls, configure_result = 1;
int got_enabled, got_abi, got_chain, got_selector;
size_t got_spans, got_nodes, got_budget;
struct wt_kf_live_counters counters;
int wt_kf_native_configure_for_grid(int enabled, int abi, int chain,
                                    size_t spans, size_t nodes, size_t budget,
                                    int selector) {
  ++calls; got_enabled=enabled; got_abi=abi; got_chain=chain;
  got_spans=spans; got_nodes=nodes; got_budget=budget; got_selector=selector;
  return configure_result;
}
void wt_kf_live_get_counters(struct wt_kf_live_counters *out) { *out = counters; }
'''

FIXTURE = r'''
#define _POSIX_C_SOURCE 200809L
#include "wt_kf_bootstrap.h"
#include "wt_kf_live.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int calls, configure_result, got_enabled, got_abi, got_chain, got_selector;
extern size_t got_spans, got_nodes, got_budget;
extern struct wt_kf_live_counters counters;
static void die(const char *m) { fputs(m, stderr); exit(2); }
static void yes(int v, const char *m) { if (!v) die(m); }
int main(int ac, char **av) {
  const char *mode = ac == 2 ? av[1] : "";
  if (!strcmp(mode, "unattested")) {
    yes(wt_kf_bootstrap_state() == -1, "pre-start unchecked state\\n");
    setenv("WT_KF_MAX_SPANS", "9", 1);
    yes(wt_kf_bootstrap_start(1, 17) == 0, "unattested start\\n");
    yes(calls == 0 && wt_kf_bootstrap_state() == 0, "environment asserted proof\\n");
  } else if (!strcmp(mode, "defaults")) {
    yes(wt_kf_bootstrap_start(1, 17) == 1, "attested start\\n");
    yes(calls == 1 && got_enabled == 1 && got_abi == 1 && got_chain == 1,
        "native flags\\n");
    yes(got_spans == 4096 && got_nodes == 8192 && got_budget == 268435456u &&
        got_selector == 17, "defaults or selector\\n");
    yes(wt_kf_bootstrap_state() == -1, "unchecked state\\n");
    counters.requested = 1; yes(wt_kf_bootstrap_state() == 0, "fallback state\\n");
    counters.staged = 1; yes(wt_kf_bootstrap_state() == 1, "staged state\\n");
  } else if (!strcmp(mode, "invalid")) {
    setenv("WT_KF_BYTE_BUDGET", "999999999999999999999999999999", 1);
    yes(wt_kf_bootstrap_start(1, 1) == 0 && calls == 0, "overflow must refuse\\n");
    unsetenv("WT_KF_BYTE_BUDGET");
    yes(wt_kf_bootstrap_start(1, 2) == 0 && calls == 0, "start once\\n");
  } else if (!strcmp(mode, "malformed")) {
    setenv("WT_KF_MAX_NODES", "nine", 1);
    yes(wt_kf_bootstrap_start(1, 1) == 0 && calls == 0, "malformed must refuse\\n");
  } else if (!strcmp(mode, "disabled")) {
    setenv("WT_KF_MAX_SPANS", "bad", 1);
    yes(wt_kf_bootstrap_start(0, 5) == 1, "disabled config\\n");
    yes(calls == 1 && got_enabled == 0 && got_selector == 5, "disabled native config\\n");
    yes(wt_kf_bootstrap_state() == 0, "disabled state\\n");
  } else if (!strcmp(mode, "dump")) {
    FILE *f; char b[2048] = {0};
    yes(wt_kf_bootstrap_start(1, 9) == 1, "dump start\\n");
    counters.requested=3; counters.staged=2; counters.fallback=1;
    counters.reason[WT_KF_LIVE_CAPACITY]=1;
    f=tmpfile(); yes(f != NULL, "tmpfile\\n"); wt_kf_bootstrap_dump(f);
    rewind(f); yes(fread(b,1,sizeof b-1,f)>0, "dump read\\n"); fclose(f);
    for (char *line = b; *line; ) {
      yes(line[0] == '#' && line[1] == ' ', "dump CSV comment\\n");
      line = strchr(line, '\n'); if (!line) break; ++line;
    }
    yes(strstr(b,"profile=test-profile") && strstr(b,"source_abi_verified=1") &&
        strstr(b,"grid_selector=9") && strstr(b,"capacity=1"), "dump fields\\n");
  } else die("mode\\n");
  puts("KF bootstrap checks passed"); return 0;
}
'''


class KfBootstrapTests(unittest.TestCase):
    def compile_bootstrap(self, macros):
        cc = shlex.split(os.environ["CC"])
        with tempfile.TemporaryDirectory(prefix="wt-kf-bootstrap-compile-") as tmp:
            return subprocess.run(
                cc + shlex.split(os.environ.get("CFLAGS", "")) + [
                    "-std=c11", "-Wall", "-Wextra", "-Werror", "-Iplov", *macros,
                    "-c", "plov/wt_kf_bootstrap.c", "-o", str(Path(tmp) / "bootstrap.o")],
                cwd=ROOT, text=True, capture_output=True, timeout=60)

    def build_and_run(self, mode, macros=()):
        cc = shlex.split(os.environ["CC"])
        with tempfile.TemporaryDirectory(prefix="wt-kf-bootstrap-") as tmp:
            path = Path(tmp)
            (path / "mock.c").write_text(MOCK)
            (path / "fixture.c").write_text(FIXTURE)
            command = cc + shlex.split(os.environ.get("CFLAGS", "")) + [
                "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Iplov",
                *macros, str(path / "mock.c"), "plov/wt_kf_bootstrap.c",
                str(path / "fixture.c"), "-o", str(path / "fixture")]
            built = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                   timeout=60)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(path / "fixture"), mode], text=True,
                                 capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
            self.assertIn("KF bootstrap checks passed", ran.stdout)

    def test_unattested_build_cannot_be_enabled_by_environment(self):
        self.build_and_run("unattested")

    def test_attested_build_uses_defaults_and_reports_states(self):
        self.build_and_run("defaults", ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1",
                                         "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1"])

    def test_bad_budget_is_immutable_refusal(self):
        self.build_and_run("invalid", ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1",
                                        "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1"])

    def test_malformed_bound_refuses_before_native_configuration(self):
        self.build_and_run("malformed", ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1",
                                          "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1"])

    def test_disabled_request_skips_budget_parse_and_configures_off(self):
        self.build_and_run("disabled", ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1",
                                         "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1"])

    def test_dump_has_profile_proofs_budgets_and_reason_labels(self):
        self.build_and_run("dump", ["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=1",
                                     "-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1",
                                     '-DWT_KF_BUILD_PROFILE_ID="test-profile"'])

    def test_build_proof_macros_reject_nonboolean_and_unpaired_chain(self):
        invalid = self.compile_bootstrap(["-DWT_KF_BUILD_SOURCE_ABI_VERIFIED=2"])
        self.assertNotEqual(invalid.returncode, 0)
        unpaired = self.compile_bootstrap(["-DWT_KF_BUILD_CALLER_CHAIN_VERIFIED=1"])
        self.assertNotEqual(unpaired.returncode, 0)


if __name__ == "__main__":
    unittest.main()
