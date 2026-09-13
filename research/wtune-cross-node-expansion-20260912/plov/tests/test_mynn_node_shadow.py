#!/usr/bin/env python3
"""Strict and fatal-sanitizer MYNN remote shadow validation."""

import os
from pathlib import Path
import shlex
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[2]
ARTIFACT=ROOT/"artifacts/20260913-node-local-physics/mynn/shadow"
DEFAULT_CC=Path("/home/u5727520/wrf-stack/wtune-fortran-dev-20260912/"
                ".pixi/envs/default/bin/x86_64-conda-linux-gnu-cc")


class MynnNodeShadowTests(unittest.TestCase):
    def test_shadow_validation(self):
        ARTIFACT.mkdir(parents=True,exist_ok=True)
        cc=shlex.split(os.environ.get("CC","")) or [str(DEFAULT_CC)]
        environment=os.environ.copy()
        environment.update(ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1",
                           UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
                           TMPDIR=str(ARTIFACT),TMP=str(ARTIFACT),TEMP=str(ARTIFACT))
        for sanitized in (False,True):
            name="sanitized" if sanitized else "strict"
            executable=ARTIFACT/("test-mynn-shadow-"+name)
            command=[*cc,"-std=c11","-O1","-g","-Wall","-Wextra","-Werror",
                     "-pedantic","-pthread","-Iplov","plov/wt_mynn_layout.c",
                     "plov/wt_mynn_batch.c","plov/wt_mynn_node.c",
                     "plov/wt_node_local.c","plov/tests/test_mynn_node_shadow.c"]
            if sanitized:
                command += ["-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all","-fno-omit-frame-pointer"]
            command += ["-o",str(executable)]
            built=subprocess.run(command,cwd=ROOT,env=environment,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                universal_newlines=True,timeout=60)
            self.assertEqual(built.returncode,0,built.stdout+built.stderr)
            run=subprocess.run([str(executable)],cwd=ROOT,env=environment,
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                universal_newlines=True,timeout=60)
            (ARTIFACT/(name+".txt")).write_text(
                " ".join(shlex.quote(x) for x in command)+"\n"+
                built.stdout+built.stderr+run.stdout+run.stderr+
                "returncode=%d\n" % run.returncode)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn("PASS MYNN node shadow",run.stdout)


if __name__=="__main__":
    unittest.main()
