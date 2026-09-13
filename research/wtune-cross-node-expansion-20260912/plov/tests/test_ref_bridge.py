"""Integration contracts for compiler-managed pointer-only wrap bridges."""
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "plov" / "gen_ref_bridge.py"


def names(count):
    return [f"a{i}" for i in range(count)]


class ReferenceBridgeTests(unittest.TestCase):
    def run_tool(self, source, symbol="target", output_text=None):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path, output = directory / "driver.f90", directory / "bridge.c"
            source_path.write_text(textwrap.dedent(source))
            if output_text is not None:
                output.write_text(output_text)
            result = subprocess.run(
                [sys.executable, str(TOOL), "--source", str(source_path),
                 "--procedure", "phase_driver", "--symbol", symbol,
                 "--dispatch", "dispatch", "--output", str(output)],
                text=True, capture_output=True, timeout=60)
            return result, output.read_text() if output.exists() else None

    def test_gnu_wrap_forwards_79_pointers_null_and_nested_dispatch_without_global_argv(self):
        count, args = 79, names(79)
        source = "subroutine phase_driver(" + ",".join(args) + ")\n"
        source += "real(4) :: " + ",".join(args) + "\nend subroutine phase_driver\n"
        result, bridge = self.run_tool(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("void *args[79]", bridge)
        self.assertNotIn("wt_argv", bridge)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "bridge.c").write_text(bridge)
            formals = ", ".join(f"void *a{i}" for i in range(count))
            passed = ", ".join(f"a{i}" for i in range(count))
            array = ", ".join(f"a{i}" for i in range(count))
            target = f'''#include <assert.h>
static int calls;
void target({formals}) {{
  void *p[{count}] = {{{array}}};
  int nested = p[0] && *(int *)p[0] == 1000;
  for (int i=0;i<{count};++i) {{ if (i == 5) assert(!p[i]); else assert(*(int *)p[i] == (nested ? 1000+i : i)); }}
  for (int i=0;i<{count};++i) if (i != 5) ++*(int *)p[i];
  ++calls;
}}
int target_calls(void) {{ return calls; }}
'''
            caller_args = ", ".join("0" if i == 5 else f"&outer[{i}]" for i in range(count))
            caller = f'''#include <assert.h>
void target({formals}); int target_calls(void);
int main(void) {{ int outer[{count}]; for(int i=0;i<{count};++i) outer[i]=i; target({caller_args});
  for(int i=0;i<{count};++i) {{ if(i != 5) assert(outer[i] == i+1); }}
  assert(target_calls()==2); return 0; }}
'''
            nested_args = ", ".join("0" if i == 5 else f"&nested[{i}]" for i in range(count))
            dispatch = f'''#include <assert.h>
#include <stddef.h>
void target({formals});
void dispatch(void **args, void (*invoke)(void **), size_t n) {{
  static int depth; assert(n == {count}); ++depth;
  if (depth == 1) {{ int nested[{count}]; for(int i=0;i<{count};++i) nested[i]=1000+i; target({nested_args}); }}
  invoke(args); --depth;
}}
'''
            for filename, body in {"target.c": target, "caller.c": caller, "dispatch.c": dispatch}.items():
                (directory / filename).write_text(body)
            compiler = shlex.split(os.environ.get("CC", "cc"))
            cmd = compiler + ["-std=c11", "-Wall", "-Wextra", "-Werror", "target.c", "caller.c", "dispatch.c", "bridge.c", "-Wl,--wrap=target", "-o", "bridge-test"]
            built = subprocess.run(cmd, cwd=directory, text=True, capture_output=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            ran = subprocess.run([str(directory / "bridge-test")], text=True, capture_output=True, timeout=60)
            self.assertEqual(ran.returncode, 0, ran.stderr)

    def test_unsupported_schema_preserves_existing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              type(state) :: a
            end subroutine phase_driver
        """, output_text="previous-output")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(body, "previous-output")

    def test_unknown_attribute_preserves_existing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(4), contiguous :: a
            end subroutine phase_driver
        """, output_text="previous-output")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(body, "previous-output")

    def test_invalid_symbol_is_rejected_without_writing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(4) :: a
            end subroutine phase_driver
        """, symbol="target; injected", output_text="previous-output")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identifier", result.stderr)
        self.assertEqual(body, "previous-output")


if __name__ == "__main__":
    unittest.main()
