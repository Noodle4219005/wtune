import os, subprocess, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class RkLayoutTests(unittest.TestCase):
 def test_schema_generator_pins_complete_proof(self):
  with tempfile.TemporaryDirectory(prefix="wt-rk-schema-") as tmp:
   tmp=Path(tmp); output=tmp/"schema.h"
   good=subprocess.run(["python3","plov/gen_rk_schema.py","--schema","plov/rk_scalar_schema.json","--output",str(output)],cwd=ROOT,text=True,capture_output=True)
   self.assertEqual(good.returncode,0,good.stderr); text=output.read_text(); self.assertIn("WT_RK_NARG 71",text); self.assertIn('"RQVFTEN"',text); self.assertIn("WT_RK_SHAPE_SPECIES_VOLUME",text)
   bad=tmp/"bad.json"; bad.write_text(Path(ROOT/"plov/rk_scalar_schema.json").read_text().replace('"scs"','"changed"',1))
   refused=subprocess.run(["python3","plov/gen_rk_schema.py","--schema",str(bad),"--output",str(output)],cwd=ROOT,text=True,capture_output=True)
   self.assertNotEqual(refused.returncode,0)
 def test_checked_layout_fixture(self):
  cc=os.environ["CC"].split()
  with tempfile.TemporaryDirectory(prefix="wt-rk-layout-") as tmp:
   exe=Path(tmp)/"rk-layout"
   command=cc+["-std=c11","-O1","-g","-Wall","-Wextra","-Werror","-fsanitize=address,undefined","-fno-omit-frame-pointer","-Iplov","plov/tests/test_rk_layout.c","plov/wt_rk_layout.c","-o",str(exe)]
   built=subprocess.run(command,cwd=ROOT,text=True,capture_output=True); self.assertEqual(built.returncode,0,built.stderr)
   run=subprocess.run([str(exe)],text=True,capture_output=True); self.assertEqual(run.returncode,0,run.stderr); self.assertIn("PASS",run.stdout)
if __name__=="__main__": unittest.main()
