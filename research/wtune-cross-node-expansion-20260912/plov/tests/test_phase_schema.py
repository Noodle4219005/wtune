"""Contract tests for the conservative Fortran phase-schema audit CLI."""
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "plov" / "gen_phase_schema.py"


class PhaseSchemaTests(unittest.TestCase):
    def run_tool(self, source, procedure="phase_driver", existing=None):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inp = tmp / "driver.f90"
            out = tmp / "schema.json"
            inp.write_text(textwrap.dedent(source))
            if existing is not None:
                out.write_text(existing)
            result = subprocess.run(
                [sys.executable, str(TOOL), "--source", str(inp),
                 "--procedure", procedure, "--output", str(out)],
                text=True, capture_output=True)
            return result, out.read_text() if out.exists() else None

    def test_manifest_preserves_order_symbols_and_supported_explicit_shapes(self):
        result, body = self.run_tool("""
            subroutine PHASE_DRIVER( &
              & temperature, count, enabled, field)
              real(kind=8), intent(inout) :: temperature
              integer(kind=4), intent(in) :: count
              logical(kind=4), optional, intent(in) :: enabled
              real(kind=8), dimension(ims:ime, kms:kme, jms:jme) :: field
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(body)
        self.assertTrue(manifest["safe_supported"])
        self.assertEqual([a["name"] for a in manifest["arguments"]],
                         ["temperature", "count", "enabled", "field"])
        self.assertEqual(manifest["arguments"][3]["dimensions"],
                         ["ims:ime", "kms:kme", "jms:jme"])
        self.assertTrue(manifest["arguments"][2]["optional"])
        self.assertEqual(manifest["arguments"][0]["intent"], "inout")
        self.assertEqual(manifest["arguments"][0]["kind"], "8")
        self.assertEqual(len(manifest["source"]["sha256"]), 64)

    def test_comments_case_and_per_variable_multidimensional_shapes_are_audited(self):
        result, body = self.run_tool("""
            SUBROUTINE phase_driver(a, b) ! header comment
              REAL(KIND = 8), INTENT(IN) :: a(1:nx, ny) ! comment
              INTEGER(KIND=4), INTENT(OUT) :: b
            END SUBROUTINE phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        a = json.loads(body)["arguments"][0]
        self.assertEqual(a["dimensions"], ["1:nx", "ny"])
        self.assertEqual(a["type"], "real")

    def test_derived_type_and_missing_declaration_are_refused_without_guessing(self):
        result, body = self.run_tool("""
            subroutine phase_driver(state, absent)
              type(model_state), intent(inout) :: state
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads(body)
        self.assertFalse(manifest["safe_supported"])
        self.assertIn("derived_type", manifest["arguments"][0]["unsupported_reasons"])
        self.assertIn("missing_declaration", manifest["arguments"][1]["unsupported_reasons"])

    def test_default_kind_and_assumed_shape_are_not_guessed(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real, intent(inout) :: a(:)
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        arg = json.loads(body)["arguments"][0]
        self.assertIn("missing_kind", arg["unsupported_reasons"])
        self.assertIn("assumed_shape", arg["unsupported_reasons"])

    def test_dimension_attribute_stops_at_its_balanced_parenthesis(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(kind=4), dimension(:), intent(in) :: a
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        arg = json.loads(body)["arguments"][0]
        self.assertEqual(arg["dimensions"], [":"])
        self.assertEqual(arg["intent"], "in")
        self.assertIn("assumed_shape", arg["unsupported_reasons"])

    def test_nonconstant_upper_bounds_and_assumed_rank_are_refused(self):
        result, body = self.run_tool("""
            subroutine phase_driver(lower_open, upper_open, rank_unknown)
              real(kind=4) :: lower_open(0:)
              real(kind=4) :: upper_open(:10)
              real(kind=4) :: rank_unknown(..)
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(body)["arguments"]
        self.assertIn("assumed_shape", args[0]["unsupported_reasons"])
        self.assertIn("assumed_shape", args[1]["unsupported_reasons"])
        self.assertIn("assumed_rank", args[2]["unsupported_reasons"])

    def test_standalone_attributes_and_callable_declarations_cannot_bypass_gate(self):
        result, body = self.run_tool("""
            subroutine phase_driver(value_arg, shape_arg, pointer_arg, callable_arg)
              real(kind=4) :: value_arg, shape_arg, pointer_arg
              value :: value_arg
              optional :: value_arg
              intent(in) :: value_arg
              dimension :: shape_arg(:)
              pointer :: pointer_arg
              external :: callable_arg
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(body)["arguments"]
        self.assertTrue(args[0]["optional"])
        self.assertEqual(args[0]["intent"], "in")
        self.assertIn("value", args[0]["unsupported_reasons"])
        self.assertIn("assumed_shape", args[1]["unsupported_reasons"])
        self.assertIn("pointer", args[2]["unsupported_reasons"])
        self.assertIn("callable", args[3]["unsupported_reasons"])

    def test_positional_intrinsic_kinds_are_explicit(self):
        result, body = self.run_tool("""
            subroutine phase_driver(real_arg, integer_arg)
              real(4) :: real_arg
              integer(8) :: integer_arg
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(body)["arguments"]
        self.assertTrue(json.loads(body)["safe_supported"])
        self.assertEqual([arg["kind"] for arg in args], ["4", "8"])

    def test_legacy_intrinsic_declarations_and_character_are_fail_closed(self):
        result, body = self.run_tool("""
            subroutine phase_driver(periodic_x, periodic_y, label)
              logical periodic_x, periodic_y
              character(len=*) :: label
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(body)["arguments"]
        self.assertNotIn("missing_declaration", args[0]["unsupported_reasons"])
        self.assertNotIn("missing_declaration", args[1]["unsupported_reasons"])
        self.assertIn("missing_kind", args[0]["unsupported_reasons"])
        self.assertIn("character", args[2]["unsupported_reasons"])

    def test_assumed_size_codimension_and_unknown_attributes_are_refused(self):
        result, body = self.run_tool("""
            subroutine phase_driver(assumed_size, coarray, standalone_coarray, unknown)
              real(4) :: assumed_size(1:*)
              real(4), codimension[*] :: coarray
              real(4) :: standalone_coarray
              codimension standalone_coarray[*]
              real(4), contiguous :: unknown
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(body)["arguments"]
        self.assertIn("assumed_size", args[0]["unsupported_reasons"])
        self.assertIn("coarray", args[1]["unsupported_reasons"])
        self.assertIn("coarray", args[2]["unsupported_reasons"])
        self.assertIn("unsupported_attribute", args[3]["unsupported_reasons"])

    def test_semicolon_separated_statement_is_audited_before_output_replacement(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(4) :: a; value :: a
            end subroutine phase_driver
        """, existing="previous-manifest")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("value", json.loads(body)["arguments"][0]["unsupported_reasons"])

    def test_compiler_abi_directives_preserve_existing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              !DEC$ ATTRIBUTES VALUE :: a
              real(4) :: a
            end subroutine phase_driver
        """, existing="previous-manifest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compiler ABI directive", result.stderr)
        self.assertEqual(body, "previous-manifest")

    def test_cpp_input_is_rejected_and_existing_output_survives(self):
        result, body = self.run_tool("""
            #ifdef USE_X
            subroutine phase_driver(a)
              real(kind=8) :: a
            end subroutine phase_driver
            #endif
        """, existing="previous-manifest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPP", result.stderr)
        self.assertEqual(body, "previous-manifest")

    def test_semicolon_value_attribute_is_manifested_as_unsupported(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(4) :: a; value :: a
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        arg = json.loads(body)["arguments"][0]
        self.assertIn("value", arg["unsupported_reasons"])
        self.assertFalse(json.loads(body)["safe_supported"])

    def test_semicolon_separated_intrinsics_parse_as_distinct_statements(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a, b)
              real(4) :: a; integer(8) :: b
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(body)["safe_supported"])

    def test_quoted_semicolon_bang_and_doubled_quote_do_not_split_or_comment(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              character(len=*), parameter :: note = 'it''s;!'
              real(4) :: a; character(len=*), parameter :: later = "!;"
            end subroutine phase_driver
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(body)["safe_supported"])

    def test_unmatched_quote_rejects_and_preserves_existing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              character(len=8) :: note = 'unclosed; !
              real(4) :: a
            end subroutine phase_driver
        """, existing="previous-manifest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("quote", result.stderr)
        self.assertEqual(body, "previous-manifest")

    def test_failure_does_not_replace_existing_output(self):
        result, body = self.run_tool("""
            subroutine other(a)
              real(kind=8) :: a
            end subroutine other
        """, existing="previous-manifest")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(body, "previous-manifest")

    def test_interface_and_contained_scopes_cannot_declare_outer_dummy(self):
        for nested in ("""
            interface
              subroutine child(a)
                real(4) :: a
              end subroutine child
            end interface
        """, """
            contains
              subroutine child(a)
                real(4) :: a
              end subroutine child
        """):
            result, body = self.run_tool("""
                subroutine phase_driver(a)
            """ + nested + """
                end subroutine phase_driver
            """)
            self.assertEqual(result.returncode, 0, result.stderr)
            arg = json.loads(body)["arguments"][0]
            self.assertIn("missing_declaration", arg["unsupported_reasons"])
            self.assertFalse(json.loads(body)["safe_supported"])

    def test_missing_matched_end_preserves_existing_output(self):
        result, body = self.run_tool("""
            subroutine phase_driver(a)
              real(4) :: a
        """, existing="previous-manifest")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("end subroutine", result.stderr)
        self.assertEqual(body, "previous-manifest")


if __name__ == "__main__":
    unittest.main()
