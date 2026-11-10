"""The single source of truth for WTune's version.

Kept in its own module because the pieces that need it -- wtune.cli's CLI, the campaign scripts
that stamp a ledger, and the test that pins CHANGELOG.md to it -- must not import each other.

Bump this and add the matching CHANGELOG.md section in the same commit; test_wtune_version.py
fails otherwise, which is the whole point of having it in a file rather than in a git tag.
"""

__version__ = "0.10.0"
