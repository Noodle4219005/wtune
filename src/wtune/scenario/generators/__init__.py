"""scenario.generators -- importing this package registers every concrete generator (their
modules each apply @base.register at import time). base._load_generators() imports this package
exactly once, lazily, the first time base.get_generator()/list_generators() is called."""

from . import analog  # noqa: F401
from . import aifs    # noqa: F401
