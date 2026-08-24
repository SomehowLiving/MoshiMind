# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Test-collection-only workaround for environments without torch installed.

``moshi/moshi/__init__.py`` eagerly imports ``models``/``conditioners``/etc.,
all of which require torch. That's correct for the real package, but it means
even torch-free modules like ``moshi.cognitive`` can't be imported for
testing in an environment where torch genuinely isn't installed (e.g. this
sandbox — see the repo's execution audit).

When torch *is* installed (real dev/CI machines), this does nothing: the
normal package import path is used unmodified.
"""

import sys
import types
import pathlib

try:
    import torch  # noqa: F401
except ImportError:
    if "moshi" not in sys.modules:
        _pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "moshi"
        stub = types.ModuleType("moshi")
        stub.__path__ = [str(_pkg_dir)]
        sys.modules["moshi"] = stub
