"""The version of this package — the ONE place it is written.

``cogno_anima.__version__`` is this constant, and the build reads the same line
(``[tool.setuptools.dynamic]`` in ``pyproject.toml``, which declares no number of its own). It
must stay a plain string literal on a line of its own: setuptools reads it from the source,
without importing the package, and ``tests/unit/test_version_has_one_source.py`` reads it the
same way.

It used to be ``importlib.metadata.version("cogno-anima")``, which is the version of whatever
distribution is INSTALLED and not of the code that is running. The two part ways whenever the
package is imported from a source tree — an editable install that predates a bump, a checkout
on ``PYTHONPATH`` — and then the code says the old number. Measured on a development box: the
tree declared 0.1.5 and ``cogno_anima.__version__`` answered 0.1.0.
"""

__version__ = "0.1.6"
