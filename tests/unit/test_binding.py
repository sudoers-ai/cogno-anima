"""``bind_delegated`` — the only shape that keeps a wrapper's protocol probe HONEST.

Three properties, and each one is a different way the naive shapes fail:

  * a member the inner HAS is bound, so the probe answers True (and the gate exists);
  * a member the inner LACKS is NOT bound, so the probe answers False (a wrapper must not
    claim a policy on behalf of a source that declared none);
  * a member the WRAPPER defines is never overwritten, so a wrapper that tightens
    ``requires_confirmation`` cannot be disarmed by forwarding the inner's version over it.

The static-resolution half (why ``__getattr__`` is not enough on 3.12) is pinned by
``test_protocol_probe_contract.py``; here we pin the binder itself.
"""

from __future__ import annotations

import inspect

from cogno_anima.tools import bind_delegated
from cogno_anima.tools.base import ToolPolicyDispatcher

_POLICY = ("is_mutating", "requires_confirmation")


class _WithPolicy:
    def tools_schema(self): return []
    async def execute(self, name, arguments): return None
    def is_mutating(self, name): return name != "read"
    def requires_confirmation(self, name): return name == "wipe"


class _NoPolicy:
    def tools_schema(self): return []
    async def execute(self, name, arguments): return None


class _Wrapper:
    def __init__(self, inner):
        self._inner = inner
        bind_delegated(self, inner, *_POLICY)

    def tools_schema(self): return self._inner.tools_schema()

    async def execute(self, name, arguments): return await self._inner.execute(name, arguments)


class _TighteningWrapper(_Wrapper):
    """Declares its own ``requires_confirmation`` — the binder must leave it alone."""

    def requires_confirmation(self, name): return True


def _statically_resolvable(obj, name: str) -> bool:
    """What Python 3.12 asks: does the attribute exist WITHOUT running ``__getattr__``?"""
    try:
        inspect.getattr_static(obj, name)
        return True
    except AttributeError:
        return False


def test_a_member_the_inner_has_is_bound_on_the_INSTANCE():
    """MUTATION: bind on the class (or via ``__slots__``) -> the negative test below dies."""
    w = _Wrapper(_WithPolicy())
    for m in _POLICY:
        assert m in vars(w), f"{m} must live in the instance __dict__, where getattr_static reads"
        assert _statically_resolvable(w, m)
    assert isinstance(w, ToolPolicyDispatcher)
    assert w.is_mutating("write") is True and w.requires_confirmation("wipe") is True


def test_a_member_the_inner_LACKS_is_not_invented():
    """The whole reason the binding is conditional: a safety wrapper must not be able to
    make the probe claim a policy for a source that declared none — that arms a gate over
    guesses, then raises AttributeError on the first call.

    MUTATION: drop the ``if fn is not None`` guard -> this dies.
    """
    w = _Wrapper(_NoPolicy())
    for m in _POLICY:
        assert m not in vars(w)
        assert not _statically_resolvable(w, m)
    assert not isinstance(w, ToolPolicyDispatcher)


def test_a_member_the_WRAPPER_declares_is_never_shadowed():
    """An instance attribute beats a class method, so binding over the wrapper's own logic
    would silently disarm it — a wrapper that holds extra calls would stop holding.

    MUTATION: drop the ``if member in declared: continue`` guard -> this dies.
    """
    w = _TighteningWrapper(_WithPolicy())
    assert "requires_confirmation" not in vars(w), "the binder shadowed the wrapper's own rule"
    assert w.requires_confirmation("anything") is True      # the wrapper's, not the inner's
    assert "is_mutating" in vars(w)                          # the merely-forwarded one still binds


def test_binding_is_idempotent_and_survives_an_unknown_member_name():
    inner = _WithPolicy()
    w = _Wrapper(inner)
    bind_delegated(w, inner, *_POLICY)                       # again
    bind_delegated(w, inner, "no_such_member_at_all")
    assert w.is_mutating("write") is True
    assert not hasattr(w, "no_such_member_at_all")
