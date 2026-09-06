"""
cogno_anima.tools.binding — bind DELEGATED protocol members onto the instance,
so ``isinstance`` keeps telling the truth about the source underneath.

A wrapper around a :class:`~cogno_anima.tools.base.ToolDispatcher` has three kinds
of protocol member, and only the third needs anything special:

  * one it OVERRIDES with logic of its own → declared on the class, nothing to do here;
  * one the inner does not have at all     → must stay ABSENT, so the probe answers False;
  * one it merely FORWARDS                 → the problem this module exists for.

Forwarding through ``__getattr__`` used to be enough: ``isinstance(x, SomeProtocol)``
resolved members with ``hasattr``, which calls ``__getattr__``, so the probe saw the
inner's method and answered the truth. **Python 3.12 resolves protocol members
STATICALLY** — ``inspect.getattr_static`` reads ``type(obj).__mro__`` and
``obj.__dict__`` and never calls ``__getattr__`` — so every such wrapper silently
stopped satisfying its protocol. That is not cosmetic: in the EGO a single probe
drives two gates, so gate A's fail-safe fired always and gate B — *opt-in, no policy
means no gate* — stopped firing at all. A destructive tool executed without the
confirmation it was supposed to be held for, on every turn, with no error anywhere.

**Declaring the members on the class is the wrong fix**, and prohibited: the probe
would then answer True for a wrapper over a source that has NO policy, which is a
safety wrapper disarming a safety gate. Binding them on the INSTANCE, only when the
inner really has them, is the shape that answers correctly under both resolutions.

``__slots__`` counts as declaring on the class, and this is the trap worth naming
because it is the first thing anyone reaches for: the slot DESCRIPTOR lives on the
class and satisfies the protocol **even when the slot was never set**. Measured
against the real protocol, over a source with no policy:

    __slots__            py3.10 True   py3.12 True    <- claims a policy it lacks
    instance binding     py3.10 False  py3.12 False   <- the truth

**That lie is OLD, not something 3.12 introduced.** Nobody had met it because the
previous shape (``__getattr__``) failed the other way, and a wrapper that DISAPPEARS
from the probe never gets the chance to lie to it. So: if a wrapper needs slots, the
policy members cannot be among them.

``tests/unit/test_protocol_probe_contract.py`` pins the rule with the mechanism 3.12
uses, so it holds on any interpreter.
"""

from __future__ import annotations

from typing import Any

__all__ = ["bind_delegated"]


def bind_delegated(wrapper: Any, inner: Any, *members: str) -> None:
    """Copy each of ``members`` from ``inner`` onto ``wrapper``, when both allow it.

    A member is bound only if BOTH hold:

      * ``inner`` actually has it — otherwise the wrapper would claim a policy its
        source does not have, which is the failure this whole mechanism exists to
        prevent;
      * the wrapper's own class does NOT define it — otherwise an instance attribute
        would SHADOW the wrapper's own logic. A wrapper that overrides
        ``requires_confirmation`` to hold extra calls is exactly that case, and
        forwarding the inner's version over it would silently disarm the hold.

    Idempotent, and safe to call before the wrapper's other attributes are set.
    """
    declared = {m for klass in type(wrapper).__mro__[:-1] for m in vars(klass)}
    for member in members:
        if member in declared:
            continue
        fn = getattr(inner, member, None)
        if fn is not None:
            setattr(wrapper, member, fn)
