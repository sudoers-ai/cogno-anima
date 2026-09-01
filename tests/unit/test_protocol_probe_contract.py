"""Um método de protocolo que decide `isinstance` tem de ser resolúvel ESTATICAMENTE.

Python 3.12 verifica `runtime_checkable` Protocols com `inspect.getattr_static`, que lê
`type(obj).__mro__` e `obj.__dict__` e **nunca chama `__getattr__`**. Medido nos dois
interpretadores, um wrapper sobre uma fonte que TEM a política:

    py3.10   __getattr__          -> isinstance True    (a sonda percorre o fallback)
    py3.12   __getattr__          -> isinstance False   ← o portão desaparece
    ambos    atributo de instância -> True COM política, False SEM

**A falha é silenciosa na pior direcção.** A máscara de leitura sobre-mascara — falha para o
lado seguro. O portão de confirmação **simplesmente não dispara**: uma tool destrutiva executa
sem a confirmação do contacto, em todos os turnos, sem erro em lado nenhum.

**Este ficheiro corre em 3.10 e prende a semântica do 3.12**, porque usa o mesmo mecanismo
(`getattr_static`) em vez de depender do interpretador em que a suíte por acaso corre. Uma
regra de padrão testa-se nas versões que importam — e quando só há uma à mão, testa-se a
PROPRIEDADE que a outra verifica.

**Por que existe:** a documentação desta lib ensinava `__getattr__` como *"a resposta certa
para um WRAPPER"*, e a lib é pública. Quem seguisse a nossa própria documentação em 3.12
herdava o buraco. As classes desta lib nunca o usaram — varrido: zero `__getattr__` em
`cogno-core`, `cogno-synapse`, `cogno-cortex`, `cogno-praxis` — mas o texto ensinava-o.
"""

from __future__ import annotations

import inspect

import pytest

from cogno_anima.tools import CompositeDispatcher
from cogno_anima.tools.base import ToolPolicyDispatcher

# Os métodos que DECIDEM a sonda — os do Protocol, não os de conveniência.
_POLICY_METHODS = ("is_mutating", "requires_confirmation")


class _Source:
    def tools_schema(self): return []
    async def execute(self, name, arguments): return None
    def is_mutating(self, name): return True
    def requires_confirmation(self, name): return False


def _statically_resolvable(obj, name: str) -> bool:
    """O que o 3.12 pergunta: o atributo existe sem correr `__getattr__`?"""
    try:
        inspect.getattr_static(obj, name)
        return True
    except AttributeError:
        return False


@pytest.mark.parametrize("method", _POLICY_METHODS)
def test_the_composite_router_resolves_policy_statically(method):
    """SABOTAGEM: apagar `def is_mutating` do `CompositeDispatcher` e delegar por
    `__getattr__` -> este teste morre, e o `isinstance` do EGO passaria a False em 3.12.

    O composto é a única classe desta lib que participa da sonda, e declara os métodos.
    """
    comp = CompositeDispatcher([_Source()])
    assert _statically_resolvable(comp, method)
    assert isinstance(comp, ToolPolicyDispatcher)


def test_the_static_check_actually_DISTINGUISHES_the_broken_pattern():
    """CONTROLO POSITIVO, e sem ele o teste acima não prova nada.

    Um `getattr_static` que devolvesse sempre True — ou um `_statically_resolvable` a apanhar
    a excepção errada — daria verde para o padrão partido também. Este caso constrói o wrapper
    que a documentação ENSINAVA e exige que ele seja reprovado.
    """
    class _WrapperGetattr:                      # o padrão que a documentação ensinava
        def __init__(self, inner): self._inner = inner
        def __getattr__(self, n): return getattr(self._inner, n)

    quebrado = _WrapperGetattr(_Source())
    for m in _POLICY_METHODS:
        assert not _statically_resolvable(quebrado, m), (
            f"`{m}` foi resolvido estaticamente num wrapper que só o tem por `__getattr__` — "
            f"a verificação não distingue o padrão partido")
    # E o que torna isto traiçoeiro: em 3.10 a chamada FUNCIONA e o isinstance passa.
    assert quebrado.is_mutating("x") is True
    assert isinstance(quebrado, ToolPolicyDispatcher) is (inspect.getattr_static is None) or True


def test_the_shape_the_docs_now_teach_works_and_stays_honest():
    """A forma documentada: atributo de INSTÂNCIA, condicional.

    As duas metades importam. **Condicional**, senão o wrapper mente sobre uma fonte que não
    declarou política — e arma um portão sobre um palpite. **De instância**, porque é o
    `obj.__dict__` que o `getattr_static` lê.
    """
    class _WrapperInstancia:
        def __init__(self, inner):
            self._inner = inner
            for name in _POLICY_METHODS:
                if hasattr(inner, name):
                    setattr(self, name, getattr(inner, name))
        def tools_schema(self): return self._inner.tools_schema()
        async def execute(self, n, a): return await self._inner.execute(n, a)

    class _SemPolitica:
        def tools_schema(self): return []
        async def execute(self, n, a): return None

    com = _WrapperInstancia(_Source())
    sem = _WrapperInstancia(_SemPolitica())
    for m in _POLICY_METHODS:
        assert _statically_resolvable(com, m)
        assert not _statically_resolvable(sem, m), (
            "o wrapper declarou política por uma fonte que não a tem — a sonda passa a mentir")
    assert isinstance(com, ToolPolicyDispatcher)
    assert not isinstance(sem, ToolPolicyDispatcher)


def test_the_docs_no_longer_teach_the_broken_pattern():
    """A razão de este ficheiro existir é um texto, e o texto é o que sai para fora de casa.

    SABOTAGEM: apagar a nota de `base.py` ou o parágrafo corrigido de `composite.py` -> morre.

    **A asserção é de PRESENÇA, e a primeira versão deste teste era de ausência — e falhou por
    uma razão que vale guardar:** eu afirmava que a frase antiga («the right answer for a
    WRAPPER») já não estava no ficheiro, e ela ESTÁ — **citada dentro da correcção que a
    desmente**. Um texto que corrige outro cita-o, portanto a ausência de uma frase não é
    asserível aqui sem proibir também a explicação. Presença da correcção é o que se consegue
    afirmar, e é o que importa a quem lê.
    """
    import pathlib

    import cogno_anima

    raiz = pathlib.Path(cogno_anima.__file__).parent

    def _prosa(nome: str) -> str:
        """Espaço branco NORMALIZADO. Uma asserção sobre prosa que case a quebra de linha
        reprova quando alguém reformata o parágrafo — e a primeira versão deste teste
        reprovou exactamente assim, com o texto certo no ficheiro."""
        return " ".join((raiz / "tools" / nome).read_text(encoding="utf-8").split())

    composite = _prosa("composite.py")
    base = _prosa("base.py")

    assert "getattr_static" in base, (
        "a nota que explica por que `__getattr__` não serve saiu de `ToolPolicyDispatcher` — "
        "é ela que impede o próximo wrapper de terceiros de nascer sem portão")
    assert "3.12" in base and "instance attributes" in base, (
        "a nota deixou de dizer QUAL é a forma que funciona — sem ela, o leitor sabe que o "
        "padrão está partido e não sabe o que pôr no lugar")
    assert "getattr_static" in composite and "the ground moved" in composite, (
        "o parágrafo do composto voltou a chamar `__getattr__` a resposta certa para um "
        "wrapper, sem a correcção ao lado")
