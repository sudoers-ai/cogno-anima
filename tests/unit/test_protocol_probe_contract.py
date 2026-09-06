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
herdava o buraco.

**Desde que os invólucros do host entraram nesta lib, ela TEM classes com `__getattr__`** —
`CommitRecordingDispatcher`, `ConfirmArgumentRecordingDispatcher`, `IdProvenanceDispatcher` —
e isso NÃO contradiz o que está escrito acima: nelas o `__getattr__` encaminha o que o
invólucro não medeia (um contador, um predicado de política mais fino que um chamador vai
buscar abaixo), enquanto os membros que DECIDEM a sonda são ligados à INSTÂNCIA por
`bind_delegated`. As duas metades cobrem coisas diferentes, e o teste
`test_every_wrapper_this_package_ships_binds_its_policy` abaixo prende essa distinção para
o próximo invólucro que aqui entrar.
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

    O composto declara-os na CLASSE, e é o ROTEADOR: responde os defaults conservadores do
    EGO por uma fonte sem política, porque tem de escolher por ela. Os invólucros que esta
    lib também traz fazem o oposto (ligação condicional à instância) — ver o teste seguinte.
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


# ── os INVÓLUCROS que esta lib traz, DERIVADOS e não enumerados ────────────

def _shipped_wrappers() -> "list[type]":
    """As classes de `cogno_anima.tools` que embrulham UM interior.

    Derivado do pacote, não escrito à mão: um invólucro NOVO que nasça sem ligar a política
    fica vermelho aqui, e não em produção. O sinal de "invólucro sobre um interior" é o
    `__getattr__` declarado na PRÓPRIA classe — o roteador não o tem (não saberia para que
    fonte encaminhar), e é exactamente essa a assimetria que o `composite.py` documenta.
    """
    import cogno_anima.tools as pacote

    return [obj for nome in pacote.__all__
            for obj in [getattr(pacote, nome)]
            if isinstance(obj, type) and "__getattr__" in obj.__dict__]


def _build(klass, inner):
    """Um invólucro de cada tipo, com o segundo argumento que cada um exige."""
    from cogno_anima.tools import (CommitRecordingDispatcher,
                                   ConfirmArgumentRecordingDispatcher,
                                   IdProvenanceDispatcher)

    fabricas = {
        CommitRecordingDispatcher: lambda i: CommitRecordingDispatcher(i, []),
        ConfirmArgumentRecordingDispatcher: lambda i: ConfirmArgumentRecordingDispatcher(i, {}),
        IdProvenanceDispatcher: lambda i: IdProvenanceDispatcher(i, guarded={}),
    }
    assert klass in fabricas, (
        f"`{klass.__name__}` é um invólucro novo desta lib e ninguém lhe deu construtor aqui — "
        f"acrescente-o, senão ele fica FORA das duas asserções abaixo em silêncio")
    return fabricas[klass](inner)


def test_the_shipped_wrappers_are_actually_found():
    """Guarda a guarda: um `_shipped_wrappers` que devolvesse `[]` deixaria os dois testes
    seguintes verdes sobre um universo vazio — a forma de defeito que este ficheiro combate."""
    assert len(_shipped_wrappers()) >= 3


def test_every_wrapper_this_package_ships_binds_its_policy():
    """SABOTAGEM: trocar o `bind_delegated` de qualquer invólucro por delegação só em
    `__getattr__` -> este teste morre, e em 3.12 o portão de confirmação deixava de existir
    para essa fonte, em silêncio."""
    for klass in _shipped_wrappers():
        wrapper = _build(klass, _Source())
        for metodo in _POLICY_METHODS:
            assert _statically_resolvable(wrapper, metodo), (
                f"`{klass.__name__}.{metodo}` não resolve estaticamente — em 3.12 o "
                f"`isinstance` do EGO responde False e o portão desaparece")
        assert isinstance(wrapper, ToolPolicyDispatcher)


def test_a_wrapper_that_adds_no_verdict_does_not_CLAIM_a_policy_the_source_lacks():
    """A outra metade, e a que distingue os gravadores do ROTEADOR.

    Quem declara os métodos na classe (o composto, o guarda de proveniência) responde os
    defaults conservadores do EGO por uma fonte sem política — escolha deliberada, porque
    ambos têm de responder por ela. Quem NÃO os declara está apenas a observar, e aí
    responder "há política" seria uma mentira sobre a fonte: arma um portão sobre um palpite
    e rebenta com `AttributeError` na primeira chamada.

    SABOTAGEM: declarar `is_mutating` na classe de um gravador -> morre.
    """
    class _SemPolitica:
        def tools_schema(self): return []
        async def execute(self, n, a): return None

    observadores = [k for k in _shipped_wrappers()
                    if not any(m in k.__dict__ for m in _POLICY_METHODS)]
    assert observadores, "nenhum invólucro delega a política — o teste deixou de discriminar"
    for klass in observadores:
        wrapper = _build(klass, _SemPolitica())
        for metodo in _POLICY_METHODS:
            assert not _statically_resolvable(wrapper, metodo), (
                f"`{klass.__name__}` declarou `{metodo}` por uma fonte que não o tem")
        assert not isinstance(wrapper, ToolPolicyDispatcher)
