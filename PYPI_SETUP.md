# Publishing the Cogno ecosystem to PyPI

The Cogno libraries are separate PyPI packages that depend on one another by
name. This guide covers the one-time setup and the two publishing paths.

## TL;DR

- **First release (bootstrap):** run `scripts/publish_all.sh` locally with a PyPI
  API token — it builds and uploads every lib **in dependency order**.
- **Every release after that:** cut a **GitHub Release** in a repo and its
  `.github/workflows/publish.yml` publishes that package via **Trusted
  Publishing** (OIDC, no stored token).

## Why order matters

`pip install cogno-anima` resolves `cogno-synapse`, which resolves `cogno-homeo`.
A dependency must already exist on PyPI before a dependent is uploaded, or the
first install of the dependent will fail to resolve. Publish order:

```
cogno-homeo → cogno-synapse → cogno-anima → { cogno-cortex, cogno-mcp, cogno-soma }
```

Independent libs (`aegis`, `engram`, `gateway`, `herald`, `meter`,
`observability`, `persona`, `praxis`) can go anytime. `scripts/publish_all.sh`
already encodes a safe total order.

## One-time setup

1. Create a PyPI account and enable 2FA: <https://pypi.org/account/register/>.
2. **Bootstrap token** (first release only): create an *account-scoped* API token
   at <https://pypi.org/manage/account/token/>. You will narrow this to
   per-project tokens (or drop it entirely for Trusted Publishing) afterwards.
3. **Trusted Publishing** (for CI, per repo): at
   <https://pypi.org/manage/account/publishing/> add a *pending publisher* for
   each package:
   - PyPI Project Name: e.g. `cogno-anima`
   - Owner: `sudoers-ai`  ·  Repository: e.g. `cogno-anima`
   - Workflow name: `publish.yml`  ·  Environment: `pypi`
4. In each GitHub repo, create an **Environment** named `pypi`
   (Settings → Environments) — the workflow references it.

## First release (bootstrap)

```bash
cd cogno-core                       # the cogno-anima repo
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxxxxx  # your account-scoped API token

./scripts/publish_all.sh --dry-run   # build everything, upload nothing (sanity)
./scripts/publish_all.sh --test      # optional: rehearse against TestPyPI
./scripts/publish_all.sh             # the real thing, in dependency order
```

The script builds each package fresh, runs `twine check`, and uploads. It skips
missing repos and stops on the first upload error (so you can fix and re-run —
already-uploaded versions are simply left in place; bump the version to re-push).

## Ongoing releases (per repo, no token)

Once the packages exist and Trusted Publishing is configured:

1. Bump `version` in that repo's `pyproject.toml`.
2. Tag and cut a GitHub Release (`vX.Y.Z`).
3. `publish.yml` builds and publishes automatically via OIDC.

## Notes

- **Versions are immutable** on PyPI — a published `X.Y.Z` cannot be replaced, only
  yanked. Rehearse with `--dry-run` / `--test` first.
- The benchmark (`cognobench/`) and `tests/` are **excluded** from the wheels by
  `pyproject.toml`'s package discovery — verified in the built artifacts.
- Author metadata across the ecosystem: **Vinicius Vale** (author), **Sudoers AI**
  (maintainer).
