# CI workflow (manual step)

GitHub App integrations cannot create files under `.github/workflows/`,
so this workflow could not be pushed by the PR author. After this PR
is merged, add it manually in the GitHub web UI:

1. In the repo, choose Add file > Create new file
2. Path: `.github/workflows/ci.yml`
3. Paste the YAML below, commit directly to `main`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install .[dev]
      - name: Compile check
        run: python -m py_compile agent_farm/*.py
      - name: Run tests
        run: python -m pytest -q
```
