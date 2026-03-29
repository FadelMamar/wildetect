## Python code running
when attempting to run python code, first try "uv run script.py" before "python script.py". Whenever "uv" is installed use it!

# Deploying the website
To deploy the website, run the following command:

```bash
npx docusaurus deploy
```
```PowerShell
cmd /C 'set "USE_SSH=true" && npx docusaurus deploy'
```

## Git commit messages
Use conventional-style prefixes:

- `feat: ...` – new feature.
- `fix: ...` – bug fix.
- `docs: ...` – documentation only.
- `test: ...` – tests only.
- `refactor: ...` – internal refactor.
- `chore: ...` – maintenance / tooling.

Example:

```bash
git commit -m "feat: add new dataset analyzer"
```

When creating a commit message, be short and to the point.