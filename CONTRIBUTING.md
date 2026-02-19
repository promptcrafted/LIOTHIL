# Contributing to LIOTHIL

Thank you for your interest in contributing.

## Security First

Before contributing, please install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

This installs gitleaks secret scanning. **Every commit is scanned for accidentally included secrets, API keys, and tokens.** If the hook blocks your commit, it caught something — check the output and remove the sensitive data before committing.

**Never commit:**
- API keys, tokens, or passwords
- `.env` files (use `.env.example` with placeholder values)
- Private keys or certificates
- Cloud provider credentials
- Personal access tokens

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Install pre-commit hooks (`pre-commit install`)
4. Make your changes
5. Test that LIOTHIL's interview flow still works correctly
6. Commit with a descriptive message
7. Push to your fork and open a Pull Request

## What We're Looking For

- **New domain templates** — Have you bootstrapped a LIOTHIL environment for a specific domain? Share the patterns that worked.
- **Interview improvements** — Better questions, better follow-up logic, better domain detection.
- **Template refinements** — Improvements to the generated file templates based on real-world use.
- **Documentation** — Examples, guides, case studies of LIOTHIL in action.
- **Security improvements** — Better secret detection patterns, safer defaults.

## What We're Not Looking For

- Domain-specific content that belongs in generated environments, not in LIOTHIL itself.
- Features that add runtime dependencies. LIOTHIL is a prompt — it should stay a prompt.
- Changes that make the CLAUDE.md so large it impacts Claude Code's context window.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be respectful. Be constructive.

## Questions?

Open an [issue](https://github.com/promptcrafted/LIOTHIL/issues) or reach out to the maintainers.
