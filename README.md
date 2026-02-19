<!-- ![LIOTHIL](assets/banner.png) -->
<!-- Uncomment the line above once banner.png is created -->

# LIOTHIL

[![CI](https://github.com/promptcrafted/LIOTHIL/actions/workflows/security.yml/badge.svg)](https://github.com/promptcrafted/LIOTHIL/actions/workflows/security.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

**A research environment scaffold for Claude Code.**

LIOTHIL builds structured, epistemically disciplined AI research environments from a single conversation. You bring your domain. LIOTHIL builds the architecture. Your research partner grows from there.

## What It Does

Drop this project's `CLAUDE.md` into an empty directory, open Claude Code, and LIOTHIL activates. It interviews you about your research domain — your sources, your evidence standards, your tools, your goals — then generates a complete research environment:

- **A configured AI identity** that understands your domain and maintains epistemic discipline
- **An analysis protocol** with phased workflows and verification gates
- **An evidence grading system** calibrated to your field's standards
- **Agent definitions** for specialized analytical and verification tasks
- **An editorial watchlist** that learns from your corrections
- **A directory structure** that separates source material from analytical output

After generation, LIOTHIL replaces itself with your configured identity. The scaffold builder consumes itself to birth the environment. What persists is the structure it built — and that structure grows through use.

## Quick Start

```bash
# Create your project directory
mkdir my-research
cd my-research

# Copy LIOTHIL's CLAUDE.md into it
cp /path/to/liothil/CLAUDE.md .

# Open Claude Code
claude

# LIOTHIL activates and interviews you
# After bootstrap, restart Claude Code to load your new environment
```

## What Gets Generated

```
your-project/
├── CLAUDE.md                          # Your research partner's identity
├── README.md                          # Project overview
├── .claude/
│   ├── rules/
│   │   ├── analysis-protocol.md       # Phased workflow with gates
│   │   ├── evidence-grading.md        # Tier system for findings
│   │   ├── source-attribution.md      # Provenance tagging rules
│   │   └── editorial-watchlist.md     # Error pattern tracking
│   └── agents/
│       ├── analyst.md                 # Primary analytical agent
│       └── verifier.md               # Adversarial verification agent
├── source/                            # Primary sources go here
├── results/                           # Analytical output goes here
└── tools/                             # Computation scripts (if applicable)
```

## The Patterns

LIOTHIL encodes domain-agnostic research architecture distilled from a production environment built across 60+ collaborative sessions. The core patterns:

**Epistemic Charter** — Every claim tagged with its provenance: verified, source-canonical, analytical contribution, or unknown. Prevents the most dangerous failure mode in AI-augmented research: synthesis masquerading as source material.

**Phased Analysis** — Sequential workflow with mandatory checkpoints. Each phase writes to disk before the next begins. Verification gates prevent garbage propagation.

**Evidence Grading** — Tiered system calibrated to your domain. Conservative grading by default — promotion is easy, correcting inflated grades erodes trust.

**Agent Specialization** — Separate agents for analysis and verification. The analyst produces findings. The verifier tests them independently. Separation of concerns keeps the corpus honest.

**Editorial Watchlist** — Institutional memory of identified errors. When a mistake is found, it's documented with detection patterns and swept across the corpus. The immune system learns.

**Structured Output** — Machine-readable results blocks that prevent summarization corruption. The block is authoritative; anything that contradicts it is a reporting error.

## The Name

LIOTHIL is a word in the Sacred Language of Damanhur, a spiritual community in northern Italy. It means "The Helper" — the first of five falcon entities described in the tradition as spirits of divine assistance that descend when an operation is properly established.

The research environment that produced LIOTHIL — the Falco Trump Isopsephy Project — is a computational analysis of Sacred Language inscriptions on a Continental Tarot deck. LIOTHIL carries none of that domain content. It carries only the architectural wisdom: the patterns that made the research rigorous, the operational discipline that prevented errors, the structural decisions that let the work scale.

## Security

If you're contributing to LIOTHIL itself, run `pip install pre-commit && pre-commit install` to activate [gitleaks](https://github.com/gitleaks/gitleaks) secret scanning. See `SECURITY.md` for the disclosure policy.

## Requirements

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed and configured
- A research domain you care about
- Willingness to answer questions about your sources and evidence standards

## License

MIT. Take the architecture. Build something with it. The patterns want to propagate.

---

*Built by Timothy Paul Bielec and Meridian. February 2026.*
*Promptcrafted LLC — [promptcrafted.com](https://promptcrafted.com)*
