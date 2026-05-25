# Contributing to MBDPO

Thank you for your interest in contributing to **MBDPO**! This is a research codebase
released to accompany our technical report *"Model-Based Diffusion Policy Optimization in
World Models."* We welcome bug reports, questions, and code contributions from
the community.

Because this is a research project maintained alongside other work, we may not
be able to respond to every issue immediately, but we read all of them and
genuinely appreciate your input.

## Ways to Contribute

- **Report a bug** — something not working as described in the paper or README.
- **Ask a question** — about the code, configuration, or how to reproduce a result.
- **Request a feature** — a new task, benchmark, or training option you'd find useful.
- **Submit code** — bug fixes, new benchmarks, documentation improvements, or other enhancements.

## Reporting Bugs

Before opening a new issue, please:

1. Search [existing issues](https://github.com/Edmond1Cheng/MBDPO/issues) to see
   if the problem has already been reported.
2. Make sure you are using the environment described in the README
   (dependency mismatches, especially gym and MuJoCo versions across simulators,
   are a common source of errors).

When opening a bug report, please include:

- A clear description of the problem and the expected behavior.
- The exact command you ran (e.g. `python train.py task=walker-walk model_size=6M`).
- Your environment: OS, Python version, PyTorch and CUDA versions, GPU, and
  which simulator/benchmark you were running.
- The full error message and stack trace (set `HYDRA_FULL_ERROR=1` for a complete trace).
- If relevant, the config or any changes you made.

## Asking Questions

For usage questions or questions about reproducing results, please open an
issue with the `question` label, or start a thread under
[Discussions](https://github.com/Edmond1Cheng/MBDPO/discussions) if enabled.

## Submitting Code (Pull Requests)

1. **Fork** the repository and clone your fork.
2. **Create a branch** for your change:
   ```bash
   git checkout -b fix/short-description
   ```
3. **Make your change.** Please keep pull requests focused on a single concern —
   smaller, well-scoped PRs are much easier to review and merge.
4. **Test that it runs.** Verify that training and evaluation still works on at least one task affected by your change.
5. **Commit** with a clear message describing what changed and why.
6. **Open a pull request** against the `main` branch. In the description, please
   explain:
   - What the change does.
   - Why it is needed (link the related issue if there is one).
   - How you tested it.

We will review your PR and may request changes before merging. Thank you for
your patience during review.

## Code Style

To keep the codebase consistent and easy to read:

- Follow the existing structure and naming conventions used in the repository.
- Match the style of the surrounding code (we broadly follow
  [PEP 8](https://peps.python.org/pep-0008/) for Python).
- Keep configuration changes within the existing config system rather than
  hard-coding values.
- Add brief comments or docstrings for non-obvious logic.

We do not enforce a strict linter, but please make a reasonable effort to keep
diffs clean (avoid unrelated formatting changes in the same PR).

## Adding a New Task or Benchmark

If you would like to add support for a new task or benchmark, please open an
issue first to discuss it. Contributions of this kind are very welcome, but
coordinating in advance helps avoid duplicated effort and keeps the task naming
and config conventions consistent.

## License

By contributing to this repository, you agree that your contributions will be
licensed under the same [MIT License](LICENSE) that covers the project.

---

**Thank you again for helping improve MBDPO!**