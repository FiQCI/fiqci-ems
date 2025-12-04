# Contribution guide

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fiqci-ems
```

2. Sync dependencies and create virtual environment (uv handles this automatically):
```bash
uv sync # or uv venv && uv pip install -e .[dev]
```

1. Install pre-commit hooks:
```bash
uv run pre-commit install
```

## Code Quality Tools

### Pre-commit
Pre-commit hooks will run automatically on `git commit`. To run manually on all files:

```bash
pre-commit run --all-files
```

### Formatting and Linting with Ruff

Format code:
```bash
ruff format src/ tests/
```

Lint and auto-fix issues:
```bash
ruff check --fix src/ tests/
```

### Type Checking with Pyrefly
Run type checking:
```bash
uv run pyrefly check
```

Or if pyrefly is in your PATH:
```bash
pyrefly check
```

## Run the Tests

Run all tests:
```bash
uv run pytest tests/
```

Run tests with coverage:
```bash
uv run pytest tests/ --cov=src/fiqci/ems --cov-report=term-missing
```
