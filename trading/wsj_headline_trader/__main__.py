"""Allow ``python -m wsj_headline_trader``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
