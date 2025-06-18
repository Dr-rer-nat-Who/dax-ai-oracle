.PHONY: setup test lint

setup:
bash scripts/setup.sh

lint:
npm run lint

test:
pytest -q
