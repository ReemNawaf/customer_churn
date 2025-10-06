.PHONY: setup lint serve train monitor docker-build docker-run

setup:
	uv sync

lint:
	ruff check .
	black --check .

serve:
	uv run uvicorn churn.api.app:app --host 0.0.0.0 --port 8000 --access-log
