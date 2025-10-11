.PHONY: setup lint serve train monitor docker-build docker-run

setup:
	uv sync

lint:
	ruff check .
	black --check .

serve:
	uv run uvicorn main:app --host 0.0.0.0 --port 8000 --access-log

mlflow:
	mlflow ui

mlflow config run:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root file:/Users/{full_path}/customer_churn/mlruns \
		--host 127.0.0.1 \
		--port 5000
