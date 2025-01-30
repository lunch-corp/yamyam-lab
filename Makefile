lint:
	poetry run ruff format .
	poetry run ruff check .

test:
	poetry run pytest