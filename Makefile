help:
	@echo "Usage:"
	@echo "  make lint   - 코드 린트 확인"
	@echo "  make test   - pytest 실행"



lint:
	poetry run ruff format .
	poetry run ruff check .

test:
	poetry run pytest