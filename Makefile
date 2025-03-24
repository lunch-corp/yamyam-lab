help:
	@echo "Usage:"
	@echo "  make lint   - 코드 린트 확인"
	@echo "  make test   - pytest 실행"



lint:
	poetry run ruff format .
	@echo "코드 린트 문제 확인 중..."
	@poetry run ruff check . > /tmp/ruff_output.txt || { \
		cat /tmp/ruff_output.txt; \
		echo "\n=====================================================";\
		echo "위에 표시된 문제들을 자동으로 수정하시겠습니까? [y/N] "; \
		read -r response; \
		if [ "$$response" = "y" ] || [ "$$response" = "Y" ]; then \
			echo "\n문제 수정 중..."; \
			poetry run ruff check . --unsafe-fixes --fix; \
			echo "\n수정이 완료되었습니다. 변경사항을 확인하고 테스트를 실행해보세요."; \
		else \
			echo "\n수정이 취소되었습니다."; \
			exit 1; \
		fi; \
	}
	@grep -q "All checks passed" /tmp/ruff_output.txt && { \
		cat /tmp/ruff_output.txt; \
		echo "\n모든 검사가 통과되었습니다. 수정이 필요하지 않습니다."; \
	}
	@rm -f /tmp/ruff_output.txt

test:
	poetry run pytest
