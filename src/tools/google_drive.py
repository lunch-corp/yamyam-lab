import os
from pathlib import Path
from typing import Dict

import gdown
import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()

except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install python-dotenv")

ABS_PATH = os.path.join(os.path.dirname(__file__), "../..")
FILES = ["diner", "review", "reviewer", "category"]


def get_env_var(var_name: str) -> str:
    """환경 변수에서 값을 가져옵니다."""
    value = os.getenv(var_name)

    if not value:
        raise ValueError(f"환경 변수 {var_name}가 설정되지 않았습니다.")

    return value


def load_drive_config():
    """Google Drive 설정을 로드합니다."""
    config_path = Path(os.path.join(ABS_PATH, "config/data/google_drive.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_from_drive(file_type: str) -> str:
    """
    Google Drive에서 파일을 다운로드합니다.

    Args:
        file_type: 'diner' 또는 'reviewer'
    """
    # 환경 변수 이름 생성
    file_id_var = f"{file_type.upper()}_FILE_ID"

    config = load_drive_config()
    local_path = os.path.join(ABS_PATH, config["local_paths"].get(file_type))

    # 환경 변수에서 값 가져오기
    file_id = get_env_var(file_id_var)

    if not file_id or not local_path:
        raise ValueError(f"Invalid file type: {file_type}")

    # 로컬 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # 파일이 이미 존재하는지 확인
    if os.path.exists(local_path):
        print(f"{file_type} 파일이 이미 존재합니다: {local_path}")
        return local_path

    # Google Drive URL 생성
    url = f"https://drive.google.com/uc?id={file_id}"

    # 파일 다운로드
    try:
        gdown.download(url, local_path, quiet=False)
        print(f"{file_type} 파일 다운로드 완료: {local_path}")
        return local_path

    except Exception as e:
        print(f"다운로드 실패: {str(e)}")
        raise


def ensure_data_files() -> Dict[str, str]:
    """필요한 모든 데이터 파일이 존재하는지 확인하고 없으면 다운로드합니다."""
    paths = {}

    for file_type in FILES:
        paths[file_type] = download_from_drive(file_type)

    return paths
