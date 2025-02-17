import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict

import gdown

try:
    from dotenv import load_dotenv

    load_dotenv()

except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install python-dotenv")

ABS_PATH = Path(__file__).resolve().parents[2]
data_dir = os.path.join(ABS_PATH, "data")

def get_env_var(var_name: str) -> str:
    """환경 변수에서 값을 가져옵니다."""
    value = os.getenv(var_name)

    if not value:
        raise ValueError(f"환경 변수 {var_name}가 설정되지 않았습니다.")

    return value

def ensure_data_files() -> Dict[str, str]:
    """
    필요한 모든 데이터 파일이 존재하는지 확인하고 없으면 다운로드합니다.
    
    Returns:
        Dict[str, str]: 파일명과 경로를 담은 딕셔너리
    """
    if not check_required_files():
        return download_and_extract_drive_folder()
    else:
        print('기존 data가 존재합니다. 파일 경로를 반환합니다.')
        return get_file_paths(data_dir)

def check_required_files() -> bool:
    """
    data 폴더에 필요한 CSV 파일들이 있는지 확인합니다.
    
    Returns:
        bool: 모든 필요한 파일이 존재하면 True, 아니면 False
    """
    required_files = ['diner.csv', 'review.csv', 'reviewer.csv']
    data_dir = os.path.join(ABS_PATH, "data")
    
    if not os.path.exists(data_dir):
        return False
        
    return all(os.path.exists(os.path.join(data_dir, file)) for file in required_files)

def download_and_extract_drive_folder() -> Dict[str, str]:
    """
    Google Drive에서 데이터셋을 다운로드하고 압축을 해제합니다.
    
    Returns:
        Dict[str, str]: 파일명과 경로를 담은 딕셔너리
    """
    folder_id_var = "DATA_FOLDER_ID"
    zip_path = os.path.join(data_dir, "dataset.zip")
    google_save_dir = os.path.join(data_dir, "google_save_dir")
    
    # data 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    # 필요한 파일들이 이미 존재하는지 확인
    if check_required_files():
        print("필요한 모든 파일이 이미 존재합니다.")
        return get_file_paths(data_dir)
    
    # 환경 변수에서 폴더 ID 가져오기
    folder_id = get_env_var(folder_id_var)
    
    if not folder_id:
        raise ValueError("Google Drive 폴더 ID가 설정되지 않았습니다.")
    
    # Google Drive URL 생성
    url = f"https://drive.google.com/uc?id={folder_id}"
    
    try:
        # ZIP 파일 다운로드
        print(f"Google Drive에서 데이터셋 다운로드 중: {zip_path}")
        gdown.download(url, zip_path, quiet=False)
        
        # 압축 해제
        print(f"압축 해제 중: {zip_path} -> {google_save_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(google_save_dir)
            
        print(f"압축 해제 완료: {google_save_dir}")
        
        # google_save_dir의 파일들을 data로 이동
        if os.path.exists(google_save_dir):
            print("google_save_dir에서 파일 이동 중...")
            # google_save_dir 내의 CSV 파일 정보를 가져옴
            google_files = get_file_paths(google_save_dir)
            
            # 파일들을 data_dir로 이동
            move_files_to_data(google_save_dir, data_dir)
            
            # 이동된 파일들의 경로를 data_dir 기준으로 갱신
            for key, info in google_files.items():
                file_name = os.path.basename(info['file_path'])
                new_path = os.path.abspath(os.path.join(data_dir, file_name))
                google_files[key]['file_path'] = new_path
            
            # google_save_dir 삭제
            shutil.rmtree(google_save_dir)
            print(f"google_save_dir 삭제 완료")
        
        # 파일 경로 반환
        return google_files
        
    except Exception as e:
        print(f"다운로드 또는 압축 해제 실패: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise

def move_files_to_data(source_dir: str, target_dir: str):
    """
    google_save_dir에서 data 디렉토리로 CSV 파일들을 이동시킵니다.
    
    Args:
        source_dir: 소스 디렉토리 경로 (google_save_dir)
        target_dir: 대상 디렉토리 경로 (data)
    """
    # CSV 파일 찾기
    for file_path in Path(source_dir).glob('*.csv'):
        file_name = file_path.name
        base_name = re.sub(r'_\d{8}', '', file_path.stem)  # 날짜 부분 제거
        new_name = f"{base_name}.csv"
        target_path = os.path.join(target_dir, new_name)
        
        # 파일 이동
        shutil.move(str(file_path), target_path)
        print(f"파일 이동 완료: {file_name} -> {new_name}")

def get_file_paths(directory_path: str) -> Dict[str, Dict[str, str]]:
    """
    data 디렉토리의 CSV 파일들의 정보를 반환합니다.
    
    Args:
        directory_path: 데이터 디렉토리 경로
    
    Returns:
        Dict[str, Dict[str, str]]: 파일 정보를 담은 딕셔너리
    """
    result = {}
    path = Path(directory_path)
    
    for file_path in path.glob('*.csv'):
        file_name = file_path.stem
        base_name = re.sub(r'_\d{8}', '', file_name)  # 날짜 부분 제거

        if base_name:
            # 날짜 패턴 찾기 (YYYYMMDD)
            date_match = re.search(r'_(\d{8})', file_name)

            if date_match:
                version = date_match.group(1)
            else:
                # raw 버전 확인
                raw_match = re.search(r'_raw', file_name)
                version = 'raw' if raw_match else 'default'
            
            new_file_name = f"{base_name}{file_path.suffix}"
            new_file_path = file_path.parent / new_file_name
            
            result[base_name] = {
                'version': version,
                'file_path': str(new_file_path.absolute())
            }
    
    return result