import io
import os
import re
import shutil
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import gdown
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

try:
    from dotenv import load_dotenv

    load_dotenv()

except ModuleNotFoundError:
    raise ModuleNotFoundError("Please install python-dotenv")


ABS_PATH = Path(__file__).resolve().parents[2]
data_dir = os.path.join(ABS_PATH, "data")
ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")


def get_env_var(var_name: str) -> str:
    """환경 변수에서 값을 가져옵니다."""
    value = os.getenv(var_name)

    if not value:
        raise ValueError(f"환경 변수 {var_name}가 설정되지 않았습니다.")

    return value


def download_from_drive() -> Dict[str, str]:
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
            for data_name, file_path in google_files.items():
                file_name = os.path.basename(file_path)
                new_path = os.path.abspath(os.path.join(data_dir, file_name))
                google_files[data_name] = new_path

            # google_save_dir 삭제
            shutil.rmtree(google_save_dir)
            print("google_save_dir 삭제 완료")

        # 파일 경로 반환
        return google_files

    except Exception as e:
        print(f"다운로드 또는 압축 해제 실패: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise


def ensure_data_files() -> Dict[str, str]:
    """
    필요한 모든 데이터 파일이 존재하는지 확인하고 없으면 다운로드합니다.

    Returns:
        Dict[str, str]: 파일명과 경로를 담은 딕셔너리
    """
    if not check_required_files():
        return download_from_drive()
    else:
        print("기존 data가 존재합니다. 파일 경로를 반환합니다.")
        return get_file_paths(data_dir)


def check_required_files() -> bool:
    """
    data 폴더에 필요한 CSV 파일들이 있는지 확인합니다.

    Returns:
        bool: 모든 필요한 파일이 존재하면 True, 아니면 False
    """
    required_files = ["diner.csv", "review.csv", "reviewer.csv"]
    data_dir = os.path.join(ABS_PATH, "data")

    if not os.path.exists(data_dir):
        return False

    return all(os.path.exists(os.path.join(data_dir, file)) for file in required_files)


def move_files_to_data(source_dir: str, target_dir: str):
    """
    google_save_dir에서 data 디렉토리로 CSV 파일들을 이동시킵니다.

    Args:
        source_dir: 소스 디렉토리 경로 (google_save_dir)
        target_dir: 대상 디렉토리 경로 (data)
    """
    # CSV 파일 찾기
    for file_path in Path(source_dir).glob("*.csv"):
        file_name = file_path.name
        base_name = re.sub(r"_\d{8}", "", file_path.stem)  # 날짜 부분 제거
        new_name = f"{base_name}.csv"
        target_path = os.path.join(target_dir, new_name)

        # 파일 이동
        shutil.move(str(file_path), target_path)
        print(f"파일 이동 완료: {file_name} -> {new_name}")


def get_file_paths(directory_path: str) -> Dict[str, str]:
    """
    data 디렉토리의 CSV 파일들의 정보를 반환합니다.

    Args:
        directory_path: 데이터 디렉토리 경로

    Returns:
        Dict[str, Dict[str, str]]: 파일 정보를 담은 딕셔너리
    """
    result = {}
    path = Path(directory_path)

    # 모든 파일 확장자 매칭을 위한 패턴
    for file_path in path.glob("*"):
        # csv, parquet, pkl 파일만 처리
        if file_path.suffix.lower() not in [".csv", ".parquet", ".pkl"]:
            continue

        file_name = file_path.stem
        base_name = re.sub(r"_\d{8}", "", file_name)  # 날짜 부분 제거

        if base_name:
            # TODO: data version 관리
            # 날짜 패턴 찾기 (YYYYMMDD)
            # date_match = re.search(r'_(\d{8})', file_name)

            # if date_match:
            #     version = date_match.group(1)
            # else:
            #     # raw 버전 확인
            #     raw_match = re.search(r'_raw', file_name)
            #     version = 'raw' if raw_match else 'default'

            new_file_name = f"{base_name}{file_path.suffix}"
            new_file_path = file_path.parent / new_file_name

            # TODO: 하드코딩 수정
            if "category" in base_name:
                base_name = "category"

            result[base_name] = str(new_file_path.absolute())
    return result


class AllowedFileType(Enum):
    ZIP = "zip"


class MimeType(Enum):
    ZIP = "application/zip"
    FOLDER = "application/vnd.google-apps.folder"


class GoogleDriveManager:
    SCOPES = ["https://www.googleapis.com/auth/drive"]
    CANDIDATES_FOLDER_ID = "1_-NOoTC-K6aZMJLM4DgNitCGXX0QkbSA"
    CANDIDATE_GENERATOR_MODEL = ["node2vec", "metapath2vec", "graphsage"]
    MAPPING = {AllowedFileType.ZIP: MimeType.ZIP}
    DOWNLOAD_PATH = os.path.join(ROOT_PATH, "candidates")

    """
    Google drive manager enabling various jobs using python client.
    """

    def __init__(
        self,
        credential_file_path_from_gcloud_console: str = None,
        reusable_token_path: str = None,
        reuse_auth_info: bool = True,
    ):
        """
        Initializes manager.

        Args:
            credential_file_path_from_gcloud_console (str): Json file downloaded from gcloud console.
            reusable_token_path (str): After running `_authenticate_and_build_client` with `credential_file_path_from_gcloud_console`,
                `token.json` will be created in `credentials/` folder. If this json file specified next time,
                login authentication is not required.
            reuse_auth_info (bool): Whether reuse auth info or not.
        """
        self.credential_file_path_from_gcloud_console = (
            credential_file_path_from_gcloud_console
        )
        self.reusable_token_path = reusable_token_path
        self.reuse_auth_info = reuse_auth_info

        self._verify_inputs()
        self.service = self._authenticate_and_build_client()

    def _verify_inputs(self):
        if self.reuse_auth_info:
            if self.reusable_token_path is None:
                raise ValueError(
                    "To reuse auth info, reusable token path must be provided."
                )
        else:
            if self.credential_file_path_from_gcloud_console is None:
                raise ValueError(
                    "For initial authentication, original credential.json from gcloud must be provided."
                )

    def _authenticate_and_build_client(self) -> Resource:
        # Check if token.json exists
        if self.reuse_auth_info:
            creds = Credentials.from_authorized_user_file(
                self.reusable_token_path, self.SCOPES
            )
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credential_file_path_from_gcloud_console, self.SCOPES
            )
            creds = flow.run_local_server(port=0)

            # Save the credentials to reuse auth info
            dir_name = os.path.join(ROOT_PATH, "credentials")
            with open(os.path.join(dir_name, "token.json"), "w") as token:
                token.write(creds.to_json())

        return build("drive", "v3", credentials=creds)

    def create_folder(
        self,
        folder_name: str,
        parent_folder_id: str = None,
    ) -> str:
        """
        Create a folder with given name.

        Args:
            folder_name (str): Name of folder.
            parent_folder_id (str): Folder id of parent folder. If specified, creates folder within parent folder.

        Returns (str):
            Id of created folder.
        """
        # create metadata for folder to make
        folder_metadata = {
            "name": folder_name,
            "mimeType": MimeType.FOLDER.value,
        }

        # If parent folder ID is provided, add it to the metadata
        if parent_folder_id:
            folder_metadata["parents"] = [parent_folder_id]

        folder = (
            self.service.files().create(body=folder_metadata, fields="id").execute()
        )

        return folder.get("id")

    def upload_file(
        self,
        file_path: str,
        folder_id: str,
        file_type: AllowedFileType,
    ):
        """
        Upload specified file stored in local directory.
        Currently, only zip file is supported for uploading to google drive.

        Args:
            file_path (str): File path in local directory.
            folder_id (str): Folder id to upload file.
            file_type (AllowedFileType): Type of file to be uploaded.

        Returns (str):
            File id in after uploading.
        """
        file_name = os.path.basename(file_path)

        file_metadata = {
            "name": file_name,
            "parents": [folder_id],  # This puts the file in the folder we just created
        }

        media = MediaFileUpload(
            file_path, mimetype=self.MAPPING.get(file_type).value, resumable=True
        )
        file = (
            self.service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

        return file.get("id")

    def list_files_in_folder(
        self, folder_id: str, mime_type=None
    ) -> List[Dict[str, Any]]:
        """
        List all files in a folder, optionally filtered by MIME type

        Args:
            folder_id (str): ID of the folder to list files from.
            mime_type (str): MIME type to filter by (None for all files).

        Returns:
            List of file metadata dictionaries
        """
        query = f"'{folder_id}' in parents and trashed=false"

        if mime_type:
            query += f" and mimeType='{mime_type}'"

        results = (
            self.service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name, createdTime, modifiedTime, mimeType)",
            )
            .execute()
        )

        return results.get("files", [])

    def download_file(self, file_id: str, download_path: str) -> str:
        """
        Download a file by ID.

        Args:
            file_id (str): ID of the file to download.
            download_path (str): Path where to save the downloaded file.

        Returns (str):
            Path to the downloaded file.
        """
        # Get the file metadata to get the filename
        file_metadata = (
            self.service.files().get(fileId=file_id, fields="name").execute()
        )
        file_name = file_metadata.get("name")

        request = self.service.files().get_media(fileId=file_id)

        file_path = os.path.join(download_path, file_name)

        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%")

        print(f"File downloaded to: {file_path}")
        return file_path

    def download_candidates_result(
        self, model_name: str, latest: bool = True, file_id: str = None
    ):
        """
        Download candidate result specified by candidate generator model name.

        Args:
            model_name (str): Name of candidate generator model.
            latest (bool): Whether download latest result or not.
            file_id (str): If latest is set as False, download candidate result whose
                file_id is equal to `file_id`.

        Returns (str):
            Path to the downloaded file.

        """
        model_folder_id = self._get_model_folder_id(model_name=model_name)

        # get latest file_id
        if latest:
            latest_file_id = self._get_latest_zip_file(model_folder_id).get("id")
        else:
            latest_file_id = file_id

        download_path = os.path.join(self.DOWNLOAD_PATH, model_name)
        os.makedirs(download_path, exist_ok=True)

        return self.download_file(
            file_id=latest_file_id,
            download_path=download_path,
        )

    def upload_candidates_result(self, model_name: str, file_path: str) -> str:
        """
        Uploads candidate result to related google drive directory.

        Args:
            model_name (str): Model name currently under training. This is used when finding folder_id of model.
            file_path (str): File path of zil file to upload to google drive.

        Returns (str):
            Id of uploaded zip file.
        """
        model_folder_id = self._get_model_folder_id(model_name=model_name)
        return self.upload_file(
            file_path=file_path,
            folder_id=model_folder_id,
            file_type=AllowedFileType.ZIP,
        )

    def _get_latest_zip_file(self, folder_id: str) -> Dict[str, Any]:
        """
        Get latest zip file.

        Args:
            folder_id (str): Folder id to search.

        Returns (Dict[str, Any]):
            Latest file sorted by its name.
        """
        files = self.list_files_in_folder(
            folder_id=folder_id,
            mime_type=MimeType.ZIP.value,
        )
        return sorted(files, key=lambda x: x["name"])[-1]

    def _get_model_folder_id(self, model_name: str):
        """
        Get folder id matched with given model_name.

        Args:
            model_name (str): Name of model to find.

        Returns (str):
            Id of folder matched with model_name.
        """
        if model_name not in self.CANDIDATE_GENERATOR_MODEL:
            raise ValueError(
                f"Unsupported model: {model_name}."
                f"Should be one of {self.CANDIDATE_GENERATOR_MODEL}"
            )
        # get model folder id
        files = self.list_files_in_folder(
            folder_id=self.CANDIDATES_FOLDER_ID,
            mime_type=MimeType.FOLDER.value,
        )
        return [file for file in files if file["name"] == model_name][0]["id"]
