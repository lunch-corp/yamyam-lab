import logging
import os
import zipfile
from typing import List


def zip_files_in_directory(
    dir_path: str,
    zip_file_name: str,
    allowed_type: List[str],
    logger: logging.Logger,
) -> None:
    # Create a list of files to zip
    files_to_zip = []

    # Loop through the directory to find pickle and parquet files
    for filename in os.listdir(dir_path):
        if filename.endswith(tuple(allowed_type)):
            files_to_zip.append(os.path.join(dir_path, filename))

    # Create the zip file
    with zipfile.ZipFile(os.path.join(dir_path, zip_file_name), "w") as zipf:
        for file in files_to_zip:
            # Add file to the zip (arcname extracts just the filename without the path)
            zipf.write(file, arcname=os.path.basename(file))
            logger.info(f"Added {os.path.basename(file)} to {zip_file_name}")

    logger.info(f"Successfully created {zip_file_name} with {len(files_to_zip)} files")


def unzip_files_in_directory(dir_path: str) -> None:
    extract_path = os.path.dirname(dir_path)

    with zipfile.ZipFile(dir_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"파일이 {extract_path}에 압축 해제되었습니다.")
