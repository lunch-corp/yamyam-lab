import argparse
import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.tools.google_drive import GoogleDriveManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, choices=["node2vec", "metapath2vec", "graphsage"]
    )
    parser.add_argument(
        "--download_file_type", type=str, choices=["candidates", "models"]
    )
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--file_id", type=str, required=False)
    parser.add_argument(
        "--credential_file_path_from_gcloud_console", type=str, required=False
    )
    parser.add_argument("--reusable_token_path", type=str, required=False)
    return parser.parse_args()


def main(args: ArgumentParser.parse_args):
    reuse_auth_info = True if args.reusable_token_path else False
    if not args.latest and args.file_id is None:
        raise ValueError("file_id should be given when latest option is set as False")
    if args.latest and args.file_id is not None:
        raise ValueError(
            "One of latest and file_id argument should be given, but got both of them"
        )
    manager = GoogleDriveManager(
        credential_file_path_from_gcloud_console=args.credential_file_path_from_gcloud_console,
        reusable_token_path=args.reusable_token_path,
        reuse_auth_info=reuse_auth_info,
    )
    manager.download_result(
        model_name=args.model_name,
        latest=args.latest,
        download_file_type=args.download_file_type,
        file_id=args.file_id,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
