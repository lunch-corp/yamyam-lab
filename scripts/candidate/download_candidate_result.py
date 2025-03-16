import argparse

from src.tools.google_drive import GoogleDriveManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["node2vec", "metapath2vec", "graphsage"])
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--credential_file_path_from_gcloud_console", type=str, required=False)
    parser.add_argument("--reusable_token_path", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    reuse_auth_info = True if args.reusable_token_path else False
    manager = GoogleDriveManager(
        credential_file_path_from_gcloud_console=args.credential_file_path_from_gcloud_console,
        reusable_token_path=args.reusable_token_path,
        reuse_auth_info=reuse_auth_info,
    )
    manager.download_candidates_result(
        model_name=args.model_name,
        download_path="candidates",
        latest=args.latest,
    )