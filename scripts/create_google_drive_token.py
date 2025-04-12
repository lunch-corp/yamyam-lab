import argparse
import json
import os


def create_token_file(
    token: str, refresh_token: str, client_id: str, client_secret: str, output_path: str
):
    """
    Creates a token.json file from a token value.
    """
    token_data = {
        "token": token,
        "refresh_token": refresh_token,
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": client_id,
        "client_secret": client_secret,
        "scopes": ["https://www.googleapis.com/auth/drive"],
        "universe_domain": "googleapis.com",
        "account": "",
        "expiry": "2025-03-01T12:22:46.680803Z",  # intentional expired date
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write token file
    with open(output_path, "w") as token_file:
        json.dump(token_data, token_file)

    print(f"Token file created at {output_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create token file and initialize Google Drive Manager"
    )
    parser.add_argument("--token", required=True)
    parser.add_argument("--refresh_token", required=True)
    parser.add_argument("--client_id", required=True)
    parser.add_argument("--client_secret", required=True)
    parser.add_argument("--output_path", default="credentials/token.json")

    return parser.parse_args()


def main():
    args = parse_arguments()
    create_token_file(
        token=args.token,
        refresh_token=args.refresh_token,
        client_id=args.client_id,
        client_secret=args.client_secret,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
