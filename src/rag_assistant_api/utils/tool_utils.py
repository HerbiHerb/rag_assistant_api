import os
from google_auth_oauthlib.flow import InstalledAppFlow


def interactive_authentication(scopes: list[str]):
    flow = InstalledAppFlow.from_client_secrets_file(
        os.getenv("GMAIL_CREDENTIALS_FP"), scopes
    )
    creds = flow.run_local_server(port=0)
    return creds
