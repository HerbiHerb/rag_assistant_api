import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from email.message import EmailMessage
from googleapiclient.discovery import build
import base64


def interactive_authentication(scopes: list[str]):
    flow = InstalledAppFlow.from_client_secrets_file(
        os.getenv("GMAIL_CREDENTIALS_FP"), scopes
    )
    creds = flow.run_local_server(port=0)
    return creds


def send_mail(
    subject: str,
    message: str,
    creds: Credentials,
):
    service = build("gmail", "v1", credentials=creds)
    email_message = EmailMessage()
    email_message.set_content(message)

    email_address = os.getenv("GMAIL_ADDRESS")
    if not email_address:
        raise ValueError("Environment variable GMAIL_ADDRESS is not set.")

    email_message["To"] = email_address
    email_message["From"] = email_address
    email_message["Subject"] = subject

    encoded_message = base64.urlsafe_b64encode(email_message.as_bytes()).decode()

    send_message = (
        service.users()
        .messages()
        .send(userId="me", body={"raw": encoded_message})
        .execute()
    )
    print(f'Message Id: {send_message["id"]}')
