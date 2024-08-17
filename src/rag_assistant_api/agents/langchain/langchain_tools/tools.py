import os
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Tuple, List, Type, Union
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from email.message import EmailMessage
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup
import base64
from langchain_text_splitters import TokenTextSplitter
from ....base_classes.database_handler import DatabaseHandler
from ....base_classes.embedding_base import EmbeddingModel
from ....utils.data_processing_utils import get_embedding


def renew_token():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    scopes = ["https://www.googleapis.com/auth/gmail.modify"]

    flow = InstalledAppFlow.from_client_secrets_file(
        os.getenv("GMAIL_CREDENTIALS_FP"),
        scopes,
    )
    creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open(os.getenv("GMAIL_TOKEN_FP"), "w") as token:
        token.write(creds.to_json())


class DocumentSearchInput(BaseModel):
    query: str = Field(
        description="The query string to search for information which could be in different documents. A general formulation shoukd be used and the query should not contain the name of any document."
    )


class DocumentSearch(BaseTool):
    name = "document_search"
    description = """Useful if you need to search for relevant information to answer the user query.
    Args:
        query: The search query for the vector database. 
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler
    args_schema: Type[BaseModel] = DocumentSearchInput

    def _run(self, query: str) -> Tuple[List[str]]:
        """Use the tool"""
        query_embeddings = get_embedding(query, embedding_model=self.embedding_model)
        vecdb_retr_data = self.database_handler.query(
            embedding=query_embeddings,
            top_k=self.database_handler.db_config.top_k,
        )
        return vecdb_retr_data.meta_data


class DocumentFilterSearchInput(BaseModel):
    search_string: str = Field(
        description="The search string to search for relevant information in the vector database. "
    )
    document_name: str = Field(
        description="The name of the document if the user has mentioned it in the search query."
    )


class DocumentFilterSearch(BaseTool):
    name = "document_filter_search"
    description = """Use this tool if you need to search for relevant information inside a specific document to answer the user query.
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler
    args_schema: Type[BaseModel] = DocumentFilterSearchInput

    def _run(self, search_string: str, document_name: str) -> Tuple[List[str]]:
        """Use the tool"""
        query_embeddings = get_embedding(
            search_string, embedding_model=self.embedding_model
        )
        vecdb_retr_data = self.database_handler.query(
            embedding=query_embeddings,
            top_k=self.database_handler.db_config.top_k,
        )
        return vecdb_retr_data.meta_data


class GetNewEmails(BaseTool):
    name = "get_new_emails"
    description = """Use this tool if the user wants that you check if he has new e-mails in his mailbox.
    """

    def _to_args_and_kwargs(self, tool_input: Union[str, dict]) -> Tuple[Tuple, dict]:
        return (), {}

    def _run(self) -> Tuple[List[str]]:
        """Use the tool"""
        scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(os.getenv("GMAIL_TOKEN_FP")):
            creds = Credentials.from_authorized_user_file(
                os.getenv("GMAIL_TOKEN_FP"), scopes
            )
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                # renew_token()
                # creds = Credentials.from_authorized_user_file(
                #     os.getenv("GMAIL_TOKEN_FP"), scopes
                # )
                pass
            else:
                renew_token()
                flow = InstalledAppFlow.from_client_secrets_file(
                    os.getenv("GMAIL_CREDENTIALS_FP"), scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(os.getenv("GMAIL_TOKEN_FP"), "w") as token:
                token.write(creds.to_json())

            # Filter and get the IDs of the message I need.
            # I'm just filtering messages that have the label "UNREAD"
        combined_email_data = []
        try:
            service = build("gmail", "v1", credentials=creds)
            results = (
                service.users()
                .messages()
                .list(userId="me", labelIds=["INBOX"], q="is:unread")
                .execute()
            )
            messages = results.get("messages", [])
            if not messages:
                print("You have no New Messages.")
            else:
                message_count = 0
                text_splitter = TokenTextSplitter(chunk_size=700, chunk_overlap=0)

                for message in messages:
                    message_text = ""
                    msg = (
                        service.users()
                        .messages()
                        .get(userId="me", id=message["id"])
                        .execute()
                    )
                    message_count = message_count + 1
                    message_text += f"##### E-Mail Nr. {message_count} #######\n\n"
                    email_data = msg["payload"]["headers"]
                    for values in email_data:
                        name = values["name"]
                        if name == "From":
                            from_name = values["value"]
                            message_text += f"From: {from_name}\n"
                            print(from_name)
                            subject = [
                                j["value"] for j in email_data if j["name"] == "Subject"
                            ]
                            print(subject)
                            message_text += f"Subject: {subject}\n"

                    if "parts" in msg["payload"]:
                        for p in msg["payload"]["parts"]:
                            if p["mimeType"] == "text/plain":
                                email_text = base64.urlsafe_b64decode(
                                    p["body"]["data"]
                                ).decode("utf-8")
                                texts = text_splitter.split_text(email_text)
                                message_text += texts[0]
                                combined_email_data.append({"text": message_text})
                                break
                            elif p["mimeType"] == "text/html":
                                data = base64.urlsafe_b64decode(
                                    p["body"]["data"]
                                ).decode("utf-8")
                                email_text = BeautifulSoup(data, "html.parser")
                                email_text = email_text.text
                                texts = text_splitter.split_text(email_text)
                                message_text += texts[0]
                                combined_email_data.append({"text": message_text})
                                break
                    else:
                        data = base64.urlsafe_b64decode(
                            msg["payload"]["body"]["data"]
                        ).decode("utf-8")
                        htmlParse = BeautifulSoup(data, "html.parser")
                        html_text = htmlParse.text
                        message_text += f"E-Mail-Text:\n{html_text}"
                        combined_email_data.append({"text": message_text})
            return combined_email_data
        except HttpError as error:
            # TODO(developer) - Handle errors from gmail API.
            print(f"An error occurred: {error}")
            return []


class SendEmail(BaseTool):
    name = "send_email"
    description = """Use this function to send an email with a subject and a message to the user email address.
    """

    def _run(self, subject: str, message: str) -> Tuple[List[str]]:
        """Use the tool"""
        scopes = ["https://www.googleapis.com/auth/gmail.modify"]
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", scopes)
        # If there are no (valid) credentials available, let the user log in.

        try:
            service = build("gmail", "v1", credentials=creds)
            email_message = EmailMessage()

            email_message.set_content(message)

            email_message["To"] = "dennisherbrik1988@gmail.com"
            email_message["From"] = "dennisherbrik1988@gmail.com"
            email_message["Subject"] = subject

            # encoded message
            encoded_message = base64.urlsafe_b64encode(
                email_message.as_bytes()
            ).decode()

            create_message = {"raw": encoded_message}
            # pylint: disable=E1101
            send_message = (
                service.users()
                .messages()
                .send(userId="me", body=create_message)
                .execute()
            )
            print(f'Message Id: {send_message["id"]}')
        except HttpError as error:
            print(f"An error occurred: {error}")
            send_message = None
        return []


class SQLQuerySearch(BaseTool):
    name = "sql_query"
    description = """"Useful if you need to get data from an sql database. The data table is called 'cp_dwh'.

    Args:
        sql_query: The search query for the vector database. 
    """
    embedding_model: EmbeddingModel
    database_handler: DatabaseHandler

    def _run(self, sql_query: str) -> Tuple[List[str]]:
        """Use the tool"""
        test = 0
        return "SQL-Answer"
