import os
from langchain.pydantic_v1 import BaseModel, Field
from pydantic.v1 import root_validator
from langchain.tools import BaseTool
from typing import Tuple, List, Type, Union, Any
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from email.message import EmailMessage
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup
import base64
import requests
from langchain_text_splitters import TokenTextSplitter
from langchain.utils import get_from_dict_or_env
from ....local_database.database_models import Conversation
from ....base_classes.database_handler import DatabaseHandler
from ....base_classes.embedding_base import EmbeddingModel
from ....utils.data_processing_utils import get_embedding
from ....utils.tool_utils import interactive_authentication, send_mail


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
        # TODO: - Add a query reformulation step with an llm to get more variation in the search for relevant documents
        #       - Add a reranker on top of the vector search
        #       - For the most relevant chunk take the text of the whole chapter in which it appears
        #       - Take chunks from other documents al well (to get more variation)
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
    description = """Use this tool if the user wants you to check if there are new emails in the mailbox."""

    def _to_args_and_kwargs(self, tool_input: Union[str, dict]) -> Tuple[Tuple, dict]:
        return (), {}

    def _run(self) -> List[dict]:
        """Use the tool"""
        scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        creds = None
        token_path = os.getenv("GMAIL_TOKEN_FP")

        if not token_path:
            raise ValueError("Environment variable GMAIL_TOKEN_FP is not set.")

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    creds = interactive_authentication(scopes)
            else:
                creds = interactive_authentication(scopes)

            with open(token_path, "w") as token:
                token.write(creds.to_json())

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
                    message_count += 1
                    message_text += f"##### E-Mail Nr. {message_count} #######\n\n"
                    email_data = msg["payload"]["headers"]

                    from_name = next(
                        (v["value"] for v in email_data if v["name"] == "From"),
                        "Unknown",
                    )
                    subject = next(
                        (v["value"] for v in email_data if v["name"] == "Subject"),
                        "No Subject",
                    )

                    message_text += f"From: {from_name}\nSubject: {subject}\n"

                    if "parts" in msg["payload"]:
                        for part in msg["payload"]["parts"]:
                            if part["mimeType"] == "text/plain":
                                email_text = base64.urlsafe_b64decode(
                                    part["body"]["data"]
                                ).decode("utf-8")
                                texts = text_splitter.split_text(email_text)
                                message_text += texts[0] if texts else ""
                                combined_email_data.append({"text": message_text})
                                break
                            elif part["mimeType"] == "text/html":
                                data = base64.urlsafe_b64decode(
                                    part["body"]["data"]
                                ).decode("utf-8")
                                email_text = BeautifulSoup(data, "html.parser").text
                                texts = text_splitter.split_text(email_text)
                                message_text += texts[0] if texts else ""
                                combined_email_data.append({"text": message_text})
                                break
                    else:
                        data = base64.urlsafe_b64decode(
                            msg["payload"]["body"]["data"]
                        ).decode("utf-8")
                        html_text = BeautifulSoup(data, "html.parser").text
                        message_text += f"E-Mail-Text:\n{html_text}"
                        combined_email_data.append({"text": message_text})

            return combined_email_data
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []


class SendEmailInput(BaseModel):
    subject: str = Field(description="The subject of the email you want to send.")
    message: str = Field(description="The actual message you want to send.")


class SendEmail(BaseTool):
    name = "send_email"
    description = """Use this function to send an email with a subject and a message to the user's email address."""
    args_schema: Type[BaseModel] = SendEmailInput

    def _run(self, subject: str, message: str) -> list:
        """Use the tool"""
        scopes = ["https://www.googleapis.com/auth/gmail.modify"]
        creds = None
        token_path = os.getenv("GMAIL_TOKEN_FP")

        if not token_path:
            raise ValueError("Environment variable GMAIL_TOKEN_FP is not set.")

        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"Error refreshing credentials: {e}")
                    creds = interactive_authentication(scopes)
            else:
                creds = interactive_authentication(scopes)

            with open(token_path, "w") as token:
                token.write(creds.to_json())
        for _ in range(2):
            try:
                send_mail(subject=subject, message=message, creds=creds)
                break
            except HttpError as error:
                print(f"An error occurred: {error}")
                creds = interactive_authentication(scopes)
                with open(token_path, "w") as token:
                    token.write(creds.to_json())
        return []


class GoogleSearchInput(BaseModel):
    search_term: str = Field(description="The search term of the google search.")


class GoogleSearch(BaseTool):
    name = "google_search_tool"
    description = """Use this tool if you need to search for current events, famous people or if the user wants you to do a google search.
    """
    args_schema: Type[BaseModel] = GoogleSearchInput
    search_engine: Any
    google_api_key: str = None
    google_cse_id: str = None
    k: int = 4
    siterestrict: bool = False

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        cse = self.search_engine.cse()
        if self.siterestrict:
            cse = cse.siterestrict()
        res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
        return res.get("items", [])

    @root_validator()
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        service = build("customsearch", "v1", developerKey=google_api_key)
        values["search_engine"] = service

        return values

    def _get_site_content(self, url: str) -> str:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0)
            texts = text_splitter.split_text(soup.text)
            return texts[0]
        except Exception as e:
            print(e)
            return ""

    def _run(self, search_term: str) -> List[dict]:
        """Run query through GoogleSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._google_search_results(search_term, num=self.k)
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_result["text"] = self._get_site_content(result["link"])
            metadata_results.append(metadata_result)

        return metadata_results


class YouTubeSearchInput(BaseModel):
    search_term: str = Field(description="The search term for YouTube videos.")


class YouTubeSearch(BaseTool):
    name = "youtube_search_tool"
    description = """Use this tool if the user wants you to search for some relevant videos on YouTube. At the beginning of your answer mention the video_id please, so that the user can reference a video."""
    args_schema: Type[BaseModel] = YouTubeSearchInput
    search_engine: Any = None
    youtube_api_key: str

    def _run(self, search_term: str) -> List[dict]:
        api_service_name = "youtube"
        api_version = "v3"
        try:
            youtube = build(
                api_service_name, api_version, developerKey=self.youtube_api_key
            )
            request = youtube.search().list(
                part="id,snippet",
                type="video",
                q=search_term,
                videoDuration="medium",
                videoDefinition="high",
                maxResults=4,
                fields="items(id(videoId),snippet(publishedAt,channelId,channelTitle,title,description))",
            )
            response = request.execute()
        except Exception as e:
            print(f"Error in YouTubeSearch tool: {e}")
            return []

        all_transcriptions = []

        for item in response.get("items", []):
            video_id = item["id"]["videoId"]
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=["de", "en"]
                )
            except Exception as e:
                print(f"Could not extract transcript for video {video_id}: {e}")
                continue

            whole_text = " ".join([t["text"] for t in transcript])
            all_transcriptions.append({"video_id": video_id, "text": whole_text})

        return all_transcriptions


class FetchLastSources(BaseTool):
    name = "fetch_last_sources"
    description = """Use this tool if the user has a follow up question regrding your answer of the last question. You can see the same sourced again to answer 
    the question.
    """
    user_id: int

    def _to_args_and_kwargs(self, tool_input: Union[str, dict]) -> Tuple[Tuple, dict]:
        return (), {}

    def _run(self, opt_str: str = None) -> Tuple[List[str]]:
        """Use the tool"""
        conv_id = Conversation.get_latest_conversation_id(user_id=self.user_id)
        meta_data = Conversation.get_meta_data_for_chat_messages(conv_id=conv_id)
        return meta_data[-1]


class PlayYouTubeVideoInput(BaseModel):
    video_id: str = Field(description="The id of the video to play.")


class PlayYouTubeVideo(BaseTool):
    name = "play_youtube_video"
    description = """Use this tool if the user wants you to play a specific video.
    """

    def _run(self, video_id: str) -> Tuple[List[str]]:
        """Use the tool"""
        return [{"play_video": video_id}]


class ChangeCompetenceLevelInput(BaseModel):
    level: str = Field(
        description="The competence level to switch to. Can be either 3 or 4."
    )


class ChangeCompetenceLevel(BaseTool):
    name = "play_youtube_video"
    description = """Use this tool if the user wants you to switch to another competence level. You have two competence levels (3 or 4).
    """

    def _run(self, level: str) -> Tuple[List[str]]:
        """Use the tool"""
        return [{"text": "changed competence level"}]


class SaveDataInput(BaseModel):
    data: str = Field(description="The data to save for the user")


class SaveData(BaseTool):
    name = "play_youtube_video"
    description = """Use this tool if the user wants you to save some date for the user. This could be previous answers you gave to the user or raw input data from the user.
    """

    def _run(self, data: str) -> Tuple[List[str]]:
        """Use the tool"""
        return [{"text": "successfully saved the data"}]


class ChartDrawerInput(BaseModel):
    xml_code: str = Field(description="The xml code to generate the chart.")


class ChartDrawer(BaseTool):
    name = "play_youtube_video"
    description = """Use this tool if the user wants a chart out of a given text. This could be used if the text describes a process with connected items or contains connected dependencies.
    """

    def _run(self, xml_code: str) -> Tuple[List[str]]:
        """Use the tool"""
        return [{"text": "successfully saved the data"}]
