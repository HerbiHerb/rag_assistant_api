from pydantic import BaseModel, Field
from typing import List


class AgentData(BaseModel):
    openai_key: str
    # model_name: str
    max_token_number: int
    embedding_model_name: str
    embedding_token_counter: str
    pinecone_key: str
    pinecone_environment: str
    pinecone_index_name: str
    chatmessages_csv_path: str
    listening_sound_path: str


class AgentAnswerData(BaseModel):
    query_msg_idx: int
    final_answer: str = Field(default="")
    function_responses: List[str] = Field(default=[])

    def add_function_response(self, new_response: str) -> None:
        self.function_responses.append(new_response)
