import os
import json
import json
from unicodedata import category
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from datetime import date
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt import ToolExecutor
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.utils.function_calling import convert_to_openai_function
import json
from ...base_classes.agent_base import LangchainAgent


class LangchainOLlamaAgent(LangchainAgent):

    def get_meta_data() -> list[dict[str, str]]:
        pass

    def run(query: str) -> str:
        return super().run()


class schema_caldav_task_mod(BaseModel):
    task_number: int = Field(
        description="Index to identify a specific task from the todo list, which need to be updated"
    )
    properties: dict = Field(
        description="A dictionary of all to-be-changed properties and their values"
    )


@tool("modify_task", args_schema=schema_caldav_task_mod, return_direct=True)
def modify_task(num: int, properties: dict) -> bool:
    """Update an existing task of the todo list"""
    print(properties)
    return True


class schema_caldav_task_cre(BaseModel):
    properties: dict = Field(
        description="""{
    "class": "optional - classification if the task is confidencial or not",
    "description": "mandatory - short description or list of subtasks written in markdown format",
    "priority": "optional - 1 to 5. 1 is the highest priority",
    "due": "optional - use the following format 2023-10-04 14:33:05+00:00",
    "summary":"mandatory - this is the title of the task. Use a meaningful name"
}"""
    )


@tool("create_todo_task", args_schema=schema_caldav_task_cre, return_direct=True)
def modify_task(properties: dict) -> bool:
    """Create a new task on the todo list"""
    print(properties)
    return True


@tool
def get_date_today() -> str:
    """provide the current date as iso format"""
    return date.today().strftime("%A") + " " + date.today().isoformat()


@tool
def final_answer(response: str) -> str:
    """Use this tool to provide your answer to the user if you have the final answer"""
    return response


@tool
def repl_tool(codestring: str, filename: str) -> str:
    "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
    save_to_file(codestring, filename)
    return python_repl.run(codestring)


def save_to_file(text: str, filename: str) -> str:
    # Open the file in write mode ('w')
    with open(filename, "w") as f:
        # Write the string to the file
        f.write(text)
    return "sucessfully save to " + filename


if __name__ == "__main__":
    python_repl = PythonREPL()

    tools = [modify_task, final_answer, repl_tool]
    functions = [convert_to_openai_function(t) for t in tools]
    model = OllamaFunctions(
        model="llama2",
        temperature=0.0,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Bind the function to the model
    model = model.bind(functions=functions)

    result = model.invoke(
        """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        Which date is Saturday next week?
        Save the .py file with a meaningful name"""
    )

    tool_executor = ToolExecutor(tools)

    action = ToolInvocation(
        tool=result.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(result.additional_kwargs["function_call"]["arguments"]),
    )

    print(action)
    result = tool_executor.invoke(action)
    print(result)
