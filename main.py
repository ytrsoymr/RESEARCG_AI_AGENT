from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
# Google GenAI API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  


# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=GOOGLE_API_KEY)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})


import json

try:
    # Extract the raw output
    raw_output = raw_response.get("output", "")
    
    # Remove Markdown code block indicators (```json ... ```)
    cleaned_output = raw_output.strip("```json\n").strip("\n```")
    
    # Convert the cleaned JSON string to a Python dictionary
    parsed_output = json.loads(cleaned_output)
    
    # Convert dictionary back to JSON string and parse with Pydantic
    structured_response = parser.parse(json.dumps(parsed_output))
    
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e, "\nRaw Response:", raw_response)
