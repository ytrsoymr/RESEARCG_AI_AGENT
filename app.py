import streamlit as st
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables
load_dotenv()

# Google GenAI API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Define the response structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define Tools
tools = [search_tool, wiki_tool, save_tool]

# Create Agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.title("📚 AI Research Assistant")
st.write("Enter a research topic, and let AI fetch a structured summary for you.")

# User Input
query = st.text_input("🔍 Research Topic", "")

if st.button("Generate Research"):
    if query:
        with st.spinner("🔎 Fetching research..."):
            try:
                raw_response = agent_executor.invoke({"query": query})

                # Extract and clean JSON output
                raw_output = raw_response.get("output", "").strip("```json\n").strip("\n```")
                parsed_output = json.loads(raw_output)
                structured_response = parser.parse(json.dumps(parsed_output))

                # Display Results
                st.subheader("📌 Research Summary")
                st.write(f"**Topic:** {structured_response.topic}")
                st.write(f"**Summary:** {structured_response.summary}")

                st.subheader("📚 Sources")
                for source in structured_response.sources:
                    st.write(f"- {source}")

                st.subheader("🔧 Tools Used")
                st.write(", ".join(structured_response.tools_used))

            except Exception as e:
                st.error(f"Error parsing response: {e}")
    else:
        st.warning("⚠️ Please enter a topic before generating research.")
