import streamlit as st
from pathlib import Path
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

@st.cache_resource
def configure_sql_agent(db_uri):
    llm = OpenAI(
        openai_api_key=openai_api_key, temperature=0, streaming=True
    )

    db = SQLDatabase.from_uri(
        database_uri=db_uri,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return agent

# User inputs
radio_opt = ['Use sample database - Chinook.db','Connect to your SQL database']
selected_opt = st.sidebar.radio(
    label='Choose suitable option',
    options=radio_opt
)
if radio_opt.index(selected_opt) == 1:
    db_uri = st.sidebar.text_input(
        label='Database URI',
        placeholder='mysql://user:pass@hostname:port/db'
    )
else:
    db_filepath = (Path(__file__).parent / "Chinook.db").absolute()
    db_uri = f"sqlite:////{db_filepath}"

openai_api_key = st.sidebar.text_input(
    label="OpenAI API Key",
    type="password",
    value=st.session_state['openai_api_key'] if 'openai_api_key' in st.session_state else ''
)

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()
else:
    st.session_state['openai_api_key'] = openai_api_key


# Setup agent
agent = configure_sql_agent(db_uri)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
