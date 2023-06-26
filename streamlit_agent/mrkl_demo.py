from pathlib import Path

import streamlit as st

from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.agents import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler

from callbacks.capturing_callback_handler import playback_callbacks
from clear_results import with_clear_container

DB_PATH = (Path(__file__).parent / "Chinook.db").absolute()

SAVED_SESSIONS = {
    "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?": "leo.pickle",
    "What is the full name of the artist who recently released an album called "
    "'The Storm Before the Calm' and are they in the FooBar database? If so, what albums of theirs "
    "are in the FooBar database?": "alanis.pickle",
}

st.set_page_config(page_title="MRKL", page_icon="🦜", layout="wide")

"# 🦜🔗 MRKL"

"""
This Streamlit app showcases a LangChain agent that replicates the
[MRKL chain](https://arxiv.org/abs/2205.00445).

This uses the [example Chinook database](https://github.com/lerocha/chinook-database).
To set it up follow the instructions [here](https://database.guide/2-sample-databases-sqlite/),
placing the .db file in the same directory as this app.

"""

# Setup credentials in Streamlit
user_openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", type="password", help="Set this to run your own custom questions."
)
user_serpapi_api_key = st.sidebar.text_input(
    "SerpAPI API Key",
    type="password",
    help="Set this to run your own custom questions. Get yours at https://serpapi.com/manage-api-key.",
)

if user_openai_api_key and user_serpapi_api_key:
    openai_api_key = user_openai_api_key
    serpapi_api_key = user_serpapi_api_key
    enable_custom = True
else:
    openai_api_key = "not_supplied"
    serpapi_api_key = "not_supplied"
    enable_custom = False

# StreamlitCallbackHandler configuration
expand_new_thoughts = st.sidebar.checkbox(
    "Expand New Thoughts",
    value=True,
    help="True if LLM thoughts should be expanded by default",
)

collapse_completed_thoughts = st.sidebar.checkbox(
    "Collapse Completed Thoughts",
    value=True,
    help="True if LLM thoughts should be collapsed when they complete",
)

max_thought_containers = st.sidebar.number_input(
    "Max Thought Containers",
    value=4,
    min_value=1,
    help="Max number of completed thoughts to show. When exceeded, older thoughts will be moved into a 'History' expander.",
)

# Tools setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, streaming=True)
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    Tool(
        name="FooBar DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    ),
]

# Initialize agent
mrkl = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# More Streamlit here!
key = "input"
shadow_key = "_input"

if key in st.session_state and shadow_key not in st.session_state:
    st.session_state[shadow_key] = st.session_state[key]

with st.form(key="form"):
    if not enable_custom:
        "Ask one of the sample questions, or enter your API Keys in the sidebar to ask your own custom questions."
    prefilled = st.selectbox("Sample questions", sorted(SAVED_SESSIONS.keys())) or ""
    mrkl_input = ""

    if enable_custom:
        mrkl_input = st.text_input("Or, ask your own question", key=shadow_key)
        st.session_state[key] = mrkl_input
    if not mrkl_input:
        mrkl_input = prefilled
    submit_clicked = st.form_submit_button("Submit Question")

# A hack to "clear" the previous result when submitting a new prompt.
if with_clear_container(submit_clicked):
    question_container = st.empty()
    results_container = st.empty()

    # Create our StreamlitCallbackHandler
    res = results_container.container()
    streamlit_handler = StreamlitCallbackHandler(
        parent_container=res,
        max_thought_containers=int(max_thought_containers),
        expand_new_thoughts=expand_new_thoughts,
        collapse_completed_thoughts=collapse_completed_thoughts,
    )

    question_container.write(f"**Question:** {mrkl_input}")

    # If we've saved this question, play it back instead of actually running LangChain
    # (so that we don't exhaust our API calls unnecessarily)
    if mrkl_input in SAVED_SESSIONS:
        session_name = SAVED_SESSIONS[mrkl_input]
        session_path = Path(__file__).parent / "runs" / session_name
        print(f"Playing saved session: {session_path}")
        answer = playback_callbacks(
            [streamlit_handler], str(session_path), max_pause_time=3
        )
        res.write(f"**Answer:** {answer}")
    else:
        answer = mrkl.run(mrkl_input, callbacks=[streamlit_handler])
        res.write(f"**Answer:** {answer}")
