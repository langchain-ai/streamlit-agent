from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler, LangChainTracer
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langsmith import Client
import streamlit as st
from streamlit_feedback import streamlit_feedback

st.set_page_config(page_title="LangChain: Simple feedback", page_icon="🦜")
st.title("🦜 LangChain: Simple feedback")

if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "LANGCHAIN_API_KEY" in st.secrets:
    langsmith_api_key = st.secrets.LANGCHAIN_API_KEY
else:
    langsmith_api_key = st.sidebar.text_input("LangSmith API Key", type="password")

ls_tracer = None
if langsmith_api_key:
    project = st.sidebar.text_input("LangSmith Project", value="default")
    ls_client = Client(api_url="https://api.smith.langchain.com", api_key=langsmith_api_key)
    ls_tracer = LangChainTracer(project_name=project, client=ls_client)

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}


@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(_client, run_id):
    return _client.read_run(run_id).url


def render_message(text, meta, skip_steps=False):
    # Re-draw any preview intermediate steps
    for step in meta.get("intermediate_steps", []):
        if step[0].tool == "_Exception" or skip_steps:
            continue
        with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
            st.write(step[0].log)
            st.write(f"**{step[1]}**")

    # Write the actual response
    st.write(text)

    # Add feedback input
    if "run_id" in meta and langsmith_api_key:
        run_id = meta["run_id"]
        left, right = st.columns([8, 1])
        with left:
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                key=f"feedback_{run_id}",
            )
        run_url = get_run_url(ls_client, run_id)
        right.markdown(f"[View run]({run_url})")
        if feedback:
            ls_client.create_feedback(
                run_id,
                feedback["type"],
                value=feedback["score"],
                comment=feedback.get("text", None),
            )


avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        render_message(msg.content, st.session_state.steps.get(str(idx), {}))

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    tools = [DuckDuckGoSearchRun(name="Search")]
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
    executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        callbacks = [StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)]
        if ls_tracer:
            callbacks.append(ls_tracer)
        response = executor(prompt, callbacks=callbacks, include_run_info=True)
        response_meta = {"intermediate_steps": response["intermediate_steps"]}
        if langsmith_api_key:
            response_meta["run_id"] = response["__run"].run_id
        st.session_state.steps[str(len(msgs.messages) - 1)] = response_meta
        render_message(response["output"], response_meta, skip_steps=True)
