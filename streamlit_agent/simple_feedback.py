from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler, LangChainTracer
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langsmith import Client
import streamlit as st

st.set_page_config(page_title="LangChain: Simple feedback", page_icon="🦜")
st.title("🦜 LangChain: Simple feedback")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
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

for idx, msg in enumerate(msgs.messages):
    with st.chat_message(msg.type):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
                st.write(step[0].log)
                st.write(f"**{step[1]}**")
        st.write(msg.content)


def render_message(text, meta):
    # Re-draw any preview intermediate steps
    for step in meta.get("intermediate_steps", []):
        if step[0].tool == "_Exception":
            continue
        with st.expander(f"✅ **{step[0].tool}**: {step[0].tool_input}"):
            st.write(step[0].log)
            st.write(f"**{step[1]}**")

    # Write the actual response
    st.write(text)

    # Add feedback input
    if "run_id" in meta and langsmith_api_key:
        run_id = meta["run_id"]
        up, down, url = st.columns([1, 1, 12])
        if up.button("👍", key=f"{run_id}_up"):
            ls_client.create_feedback(run_id, "thumbs_up", score=True)
        if down.button("👎", key=f"{run_id}_down"):
            ls_client.create_feedback(run_id, "thumbs_down", score=True)
        run_url = ls_client.read_run(run_id).url
        url.markdown(f"""[View run in LangSmith]({run_url})""")


if "messages" not in st.session_state or st.sidebar.button("Reset conversation history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
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
        st.write(response["output"])
        response_meta = {"intermediate_steps": response["intermediate_steps"]}
        if langsmith_api_key:
            response_meta["run_id"] = response["__run"].run_id,
        st.session_state.steps[str(len(msgs.messages) - 1)] = response_meta
        render_message(response["output"], response_meta)
