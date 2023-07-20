# 🦜️🔗 LangChain 🤝 Streamlit agent examples

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/streamlit-agent?quickstart=1)

This repository contains reference implementations of various LangChain agents as Streamlit apps including:

- `basic_streaming.py`: Simple streaming app with `langchain.chat_models.ChatOpenAI` ([View the app](https://langchain-streaming-example.streamlit.app/))
- `mrkl_demo.py`: An agent that replicates the [MRKL demo](https://python.langchain.com/docs/modules/agents/how_to/mrkl) ([View the app](https://langchain-mrkl.streamlit.app))
- `minimal_agent.py`: A minimal agent with search (requires setting `OPENAI_API_KEY` env to run)
- `search_and_chat.py`: A search-enabled chatbot that remembers chat history ([View the app](https://langchain-chat-search.streamlit.app/))
- `chat_with_documents.py`: Chatbot capable of answering queries by referring custom documents ([View the app](https://langchain-document-chat.streamlit.app/))
- `chat_with_sql_db.py`: Chatbot which can communicate with your database

Apps feature LangChain 🤝 Streamlit integrations such as the
[Callback integration](https://python.langchain.com/docs/modules/callbacks/integrations/streamlit).

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Install git pre-commit hooks
$ poetry shell
$ pre-commit install
```

## Running

```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run streamlit_agent/mrkl_demo.py
```

## Contributing

We plan to add more agent examples over time - PRs welcome

- [ ] SQL agent
