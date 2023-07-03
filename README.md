# 🦜️🔗 LangChain 🤝 Streamlit agent examples

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/langchain-ai/streamlit-agent?quickstart=1)

This repository contains reference implementations of various LangChain agents as Streamlit apps including:

- `basic_streaming.py`: How to do streaming with a simple app using `langchain.chat_models.ChatOpenAI`
- `mrkl_demo.py`: An agent that replicates the [MRKL demo](https://python.langchain.com/docs/modules/agents/how_to/mrkl)
- `minimal_agent.py`: A minimal agent with search (requires setting `OPENAI_API_KEY` env to run)
- `search_and_chat.py`: A search-enabled chatbot that remembers chat history

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

- [ ] Chat QA over docs
- [ ] SQL agent
