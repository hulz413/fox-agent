# fox-agent

An AI agent framework for learning and building practical agent workflows.

## Quickstart

1. Install

```bash
pip install -e .
```

2. Create a .env file with the following variables

```env
FOX_AGENT_CHAT_API_KEY=your_api_key
FOX_AGENT_CHAT_BASE_URL=your_base_url
FOX_AGENT_CHAT_MODEL=your_model
```

3. Run interactive chat

```bash
fox-agent
```

4. Run a single prompt

```bash
fox-agent -p "What is an AI agent?"
```

5. Run with stdin input

```bash
cat log.txt | fox-agent -p "Analyze this log"
```
