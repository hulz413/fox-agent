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

## Local semantic embeddings with Ollama

The default `FOX_AGENT_EMBEDDING_PROVIDER=simple` is a lightweight baseline. It hashes tokens into vectors, so it is useful for learning the RAG flow but is not true semantic retrieval.

For local semantic embeddings, run an OpenAI-compatible embedding endpoint with Ollama:

```bash
ollama pull nomic-embed-text
```

Then add these variables to `.env`:

```env
FOX_AGENT_EMBEDDING_PROVIDER=openai
FOX_AGENT_EMBEDDING_BASE_URL=http://localhost:11434/v1
FOX_AGENT_EMBEDDING_API_KEY=ollama
FOX_AGENT_EMBEDDING_MODEL=nomic-embed-text
```

`FOX_AGENT_EMBEDDING_API_KEY` can be any non-empty value for local Ollama. If Ollama is not already running, start it before running `fox-agent`.

After switching embedding models, rebuild the knowledge index. Embeddings from different providers or models have different vector spaces and cannot be mixed in the same index.
