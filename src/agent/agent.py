from __future__ import annotations

from typing import Literal
from src.agent.config import AgentConfig
from src.llm.client import LLMClient
from src.llm.schemas import LLMResponse, Message
from src.llm.embedding_provider import EmbeddingProvider
from src.llm.chat_provider import ChatProvider
from src.llm.openai_chat_provider import OpenAIChatProvider
from src.runtime.session import ChatSession
from src.memory.json_store import JsonMemoryStore
from src.planning.planner import Planner
from src.tools.builtins import build_builtin_tools
from src.tools.registry import ToolRegistry
from src.tools.schemas import ToolDefinition
from src.knowledge.chunker import TextChunker
from src.llm.simple_embedding_provider import SimpleEmbeddingProvider
from src.llm.openai_embedding_provider import OpenAIEmbeddingProvider
from src.knowledge.loader import DocumentLoader
from src.knowledge.json_store import JsonVectorStore
from src.knowledge.retriever import KnowledgeRetriever
from src.knowledge.schemas import RetrievedChunk


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = LLMClient(self._build_chat_provider())
        self.memory_store = JsonMemoryStore(config.memory_store_path)
        self.document_loader = DocumentLoader()
        self.text_chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.embeddings_provider = self._build_embedding_provider()
        self.vector_store = JsonVectorStore(config.knowledge_index_path)
        self.knowledge_retriever = KnowledgeRetriever(
            embedding_provider=self.embeddings_provider,
            vector_store=self.vector_store,
        )
        self.tool_registry = ToolRegistry()

        for tool in build_builtin_tools(self.memory_store):
            self.tool_registry.register(tool)

        self.planner = Planner(self.client)
        self._init_knowledge_base()
        self.session = ChatSession(
            client=self.client,
            tool_registry=self.tool_registry,
            tool_policy=config.to_tool_policy(),
            planner=self.planner,
            memory_store=self.memory_store,
            knowledge_retriever=self.knowledge_retriever,
            config=config.to_session_config(),
            system_prompt=config.system_prompt,
        )

    def _init_knowledge_base(self) -> None:
        self.vector_store.load()

        if not self.config.knowledge_base_path:
            return

        documents = self.document_loader.load_path(self.config.knowledge_base_path)
        chunks = self.text_chunker.chunk_documents(documents)
        self.knowledge_retriever.index(chunks)
        self.vector_store.save()

    def _build_chat_provider(self) -> ChatProvider:
        return OpenAIChatProvider(self.config.to_llm_config())

    def _build_embedding_provider(self) -> EmbeddingProvider:
        match self.config.embedding_provider:
            case "simple":
                return SimpleEmbeddingProvider()
            case "openai":
                if not self.config.embedding_api_key or not self.config.embedding_model:
                    raise ValueError(
                        "FOX_AGENT_EMBEDDING_API_KEY and FOX_AGENT_EMBEDDING_MODEL are all required when embedding_provider=openai"
                    )
                return OpenAIEmbeddingProvider(
                    api_key=self.config.embedding_api_key,
                    base_url=self.config.embedding_base_url,
                    model=self.config.embedding_model,
                    timeout=self.config.embedding_timeout,
                )
            case _:
                raise ValueError(
                    f"Unknown embedding_provider: {self.config.embedding_provider}"
                )

    def run(
        self,
        user_input: str,
        plan_mode: Literal["auto", "enable", "disable"] | None = None,
        memory_mode: Literal["auto", "disable"] | None = None,
        retrieval_mode: Literal["auto", "disable"] | None = None,
    ) -> LLMResponse:
        return self.session.chat(
            user_input=user_input,
            plan_mode=plan_mode,
            memory_mode=memory_mode,
            retrieval_mode=retrieval_mode,
        )

    def search_knowledge(
        self, query: str, k: int | None = None
    ) -> list[RetrievedChunk]:
        return self.knowledge_retriever.retrieve(
            query=query,
            k=k,
            min_score=self.config.retrieval_min_score,
        )

    def render_knowledge_search(self, query: str, k: int | None = None) -> str:
        return self.knowledge_retriever.render_debug(self.search_knowledge(query, k=k))

    def register_tool(self, tool: ToolDefinition) -> None:
        self.tool_registry.register(tool)

    def clear(self) -> None:
        self.session.clear()

    def get_history(self) -> list[Message]:
        return self.session.get_history()
