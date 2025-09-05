"""Base GraphQL Toolkit implementation."""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.agents.agent_toolkits.base import BaseToolkit
from pydantic import ConfigDict

from .tools import (
    GraphQLSchemaInfoTool,
    GraphQLTypeDetailTool,
    GraphQLQueryValidatorTool,
    GraphQLExecuteTool
)


class GraphQLSource:
    """
    GraphQL database connection wrapper.
    Similar to langchain's SQLDatabase but for GraphQL endpoints.
    """
    
    def __init__(
        self,
        endpoint: str,
        entity_schema: str,
        headers: Optional[Dict[str, str]] = None,
        schema_cache_ttl: int = 3600,
        node_type: Optional[str] = None,
        manifest: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GraphQL database connection.
        
        Args:
            endpoint: GraphQL endpoint URL
            entity_schema: Raw schema content for entity definitions
            headers: Optional HTTP headers
            schema_cache_ttl: Schema cache time-to-live in seconds
            node_type: Optional explicit node type (SubQL, The Graph, etc.)
            manifest: Optional project manifest for node type detection
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.schema_cache_ttl = schema_cache_ttl
        self.entity_schema = entity_schema
        self.node_type = node_type
        self.manifest = manifest or {}
        self._schema_cache: Optional[Dict] = None
        self._schema_timestamp = 0
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get GraphQL schema with caching."""
        import time
        from .graphql import fetch_graphql_schema
        
        current_time = time.time()
        if (self._schema_cache is None or 
            current_time - self._schema_timestamp > self.schema_cache_ttl):
            
            introspection_result = await fetch_graphql_schema(
                self.endpoint, 
                include_arg_descriptions=True,
                headers=self.headers
            )
            self._schema_cache = introspection_result
            self._schema_timestamp = current_time
        
        return self._schema_cache
    
    async def get_schema_data(self) -> Dict[str, Any]:
        """Get just the __schema part for compatibility with existing code."""
        introspection_result = await self.get_schema()
        return introspection_result.get("data", {}).get("__schema", {})
    
    async def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query."""
        import aiohttp
        
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers={**self.headers, "Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"GraphQL query failed: {response.status}")
    
    def get_endpoint(self) -> str:
        """Get the GraphQL endpoint URL."""
        return self.endpoint


class GraphQLToolkit(BaseToolkit):
    """
    GraphQL Agent Toolkit.
    
    Provides tools for LLM agents to interact with GraphQL APIs,
    similar to LangChain's SQLDatabaseToolkit.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    graphql_source: GraphQLSource
    llm: Optional[BaseLanguageModel] = None
    
    def __init__(
        self,
        graphql_source: GraphQLSource,
        llm: Optional[BaseLanguageModel] = None,
        **kwargs
    ):
        """
        Initialize the GraphQL toolkit.
        
        Args:
            graphql_source: GraphQL source connection with schema access
            llm: Optional language model for advanced tools
        """
        super().__init__(graphql_source=graphql_source, llm=llm, **kwargs)
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all available GraphQL tools.
        
        Returns:
            List of GraphQL tools
        """
        tools = [
            GraphQLSchemaInfoTool(
                graphql_source=self.graphql_source,
                node_type=self.graphql_source.node_type
            ),
            GraphQLTypeDetailTool(graphql_source=self.graphql_source),
            GraphQLQueryValidatorTool(graphql_source=self.graphql_source),
            GraphQLExecuteTool(graphql_source=self.graphql_source)
        ]
        
        return tools
    
    @property
    def dialect(self) -> str:
        """Get the dialect name."""
        return "graphql"



# Factory function for creating GraphQL toolkit
def create_graphql_toolkit(
    endpoint: str,
    entity_schema: str,
    headers: Optional[Dict[str, str]] = None,
    llm: Optional[BaseLanguageModel] = None,
    node_type: Optional[str] = None,
    manifest: Optional[Dict[str, Any]] = None
) -> GraphQLToolkit:
    """
    Create a GraphQL toolkit instance with automatic node type detection.
    
    Args:
        endpoint: GraphQL endpoint URL
        entity_schema: Raw schema content for entity definitions
        headers: Optional HTTP headers for authentication
        llm: Optional language model
        node_type: Optional explicit node type (if known)
        manifest: Optional project manifest for node type detection
        
    Returns:
        GraphQL toolkit instance
    """
    graphql_source = GraphQLSource(
        endpoint=endpoint, 
        entity_schema=entity_schema, 
        headers=headers,
        node_type=node_type,
        manifest=manifest
    )
    return GraphQLToolkit(graphql_source=graphql_source, llm=llm)