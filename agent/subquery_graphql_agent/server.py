#!/usr/bin/env python3
"""
Multi-Project SubQuery GraphQL Agent Server

This server can handle multiple SubQuery projects by registering IPFS CIDs
that point to project manifests. Each project gets its own GraphQL agent
with customizable prompts and capabilities.

Features:
- Register projects via IPFS CID manifest fetching
- Project-specific chat completion endpoints: /<cid>/chat/completions
- Agent instance caching with TTL
- Configurable project domains and capabilities
- OpenAI-compatible streaming and non-streaming responses
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import yaml
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional for development

import httpx
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
try:
    # Try relative imports first (when run as part of a package)
    from . import create_graphql_toolkit
    from .tools import create_system_prompt  
    from .node_types import GraphqlProvider, detect_node_type
except ImportError:
    # Fall back to absolute imports (when run directly or imported from outside)
    try:
        from base import create_graphql_toolkit
        from tools import create_system_prompt
        from node_types import GraphqlProvider, detect_node_type
    except ImportError:
        # Last resort - try importing from current directory
        import base
        import tools
        import node_types
        create_graphql_toolkit = base.create_graphql_toolkit
        create_system_prompt = tools.create_system_prompt
        GraphqlProvider = node_types.GraphqlProvider
        detect_node_type = node_types.detect_node_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Set log level from environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if hasattr(logging, log_level):
    logger.setLevel(getattr(logging, log_level))
    logging.getLogger().setLevel(getattr(logging, log_level))

# Configuration
PROJECTS_DIR = Path("./projects")
CACHE_TTL = 3600  # 1 hour agent cache TTL
IPFS_API_URL = os.getenv("IPFS_API_URL", "https://unauthipfs.subquery.network/ipfs/api/v0")

async def fetch_from_ipfs(cid: str, path: str = "") -> str:
    """
    Fetch content from IPFS using multiple methods with fallbacks.
    
    Args:
        cid: IPFS CID
        path: Optional path within the IPFS directory
        
    Returns:
        str: Content of the file
    """
    ipfs_path = f"{cid}/{path}" if path else cid
    
    # Try SubQuery IPFS node first, then gateway fallbacks
    sources = [
        # SubQuery IPFS node (cat API with POST method) - PRIMARY
        {
            "name": "SubQuery IPFS Cat API",
            "url": f"{IPFS_API_URL}/cat",
            "method": "post",
            "params": {"arg": ipfs_path}
        },
        # Gateway fallbacks
        {
            "name": "Gateway (ipfs.io)",
            "url": f"https://ipfs.io/ipfs/{ipfs_path}",
            "method": "get"
        },
        {
            "name": "Gateway (gateway.pinata.cloud)",
            "url": f"https://gateway.pinata.cloud/ipfs/{ipfs_path}",
            "method": "get"
        },
        {
            "name": "Gateway (dweb.link)",
            "url": f"https://dweb.link/ipfs/{ipfs_path}",
            "method": "get"
        }
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for source in sources:
            try:
                logger.debug(f"Trying {source['name']}: {source['url']}")
                
                if source["method"] == "post":
                    response = await client.post(source["url"], params=source.get("params", {}))
                else:
                    response = await client.get(source["url"])
                
                if response.status_code == 200:
                    content = response.text
                    logger.info(f"Successfully fetched from {source['name']} ({len(content)} chars)")
                    return content
                else:
                    logger.warning(f"{source['name']} failed: {response.status_code} - {response.text[:100]}")
                    
            except Exception as e:
                logger.error(f"{source['name']} error: {e}")
                continue
    
    # If all sources fail
    raise HTTPException(
        status_code=400,
        detail=f"Failed to fetch {ipfs_path} from all IPFS sources"
    )


async def analyze_project_with_llm(manifest: dict, schema_content: str, llm=None) -> dict:
    """
    Use LLM to analyze project manifest and schema to generate appropriate prompts.
    
    Args:
        manifest: Project manifest data
        schema_content: GraphQL schema content
        
    Returns:
        dict: Generated domain_name, domain_capabilities, and decline_message
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain.schema import HumanMessage
        
        # Use provided LLM or create one with same config as GraphQLAgent
        if llm is None:
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(
                model=model_name,
                temperature=0  # Same as GraphQLAgent
            )
        
        # Prepare schema content for LLM (truncate if too long)
        schema_preview = schema_content[:3000] if len(schema_content) > 3000 else schema_content
        
        # Get project basics
        project_name = manifest.get('name', 'Unknown Project')
        project_description = manifest.get('description', '')
        
        # Get network/chain info
        network_info = ""
        if 'network' in manifest:
            network = manifest['network']
            if isinstance(network, dict):
                chain_id = network.get('chainId', network.get('endpoint', ''))
                network_info = f"Network: {chain_id}"
        
        # Get datasource info
        datasources_info = ""
        if 'dataSources' in manifest:
            ds_kinds = [ds.get('kind', 'unknown') for ds in manifest['dataSources']]
            datasources_info = f"Data sources: {', '.join(set(ds_kinds))}"
        
        # Create focused analysis prompt
        analysis_prompt = f"""Analyze this SubQuery indexing project and generate specific agent configuration:

PROJECT INFO:
- Name: {project_name}
- Description: {project_description}
- {network_info}
- {datasources_info}

GRAPHQL SCHEMA:
```graphql
{schema_preview}
```

Based on the project info and GraphQL schema entities, generate:

1. A clear domain_name that describes what this project indexes
2. Specific domain_capabilities based on the actual GraphQL entities and what queries users can make
3. A decline_message that mentions the specific domain
4. Suggested questions that users can ask to explore the data

IMPORTANT: Look at the GraphQL types to understand what this project tracks.

Respond ONLY with valid JSON in this exact format (no markdown code blocks):
{{
  "domain_name": "Specific Project Name",
  "domain_capabilities": [
    "Query [specific entity] data and relationships",
    "Analyze [specific metrics] and trends", 
    "Track [specific events/transactions]",
    "Monitor [specific blockchain activities]"
  ],
  "decline_message": "I'm specialized in {project_name} data queries. I can help you with [specific data types], but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
  "suggested_questions": [
    "Show me recent [specific entity type] transactions",
    "What are the top [entity] by [field]?",
    "How many [events] happened in the last day?",
    "Can you show me a sample GraphQL query for [entity]?"
  ]
}}

Make each capability very specific to the entities found in the schema."""

        logger.info("Analyzing project with LLM...")
        logger.info(f"Project info - Name: {project_name}, Description: {project_description[:100]}...")
        logger.debug(f"Network: {network_info}")
        logger.debug(f"Data sources: {datasources_info}")
        logger.debug(f"Schema length: {len(schema_content)} chars (preview: {len(schema_preview)} chars)")
        logger.debug(f"Sending prompt to LLM (length: {len(analysis_prompt)} chars)")
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        logger.debug(f"LLM Raw Response: {response.content}")
        
        # Parse JSON response - handle markdown code blocks
        try:
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            if content.startswith('```'):
                content = content[3:]   # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
            
            content = content.strip()
            
            result = json.loads(content)
            
            # Ensure all required fields are present
            if 'suggested_questions' not in result:
                logger.warning("LLM response missing suggested_questions, adding defaults")
                result['suggested_questions'] = [
                    "What types of data can I query from this project?",
                    "Show me a sample GraphQL query",
                    "What entities are available in this schema?",
                    "How can I filter the data?"
                ]
            
            logger.info(f"LLM analysis completed: {result['domain_name']}")
            logger.info(f"Generated capabilities: {len(result['domain_capabilities'])} items")
            logger.info(f"Generated questions: {len(result['suggested_questions'])} items")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"LLM response was not valid JSON: {e}")
            logger.debug(f"Full raw response: {response.content}")
            logger.debug(f"Cleaned content: {content}")
            raise ValueError("Invalid JSON response from LLM")
            
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}, using enhanced fallback")
        
        # Enhanced fallback analysis
        project_name = manifest.get('name', 'SubQuery Project')
        project_description = manifest.get('description', '')
        
        # Generate better domain name
        if project_description and len(project_description) > 10:
            domain_name = f"{project_name} - {project_description[:50]}..."
        else:
            domain_name = project_name
            
        # Generate basic capabilities
        capabilities = [
            "Query blockchain data indexed by this project",
            "Analyze transaction patterns and trends", 
            "Track historical blockchain activities",
            "Monitor smart contract events and state changes"
        ]
            
        return {
            "domain_name": domain_name,
            "domain_capabilities": capabilities,
            "decline_message": f"I'm specialized in {project_name} data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about {project_name} data instead.",
            "suggested_questions": [
                "What types of data can I query from this project?",
                "Show me a sample GraphQL query",
                "What entities are available in this schema?",
                "How can I filter the data?"
            ]
        }

@dataclass
class ProjectConfig:
    """Configuration for a SubQuery or The Graph project."""
    cid: str
    endpoint: str
    schema_content: str
    node_type: str = GraphqlProvider.UNKNOWN
    manifest: Dict[str, Any] = None
    domain_name: str = "GraphQL Project"
    domain_capabilities: List[str] = None
    decline_message: str = "I'm specialized in this project's data queries. I can help you with the indexed blockchain data, but I cannot assist with [their topic]. Please ask me about this project's data instead."
    suggested_questions: List[str] = None
    authorization: Optional[str] = None
    
    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.domain_capabilities is None:
            self.domain_capabilities = [
                "Blockchain data indexed by this project",
                "Entity relationships and queries",
                "Project-specific metrics and analytics"
            ]
        if self.suggested_questions is None:
            self.suggested_questions = [
                "What types of data can I query from this project?",
                "Show me a sample GraphQL query",
                "What entities are available in this schema?",
                "How can I filter the data?"
            ]

class ProjectManager:
    """Manages SubQuery projects and their configurations."""
    
    def __init__(self):
        self.projects: Dict[str, Dict[str, ProjectConfig]] = {}
        self.agent_cache: Dict[tuple, tuple] = {}  # (user_id, cid) -> (agent, timestamp)
        self._shared_llm = None  # Cached LLM instance for analysis
        PROJECTS_DIR.mkdir(exist_ok=True)
        self._load_projects()
    
    def _get_shared_llm(self):
        """Get or create a shared LLM instance with same config as GraphQLAgent."""
        if self._shared_llm is None:
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            self._shared_llm = ChatOpenAI(
                model=model_name,
                temperature=0
            )
            logger.info(f"Initialized shared LLM: {model_name}")
        return self._shared_llm
    
    def _get_project_file(self, user_id: str, cid: str) -> Path:
        """Get the file path for a user's project config."""
        user_dir = PROJECTS_DIR / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir / f"{cid}.json"
    
    def _load_projects(self):
        """Load all projects from disk into memory, grouped by user_id."""
        self.projects.clear()
        if not PROJECTS_DIR.exists():
            return
        for user_dir in PROJECTS_DIR.iterdir():
            if user_dir.is_dir():
                user_id = user_dir.name
                self.projects[user_id] = {}
                for file in user_dir.glob("*.json"):
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)
                            config = ProjectConfig(**data)
                            self.projects[user_id][config.cid] = config
                            logger.info(f"Loaded project: {config.cid} for user {user_id}")
                    except Exception as e:
                        logger.error(f"Failed to load project {file}: {e}")
    
    def _save_project(self, user_id: str, config: ProjectConfig):
        """Save a user's project config to disk."""
        file = self._get_project_file(user_id, config.cid)
        with open(file, "w") as f:
            json.dump(asdict(config), f, indent=2)
    
    async def register_project(self, user_id: str, cid: str, endpoint: str, authorization: Optional[str] = None) -> ProjectConfig:
        """Register a new project from IPFS CID for a specific user."""
        if user_id in self.projects and cid in self.projects[user_id]:
            return self.projects[user_id][cid]
        try:
            # Fetch manifest from IPFS using cat API
            logger.info(f"Fetching project manifest for CID: {cid}")
            manifest_content = await fetch_from_ipfs(cid)
            # Parse manifest (YAML or JSON)
            try:
                manifest = yaml.safe_load(manifest_content)
            except yaml.YAMLError:
                manifest = json.loads(manifest_content)
            # Handle different schema path formats
            schema_info = manifest.get('schema', {})
            if isinstance(schema_info, dict):
                # The Graph format: schema: { file: { "/": "/ipfs/QmXXX" } }
                if 'file' in schema_info and isinstance(schema_info['file'], dict) and '/' in schema_info['file']:
                    schema_path = schema_info['file']['/']
                    if schema_path.startswith('/ipfs/'):
                        # Extract CID from The Graph format: /ipfs/QmXXX
                        schema_cid = schema_path.replace('/ipfs/', '')
                        logger.debug(f"Fetching The Graph schema from IPFS CID: {schema_cid}")
                        schema_content = await fetch_from_ipfs(schema_cid)
                    else:
                        logger.debug(f"Fetching schema file: {schema_path}")
                        schema_content = await fetch_from_ipfs(cid, schema_path)
                else:
                    # SubQL format: schema: { file: "schema.graphql" }
                    schema_path = schema_info.get('file', 'schema.graphql')
                    if schema_path.startswith('http'):
                        logger.debug(f"Fetching schema from external URL: {schema_path}")
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            schema_response = await client.get(schema_path)
                            schema_response.raise_for_status()
                            schema_content = schema_response.text
                    elif schema_path.startswith('ipfs://'):
                        schema_cid = schema_path.replace('ipfs://', '')
                        logger.debug(f"Fetching SubQL schema from IPFS CID: {schema_cid}")
                        schema_content = await fetch_from_ipfs(schema_cid)
                    else:
                        logger.debug(f"Fetching schema file: {schema_path}")
                        schema_content = await fetch_from_ipfs(cid, schema_path)
            else:
                # Fallback for simple string format
                schema_path = str(schema_info) if schema_info else 'schema.graphql'
                logger.debug(f"Fetching schema file: {schema_path}")
                schema_content = await fetch_from_ipfs(cid, schema_path)
            
            # Detect node type
            detected_node_type = detect_node_type(manifest)
            logger.info(f"Detected node type: {detected_node_type}")

            shared_llm = self._get_shared_llm()
            llm_analysis = await analyze_project_with_llm(manifest, schema_content, shared_llm)
            config = ProjectConfig(
                cid=cid,
                endpoint=endpoint,
                schema_content=schema_content,
                node_type=detected_node_type,
                manifest=manifest,
                domain_name=llm_analysis["domain_name"],
                domain_capabilities=llm_analysis["domain_capabilities"],
                decline_message=llm_analysis["decline_message"],
                suggested_questions=llm_analysis.get("suggested_questions", []),
                authorization=authorization
            )
            # Save to disk and memory
            self._save_project(user_id, config)
            if user_id not in self.projects:
                self.projects[user_id] = {}
            self.projects[user_id][cid] = config
            logger.info(f"Registered project: {llm_analysis['domain_name']} ({cid}) for user {user_id}")
            return config
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to register project {cid}: {str(e)}"
            )
    
    def get_project(self, user_id: str, cid: str) -> Optional[ProjectConfig]:
        """Get a project configuration for a user."""
        return self.projects.get(user_id, {}).get(cid)
    
    def update_project_config(self, user_id: str, cid: str, **updates) -> ProjectConfig:
        """Update project configuration."""
        if user_id not in self.projects or cid not in self.projects[user_id]:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
        
        config = self.projects[user_id][cid]
        
        # Update fields
        for field, value in updates.items():
            if hasattr(config, field):
                setattr(config, field, value)
        
        # Save changes
        self._save_project(user_id, config)
        
        # Invalidate agent cache
        cache_key = (user_id, cid)
        if cache_key in self.agent_cache:
            del self.agent_cache[cache_key]
        
        logger.info(f"Updated project config: {cid}")
        return config
    
    def get_agent(self, user_id: str, cid: str) -> 'GraphQLAgent':
        """Get or create a cached agent for the project (per user)."""
        current_time = time.time()
        cache_key = (user_id, cid)
        # Check cache
        if cache_key in self.agent_cache:
            agent, timestamp = self.agent_cache[cache_key]
            if current_time - timestamp < CACHE_TTL:
                return agent
            else:
                # Cache expired
                del self.agent_cache[cache_key]
        # Create new agent
        config = self.get_project(user_id, cid)
        if not config:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
        agent = GraphQLAgent(config)
        self.agent_cache[cache_key] = (agent, current_time)
        logger.info(f"Created agent for project: {cid} (user: {user_id})")
        return agent
    
    def delete_project(self, user_id: str, cid: str) -> bool:
        """Delete a project and its configuration."""
        if user_id not in self.projects or cid not in self.projects[user_id]:
            raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
        
        # Remove from memory
        del self.projects[user_id][cid]
        
        # Remove cached agent
        cache_key = (user_id, cid)
        if cache_key in self.agent_cache:
            del self.agent_cache[cache_key]
        
        # Remove from disk - catch errors and return success
        project_file = self._get_project_file(user_id, cid)
        try:
            if project_file.exists():
                project_file.unlink()
                logger.info(f"Deleted project file: {project_file}")
        except Exception as e:
            logger.warning(f"Could not delete project file {project_file}: {e}")
            # Continue with deletion even if file deletion fails
        
        logger.info(f"Deleted project: {cid} for user {user_id}")
        return True
    
    def list_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """List all registered projects for a user by reading the user's project directory."""
        user_dir = PROJECTS_DIR / user_id
        projects = []
        if not user_dir.exists() or not user_dir.is_dir():
            return []
        for file in user_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    config = ProjectConfig(**data)
                    cache_key = (user_id, config.cid)
                    projects.append({
                        "cid": config.cid,
                        "domain_name": config.domain_name,
                        "endpoint": config.endpoint,
                        "cached": cache_key in self.agent_cache,
                        "provider_type": config.node_type,
                        "has_authorization": bool(config.authorization)
                    })
            except Exception as e:
                logger.error(f"Failed to load project {file}: {e}")
        return projects

class GraphQLAgent:
    """GraphQL agent for a specific SubQuery project."""
    
    def __init__(self, config: ProjectConfig):
        """Initialize the agent with project configuration."""
        self.config = config
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize LLM
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        )
        
        # Create tools with node type information and authorization header
        headers = {}
        if config.authorization:
            headers["Authorization"] = config.authorization
        
        toolkit = create_graphql_toolkit(
            config.endpoint, 
            config.schema_content,
            headers=headers if headers else None,
            node_type=config.node_type,
            manifest=config.manifest
        )
        self.tools = toolkit.get_tools()
        
        # Setup agent
        self._setup_agent()
    
    def _setup_agent(self):
        # Create system prompt for langgraph
        prompt = create_system_prompt(
            domain_name=self.config.domain_name,
            domain_capabilities=self.config.domain_capabilities,
            decline_message=self.config.decline_message
        )
        
        # Create agent with system message
        self.executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=prompt
        )

    async def query_no_stream(self, question):
        response = await self.executor.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={
                "recursion_limit": 25,
            }
        )
        return response

    async def query(self, messages: list, include_think: bool = False):
        """Streaming query using langgraph agent with conversation history support."""
        logger.info(f"GraphQLAgent.query called with include_think={include_think}")
        think_started = False
        chunk_size = 60
        
        # Convert message format if needed
        if isinstance(messages, str):
            # Backward compatibility: if a string is passed, treat as single user message
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list) and messages:
            # Convert ChatCompletionMessage objects to dict format if needed
            formatted_messages = []
            for msg in messages:
                if hasattr(msg, 'role') and hasattr(msg, 'content'):
                    # Pydantic model - convert to dict
                    formatted_messages.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    # Already in correct format
                    formatted_messages.append(msg)
                else:
                    # Fallback
                    formatted_messages.append({"role": "user", "content": str(msg)})
            messages = formatted_messages
        
        # Get last user message for logging
        last_user_msg = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        
        try:
            logger.info(f"Processing query for {self.config.cid} with {len(messages)} messages: {last_user_msg[:100]}...")
            async for event in self.executor.astream({"messages": messages}):
                logger.debug(f"Event keys: {list(event.keys())}")
                logger.debug(f"Event: {event}")
                
                # Handle langgraph events - they contain node names as keys
                for node_name, node_output in event.items():
                    if node_name == "agent":
                        # Agent node - contains tool calls or final message
                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            for message in messages:
                                # Handle tool calls in agent messages
                                if hasattr(message, 'tool_calls') and message.tool_calls and include_think:
                                    if not think_started:
                                        yield "<think>\n"
                                        think_started = True
                                    for tool_call in message.tool_calls:
                                        tool_name = tool_call.get('name', 'unknown')
                                        yield f"[Tool: {tool_name}]\n"
                                        
                                # Handle regular message content
                                elif hasattr(message, 'content') and message.content:
                                    if think_started:
                                        yield "</think>\n"
                                        think_started = False
                                    
                                    content = str(message.content).strip()
                                    idx = 0
                                    while idx < len(content):
                                        chunk = content[idx:idx+chunk_size]
                                        yield chunk
                                        idx += chunk_size
                                        
                    elif node_name == "tools" and include_think:
                        # Tools node - contains tool execution results
                        if not think_started:
                            yield "<think>\n"
                            think_started = True
                            
                        if isinstance(node_output, dict) and "messages" in node_output:
                            messages = node_output["messages"]
                            for message in messages:
                                if hasattr(message, 'content') and message.content:
                                    yield "\n[Tool Output]:\n"
                                    
                                    observation = str(message.content)
                                    # Truncate schema info output
                                    if 'graphql_schema_info' in str(message.name if hasattr(message, 'name') else ''):
                                        max_length = 2000
                                        if len(observation) > max_length:
                                            observation = observation[:max_length] + f"\n\n... [Output truncated after {max_length} characters to save tokens.]"
                                    
                                    idx = 0
                                    while idx < len(observation):
                                        chunk = observation[idx:idx+chunk_size]
                                        yield chunk
                                        idx += chunk_size
                                    yield "\n\n"
            
            # Close any remaining think block
            if think_started:
                yield "</think>\n"
                
        except Exception as e:
            logger.error(f"Query failed for {self.config.cid}: {str(e)}")
            yield f"I encountered an issue processing your query. Error: {str(e)}"
            return

# Global project manager
project_manager = ProjectManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Multi-Project SubQuery GraphQL Agent Server starting...")
    logger.info(f"Projects directory: {PROJECTS_DIR.absolute()}")
    logger.info(f"IPFS API: {IPFS_API_URL}")
    
    # Load existing projects
    project_count = sum(len(projects) for projects in project_manager.projects.values())
    if project_count > 0:
        logger.info(f"Loaded {project_count} existing projects")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Project GraphQL Agent Server...")

app = FastAPI(
    title="Multi-Project SubQuery GraphQL Agent API", 
    description="OpenAI-compatible API server for multiple SubQuery projects",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://graphql-agent-app.subquery.network"
    ],  # Frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class RegisterProjectRequest(BaseModel):
    cid: str = Field(..., description="IPFS CID of the SubQuery project manifest")
    endpoint: str = Field(..., description="GraphQL endpoint URL for the SubQuery project")
    authorization: Optional[str] = Field(None, description="Authorization header value for the GraphQL endpoint (e.g., 'Bearer token123')")

class RegisterProjectResponse(BaseModel):
    cid: str
    domain_name: str
    endpoint: str
    message: str

class UpdateProjectConfigRequest(BaseModel):
    domain_name: Optional[str] = None
    domain_capabilities: Optional[List[str]] = None
    decline_message: Optional[str] = None
    endpoint: Optional[str] = None
    suggested_questions: Optional[List[str]] = None
    authorization: Optional[str] = None

class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatCompletionMessage] = Field(..., description="List of messages")
    stream: bool = Field(default=False, description="Whether to stream responses")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    think: bool = Field(default=False, description="Whether to include think blocks with tool execution details")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

# API Endpoints

@app.post("/register", response_model=RegisterProjectResponse)
async def register_project(request: RegisterProjectRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    config = await project_manager.register_project(user_id, request.cid, request.endpoint, request.authorization)
    return RegisterProjectResponse(
        cid=config.cid,
        domain_name=config.domain_name,
        endpoint=config.endpoint,
        message=f"Project {config.domain_name} registered successfully"
    )

@app.get("/projects")
async def list_projects(user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    projects = project_manager.list_projects(user_id)
    return {
        "projects": projects,
        "total": len(projects)
    }

@app.get("/projects/{cid}")
async def get_project(cid: str = PathParam(..., description="Project CID"), user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    config = project_manager.get_project(user_id, cid)
    if not config:
        raise HTTPException(status_code=404, detail=f"Project {cid} not found for user {user_id}")
    # Mask authorization for security (show only if set)
    has_auth = bool(config.authorization)
    auth_preview = None
    if config.authorization:
        if config.authorization.lower().startswith('bearer '):
            token = config.authorization[7:]  # Remove 'Bearer '
            auth_preview = f"Bearer {token[:8]}..." if len(token) > 8 else "Bearer ***"
        else:
            auth_preview = f"{config.authorization[:8]}..." if len(config.authorization) > 8 else "***"
    
    cache_key = (user_id, cid)
    return {
        "cid": config.cid,
        "domain_name": config.domain_name,
        "domain_capabilities": config.domain_capabilities,
        "decline_message": config.decline_message,
        "endpoint": config.endpoint,
        "suggested_questions": config.suggested_questions,
        "has_authorization": has_auth,
        "authorization_preview": auth_preview,
        "cached": cache_key in project_manager.agent_cache,
        "provider_type": config.node_type
    }

@app.patch("/projects/{cid}")
async def update_project_config(cid: str, request: UpdateProjectConfigRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    updates = {k: v for k, v in request.dict().items() if v is not None}
    config = project_manager.update_project_config(user_id, cid, **updates)
    return {
        "cid": config.cid,
        "domain_name": config.domain_name,
        "domain_capabilities": config.domain_capabilities,
        "decline_message": config.decline_message,
        "endpoint": config.endpoint,
        "message": f"Project {cid} configuration updated"
    }

@app.delete("/projects/{cid}")
async def delete_project(cid: str, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    success = project_manager.delete_project(user_id, cid)
    return {
        "cid": cid,
        "deleted": success,
        "message": f"Project {cid} deleted successfully"
    }

@app.post("/{cid}/chat/completions")
async def project_chat_completions(cid: str, request: ChatCompletionRequest, user_id: str = Query(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required as query parameter")
    agent = project_manager.get_agent(user_id, cid)
    
    # Validate that we have messages
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Ensure there's at least one user message
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(agent, request.messages, request),
            media_type="text/plain"
        )
    else:
        return await non_stream_chat_completion(agent, request.messages, request)

async def non_stream_chat_completion(
    agent: GraphQLAgent, 
    messages: list, 
    request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Handle non-streaming chat completion."""
    try:
        # Process query through GraphQL agent
        response_parts = []
        async for chunk in agent.query(messages, include_think=request.think):
            response_parts.append(chunk)
        response_content = "".join(response_parts)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": sum(len(msg.content.split()) for msg in messages if hasattr(msg, 'content')),
                "completion_tokens": len(response_content.split()),
                "total_tokens": sum(len(msg.content.split()) for msg in messages if hasattr(msg, 'content')) + len(response_content.split())
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def stream_chat_completion(
    agent: GraphQLAgent,
    messages: list, 
    request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Handle streaming chat completion."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    try:
        # Stream agent response with conversation history
        async for part in agent.query(messages, include_think=request.think):
            chunk_size = 60
            idx = 0
            while idx < len(part):
                chunk_content = part[idx:idx+chunk_size]
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": chunk_content},
                        "finish_reason": None
                    }]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.05)
                idx += chunk_size
        # Final chunk
        final_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        error_chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {"content": f"Error: {str(e)}"},
                "finish_reason": "error"
            }]
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

# Legacy endpoints for backward compatibility
@app.post("/v1/chat/completions")
async def legacy_chat_completions(request: ChatCompletionRequest):
    """Legacy OpenAI compatible chat completions endpoint."""
    # Use the first available project or SubQuery Network as default
    projects = project_manager.list_projects()
    if not projects:
        raise HTTPException(
            status_code=503, 
            detail="No projects registered. Please register a project first using POST /register"
        )
    
    # Use first project as default
    default_cid = projects[0]["cid"]
    return await project_chat_completions(default_cid, request)

@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "anymodel",
                "object": "model",
                "created": 1234567890,
                "owned_by": "subql-graphql-agent"
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "projects_count": sum(len(projects) for projects in project_manager.projects.values()),
        "cached_agents": len(project_manager.agent_cache),
        "ipfs_api": IPFS_API_URL
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Multi-Project SubQuery GraphQL Agent API server on port {port}")
    logger.info("API endpoints:")
    logger.info(f"  - POST http://localhost:{port}/register")
    logger.info(f"  - GET  http://localhost:{port}/projects")
    logger.info(f"  - POST http://localhost:{port}/<cid>/chat/completions")
    logger.info(f"  - GET  http://localhost:{port}/health")
    logger.info(f"  - POST http://localhost:{port}/v1/chat/completions (legacy)")
    
    uvicorn.run(app, host="0.0.0.0", port=port)