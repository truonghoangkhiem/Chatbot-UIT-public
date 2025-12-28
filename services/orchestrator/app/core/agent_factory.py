"""
Factory for creating specialized agents based on configuration.

This module implements the Factory pattern for creating agents,
allowing for easy configuration-driven agent instantiation.

Optimized 3-Agent Pipeline:
- SmartPlannerAgent: Planning + Query Rewriting
- AnswerAgent: Answer generation
- ResponseFormatterAgent: Verification + Formatting
"""

from typing import Dict, Any, Optional, Type, Union, List
from abc import ABC, abstractmethod

from ..agents.base import SpecializedAgent, AgentConfig, AgentType
from ..agents.answer_agent import AnswerAgent
from ..agents.smart_planner_agent import SmartPlannerAgent
from ..agents.response_formatter_agent import ResponseFormatterAgent
from ..ports.agent_ports import AgentPort
from ..core.config_manager import ConfigurationManager, get_config_manager


class AgentFactory(ABC):
    """Abstract base class for agent factories."""
    
    @abstractmethod
    def create_agent(
        self, 
        agent_id: str, 
        agent_port: AgentPort,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> SpecializedAgent:
        """Create an agent instance."""
        pass


class ConfigurableAgentFactory(AgentFactory):
    """
    Factory that creates agents based on configuration files.
    
    This factory reads agent configurations from YAML files and
    creates appropriate agent instances with injected dependencies.
    """
    
    # Registry of available agent classes (optimized 3-agent pipeline)
    AGENT_CLASSES: Dict[str, Type[SpecializedAgent]] = {
        "smart_planner": SmartPlannerAgent,
        "answer_agent": AnswerAgent,
        "response_formatter": ResponseFormatterAgent,
    }
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the factory.
        
        Args:
            config_manager: Configuration manager instance. If None, uses global instance.
        """
        self.config_manager = config_manager or get_config_manager()
    
    def create_agent(
        self, 
        agent_id: str, 
        agent_port: AgentPort,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> SpecializedAgent:
        """
        Create an agent instance based on configuration.
        
        Args:
            agent_id: Agent configuration ID (e.g., "planner", "answer_agent")
            agent_port: Port for agent communication
            config_overrides: Optional overrides for configuration values
            
        Returns:
            Configured agent instance
            
        Raises:
            ValueError: If agent configuration or class not found
        """
        try:
            # Get full configuration for the agent
            full_config = self.config_manager.get_agent_full_config(agent_id)
            
            # Apply overrides if provided
            if config_overrides:
                full_config.update(config_overrides)
            
            # Get agent type and find corresponding class
            agent_type = full_config.get("agent_type", agent_id)
            agent_class = self.AGENT_CLASSES.get(agent_type)
            
            if not agent_class:
                raise ValueError(f"No agent class found for type: {agent_type}")
            
            # Create AgentConfig instance
            agent_config = self._create_agent_config(full_config)
            
            # Create and return agent instance
            return agent_class(config=agent_config, agent_port=agent_port)
            
        except Exception as e:
            raise ValueError(f"Failed to create agent '{agent_id}': {e}")
    
    def _create_agent_config(self, config_data: Dict[str, Any]) -> AgentConfig:
        """
        Create AgentConfig from configuration data.
        
        Args:
            config_data: Configuration dictionary
            
        Returns:
            AgentConfig instance
        """
        # Map agent type string to enum (optimized 3-agent pipeline)
        agent_type_map = {
            "smart_planner": AgentType.SMART_PLANNER,
            "answer_agent": AgentType.ANSWER_AGENT,
            "response_formatter": AgentType.RESPONSE_FORMATTER,
        }
        
        agent_type_str = config_data.get("agent_type", "")
        agent_type = agent_type_map.get(agent_type_str)
        
        if not agent_type:
            raise ValueError(f"Unknown agent type: {agent_type_str}")
        
        return AgentConfig(
            agent_type=agent_type,
            model=config_data.get("model", ""),
            system_prompt=config_data.get("system_prompt", ""),
            temperature=config_data.get("temperature", 0.7),
            max_tokens=config_data.get("max_tokens"),
            timeout=config_data.get("timeout", 30),
            max_retries=config_data.get("max_retries", 3),
            parameters=config_data.get("parameters", {})
        )
    
    def create_all_agents(
        self, 
        agent_port: AgentPort,
        global_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SpecializedAgent]:
        """
        Create all available agents.
        
        Args:
            agent_port: Port for agent communication
            global_overrides: Global overrides to apply to all agents
            
        Returns:
            Dictionary mapping agent IDs to agent instances
        """
        agents = {}
        available_agents = self.config_manager.list_available_agents()
        
        for agent_id in available_agents:
            try:
                agent = self.create_agent(
                    agent_id=agent_id,
                    agent_port=agent_port,
                    config_overrides=global_overrides
                )
                agents[agent_id] = agent
            except Exception as e:
                print(f"Warning: Failed to create agent '{agent_id}': {e}")
                continue
        
        return agents
    
    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types."""
        return list(self.AGENT_CLASSES.keys())
    
    def register_agent_class(self, agent_type: str, agent_class: Type[SpecializedAgent]) -> None:
        """
        Register a new agent class.
        
        Args:
            agent_type: Agent type identifier
            agent_class: Agent class to register
        """
        self.AGENT_CLASSES[agent_type] = agent_class
    
    def create_agent_with_parameters(
        self,
        agent_id: str,
        agent_port: AgentPort,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> SpecializedAgent:
        """
        Create agent with specific parameters.
        
        Args:
            agent_id: Agent configuration ID
            agent_port: Port for agent communication
            model: Override model name
            temperature: Override temperature
            max_tokens: Override max tokens
            system_prompt: Override system prompt
            **kwargs: Additional parameter overrides
            
        Returns:
            Configured agent instance
        """
        overrides = {}
        
        if model is not None:
            overrides["model"] = model
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        if system_prompt is not None:
            overrides["system_prompt"] = system_prompt
            
        overrides.update(kwargs)
        
        return self.create_agent(
            agent_id=agent_id,
            agent_port=agent_port,
            config_overrides=overrides
        )


class StaticAgentFactory(AgentFactory):
    """
    Factory that creates agents using configuration from YAML.
    
    This factory is a lightweight wrapper around ConfigurableAgentFactory.
    It's kept for backward compatibility but delegates to ConfigurableAgentFactory.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """Initialize the static factory."""
        self.config_manager = config_manager or get_config_manager()
        self.configurable_factory = ConfigurableAgentFactory(self.config_manager)
    
    def create_agent(
        self, 
        agent_id: str, 
        agent_port: AgentPort,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> SpecializedAgent:
        """
        Create agent using configuration from YAML.
        
        Args:
            agent_id: Agent type identifier
            agent_port: Port for agent communication
            config_overrides: Optional configuration overrides
            
        Returns:
            Agent instance configured from YAML
        """
        # Delegate to ConfigurableAgentFactory
        return self.configurable_factory.create_agent(
            agent_id=agent_id,
            agent_port=agent_port,
            config_overrides=config_overrides
        )


# Global factory instance
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory(use_config: bool = True) -> AgentFactory:
    """
    Get the global agent factory instance.
    
    Args:
        use_config: Whether to use configuration-based factory (True) or static factory (False)
        
    Returns:
        AgentFactory instance
    """
    global _agent_factory
    if _agent_factory is None:
        if use_config:
            _agent_factory = ConfigurableAgentFactory()
        else:
            _agent_factory = StaticAgentFactory()
    return _agent_factory


def set_agent_factory(factory: AgentFactory) -> None:
    """
    Set the global agent factory instance.
    
    Args:
        factory: AgentFactory instance to use globally
    """
    global _agent_factory
    _agent_factory = factory