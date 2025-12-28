#!/usr/bin/env python3
"""
Configuration Management Utility for Orchestrator Service.

This script provides utilities to manage agent configurations,
test different settings, and validate configuration files.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.config_manager import ConfigurationManager, get_config_manager
from core.agent_factory import ConfigurableAgentFactory


def list_configurations(config_path: Optional[str] = None):
    """List all available configurations."""
    try:
        config_manager = ConfigurationManager(config_path)
        
        print("=== SYSTEM CONFIGURATION ===")
        system_config = config_manager.get_system_config()
        print(f"Default Timeout: {system_config.default_timeout}s")
        print(f"Default Max Retries: {system_config.default_max_retries}")
        print(f"Verification Enabled: {system_config.enable_verification}")
        print(f"Planning Enabled: {system_config.enable_planning}")
        print()
        
        print("=== AVAILABLE MODELS ===")
        models = config_manager.list_available_models()
        for model_id in models:
            model_config = config_manager.get_model_config(model_id)
            print(f"{model_id}:")
            print(f"  Model: {model_config.name}")
            print(f"  Temperature: {model_config.temperature}")
            print(f"  Max Tokens: {model_config.max_tokens}")
            print(f"  Timeout: {model_config.timeout}")
            print()
        
        print("=== AVAILABLE AGENTS ===")
        agents = config_manager.list_available_agents()
        for agent_id in agents:
            agent_config = config_manager.get_agent_config(agent_id)
            model_config = config_manager.get_model_config(agent_config.model_config)
            print(f"{agent_id}:")
            print(f"  Type: {agent_config.agent_type}")
            print(f"  Model: {model_config.name}")
            print(f"  Temperature: {model_config.temperature}")
            print(f"  Max Tokens: {model_config.max_tokens}")
            print(f"  Parameters: {len(agent_config.parameters)} custom parameters")
            print()
            
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    return True


def validate_configuration(config_path: Optional[str] = None):
    """Validate configuration file."""
    try:
        config_manager = ConfigurationManager(config_path)
        factory = ConfigurableAgentFactory(config_manager)
        
        print("✓ Configuration file loaded successfully")
        
        # Validate system config
        system_config = config_manager.get_system_config()
        print("✓ System configuration is valid")
        
        # Validate model configs
        models = config_manager.list_available_models()
        print(f"✓ Found {len(models)} model configurations")
        
        for model_id in models:
            model_config = config_manager.get_model_config(model_id)
            if not model_config.name:
                print(f"✗ Model '{model_id}' has no name specified")
                return False
            print(f"  ✓ Model '{model_id}' is valid")
        
        # Validate agent configs
        agents = config_manager.list_available_agents()
        print(f"✓ Found {len(agents)} agent configurations")
        
        for agent_id in agents:
            try:
                agent_config = config_manager.get_agent_config(agent_id)
                if agent_config.model_config not in models:
                    print(f"✗ Agent '{agent_id}' references unknown model '{agent_config.model_config}'")
                    return False
                print(f"  ✓ Agent '{agent_id}' is valid")
            except Exception as e:
                print(f"✗ Agent '{agent_id}' configuration error: {e}")
                return False
        
        print("\n✅ All configurations are valid!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False


def test_agent_creation(config_path: Optional[str] = None):
    """Test agent creation with current configuration."""
    try:
        from adapters.openrouter_adapter import OpenRouterAdapter
        import os
        
        # Check if required environment variables are set
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠️  OPENROUTER_API_KEY not set - using mock adapter")
            # Create a mock adapter for testing
            class MockAgentPort:
                async def generate_response(self, request):
                    from core.domain import AgentResponse
                    return AgentResponse(content="Mock response", metadata={})
            agent_port = MockAgentPort()
        else:
            agent_port = OpenRouterAdapter(
                api_key=api_key,
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                default_model=os.getenv("OPENROUTER_DEFAULT_MODEL", "openai/gpt-3.5-turbo")
            )
        
        config_manager = ConfigurationManager(config_path)
        factory = ConfigurableAgentFactory(config_manager)
        
        print("Testing agent creation...")
        
        agents = config_manager.list_available_agents()
        for agent_id in agents:
            try:
                agent = factory.create_agent(agent_id, agent_port)
                print(f"✓ Successfully created {agent_id} agent")
                print(f"  Model: {agent.config.model}")
                print(f"  Temperature: {agent.config.temperature}")
            except Exception as e:
                print(f"✗ Failed to create {agent_id} agent: {e}")
                return False
        
        print("\n✅ All agents created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Agent creation test failed: {e}")
        return False


def update_model_settings(config_path: Optional[str] = None, model_id: str = None, **kwargs):
    """Update model settings interactively."""
    try:
        config_manager = ConfigurationManager(config_path)
        
        if not model_id:
            print("Available models:")
            models = config_manager.list_available_models()
            for i, mid in enumerate(models, 1):
                print(f"{i}. {mid}")
            
            choice = input("Select model to update (number): ").strip()
            try:
                model_id = models[int(choice) - 1]
            except (ValueError, IndexError):
                print("Invalid selection")
                return False
        
        print(f"\nUpdating model '{model_id}':")
        model_config = config_manager.get_model_config(model_id)
        
        print(f"Current settings:")
        print(f"  Name: {model_config.name}")
        print(f"  Temperature: {model_config.temperature}")
        print(f"  Max Tokens: {model_config.max_tokens}")
        print(f"  Timeout: {model_config.timeout}")
        
        # Collect overrides
        overrides = {}
        
        new_name = input(f"New model name (current: {model_config.name}, press Enter to keep): ").strip()
        if new_name:
            overrides["name"] = new_name
        
        new_temp = input(f"New temperature (current: {model_config.temperature}, press Enter to keep): ").strip()
        if new_temp:
            try:
                overrides["temperature"] = float(new_temp)
            except ValueError:
                print("Invalid temperature value")
                return False
        
        new_tokens = input(f"New max tokens (current: {model_config.max_tokens}, press Enter to keep): ").strip()
        if new_tokens:
            try:
                overrides["max_tokens"] = int(new_tokens) if new_tokens.lower() != "none" else None
            except ValueError:
                print("Invalid max tokens value")
                return False
        
        new_timeout = input(f"New timeout (current: {model_config.timeout}, press Enter to keep): ").strip()
        if new_timeout:
            try:
                overrides["timeout"] = int(new_timeout) if new_timeout.lower() != "none" else None
            except ValueError:
                print("Invalid timeout value")
                return False
        
        if overrides:
            config_manager.override_model_config(model_id, overrides)
            print(f"\n✅ Model '{model_id}' updated successfully!")
            
            # Show updated config
            updated_config = config_manager.get_model_config(model_id)
            print(f"Updated settings:")
            print(f"  Name: {updated_config.name}")
            print(f"  Temperature: {updated_config.temperature}")
            print(f"  Max Tokens: {updated_config.max_tokens}")
            print(f"  Timeout: {updated_config.timeout}")
        else:
            print("No changes made.")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to update model settings: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Orchestrator Configuration Management")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List configurations
    list_parser = subparsers.add_parser("list", help="List all configurations")
    
    # Validate configuration
    validate_parser = subparsers.add_parser("validate", help="Validate configuration file")
    
    # Test agent creation
    test_parser = subparsers.add_parser("test", help="Test agent creation")
    
    # Update model settings
    update_parser = subparsers.add_parser("update-model", help="Update model settings")
    update_parser.add_argument("model_id", nargs="?", help="Model ID to update")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    success = False
    
    if args.command == "list":
        success = list_configurations(args.config)
    elif args.command == "validate":
        success = validate_configuration(args.config)
    elif args.command == "test":
        success = test_agent_creation(args.config)
    elif args.command == "update-model":
        success = update_model_settings(args.config, getattr(args, 'model_id', None))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()