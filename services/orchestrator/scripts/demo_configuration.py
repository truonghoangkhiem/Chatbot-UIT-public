#!/usr/bin/env python3
"""
Demo script showing the new centralized configuration system.

This script demonstrates how easy it is to:
1. Load different configurations
2. Switch models without code changes  
3. Customize agent parameters
4. Create agents with the factory pattern
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from core.config_manager import ConfigurationManager
from core.agent_factory import ConfigurableAgentFactory
from core.container import get_container


async def demo_configuration_loading():
    """Demonstrate configuration loading and management."""
    print("🔧 CONFIGURATION LOADING DEMO")
    print("=" * 50)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "agents_config.yaml"
    config_manager = ConfigurationManager(str(config_path))
    
    # Show system configuration
    system_config = config_manager.get_system_config()
    print(f"✅ System Config Loaded:")
    print(f"   - Timeout: {system_config.default_timeout}s")
    print(f"   - Max Retries: {system_config.default_max_retries}")
    print(f"   - Verification: {system_config.enable_verification}")
    print(f"   - Planning: {system_config.enable_planning}")
    
    # Show available models
    models = config_manager.list_available_models()
    print(f"\n📦 Available Models ({len(models)}):")
    for model_id in models[:3]:  # Show first 3
        model_config = config_manager.get_model_config(model_id)
        print(f"   - {model_id}: {model_config.name} (T={model_config.temperature})")
    
    # Show available agents
    agents = config_manager.list_available_agents()
    print(f"\n🤖 Available Agents ({len(agents)}):")
    for agent_id in agents:
        agent_config = config_manager.get_agent_config(agent_id)
        print(f"   - {agent_id}: {agent_config.agent_type}")
    
    print("\n✅ Configuration loaded successfully!\n")


async def demo_model_switching():
    """Demonstrate how easy it is to switch models."""
    print("🔄 MODEL SWITCHING DEMO")
    print("=" * 50)
    
    config_path = Path(__file__).parent.parent / "config" / "agents_config.yaml"
    config_manager = ConfigurationManager(str(config_path))
    
    # Show original planner model
    planner_config = config_manager.get_agent_full_config("planner")
    print(f"📋 Original Planner Model: {planner_config['model']}")
    print(f"   Temperature: {planner_config['temperature']}")
    print(f"   Max Tokens: {planner_config['max_tokens']}")
    
    # Override model at runtime (no code change needed!)
    print(f"\n🔧 Switching to a different model...")
    config_manager.override_model_config("planner_model", {
        "name": "openai/gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 800
    })
    
    # Show updated configuration
    updated_config = config_manager.get_agent_full_config("planner")
    print(f"📋 Updated Planner Model: {updated_config['model']}")
    print(f"   Temperature: {updated_config['temperature']}")
    print(f"   Max Tokens: {updated_config['max_tokens']}")
    
    print("\n✅ Model switched without any code changes!\n")


async def demo_agent_factory():
    """Demonstrate agent creation with factory pattern."""
    print("🏭 AGENT FACTORY DEMO")
    print("=" * 50)
    
    # Create mock agent port for demo
    class MockAgentPort:
        async def generate_response(self, request):
            from core.domain import AgentResponse
            return AgentResponse(
                content=f"Mock response for model: {request.model}",
                metadata={"mock": True}
            )
    
    mock_port = MockAgentPort()
    
    config_path = Path(__file__).parent.parent / "config" / "agents_config.yaml"
    config_manager = ConfigurationManager(str(config_path))
    factory = ConfigurableAgentFactory(config_manager)
    
    # Create agents using factory
    print("🔨 Creating agents with factory...")
    
    agents = {}
    for agent_id in ["planner", "answer_agent"]:
        try:
            agent = factory.create_agent(agent_id, mock_port)
            agents[agent_id] = agent
            print(f"   ✅ Created {agent_id}")
            print(f"      Model: {agent.config.model}")
            print(f"      Temperature: {agent.config.temperature}")
        except Exception as e:
            print(f"   ❌ Failed to create {agent_id}: {e}")
    
    # Create agent with parameter overrides
    print(f"\n🎛️  Creating planner with custom parameters...")
    custom_planner = factory.create_agent_with_parameters(
        "planner",
        mock_port,
        temperature=0.8,  # More creative
        max_tokens=1500,  # More verbose
        model="google/gemma-3-27b-it:free"  # Different model
    )
    
    print(f"   ✅ Custom planner created:")
    print(f"      Model: {custom_planner.config.model}")
    print(f"      Temperature: {custom_planner.config.temperature}")
    print(f"      Max Tokens: {custom_planner.config.max_tokens}")
    
    print("\n✅ All agents created successfully!\n")


async def demo_container_integration():
    """Demonstrate container integration with configuration."""
    print("📦 CONTAINER INTEGRATION DEMO")
    print("=" * 50)
    
    config_path = Path(__file__).parent.parent / "config" / "agents_config.yaml"
    
    # Get container with configuration
    container = get_container(str(config_path))
    
    # Show services
    print("🔧 Available services:")
    print("   - ✅ Configuration Manager")
    print("   - ✅ Agent Factory")
    print("   - ✅ Agent Port (requires API key)")
    print("   - ✅ RAG Port")
    print("   - ✅ Conversation Manager")
    
    # Get configuration manager
    config_manager = container.get_config_manager()
    system_config = config_manager.get_system_config()
    print(f"\n⚙️  System settings from container:")
    print(f"   - Verification: {system_config.enable_verification}")
    print(f"   - Planning: {system_config.enable_planning}")
    
    # Get agent factory
    factory = container.get_agent_factory()
    available_types = factory.get_available_agent_types()
    print(f"\n🏭 Factory supports {len(available_types)} agent types:")
    for agent_type in available_types:
        print(f"   - {agent_type}")
    
    print("\n✅ Container integration working perfectly!\n")


async def demo_parameter_customization():
    """Demonstrate agent parameter customization."""
    print("🎛️  PARAMETER CUSTOMIZATION DEMO")
    print("=" * 50)
    
    config_path = Path(__file__).parent.parent / "config" / "agents_config.yaml"
    config_manager = ConfigurationManager(str(config_path))
    
    # Show planner parameters
    planner_config = config_manager.get_agent_config("planner")
    print("📋 Planner Agent Parameters:")
    
    parameters = planner_config.parameters
    if "complexity_thresholds" in parameters:
        thresholds = parameters["complexity_thresholds"]
        print(f"   🎯 Complexity Thresholds:")
        print(f"      - Simple max length: {thresholds.get('simple_max_length', 'Not set')}")
        print(f"      - Complex min length: {thresholds.get('complex_min_length', 'Not set')}")
    
    if "plan_templates" in parameters:
        templates = parameters["plan_templates"]
        print(f"   📝 Plan Templates:")
        for complexity, template in templates.items():
            print(f"      - {complexity}: {len(template)} steps")
    
    # Override parameters
    print(f"\n🔧 Customizing planner parameters...")
    config_manager.override_agent_config("planner", {
        "parameters": {
            "complexity_thresholds": {
                "simple_max_length": 30,      # Stricter simple threshold
                "complex_min_length": 200     # Higher complex threshold
            }
        }
    })
    
    # Show updated parameters
    updated_config = config_manager.get_agent_config("planner")
    updated_params = updated_config.parameters
    thresholds = updated_params["complexity_thresholds"]
    print(f"   ✅ Updated thresholds:")
    print(f"      - Simple max length: {thresholds['simple_max_length']}")
    print(f"      - Complex min length: {thresholds['complex_min_length']}")
    
    print("\n✅ Parameters customized successfully!\n")


async def main():
    """Run all demos."""
    print("🚀 CENTRALIZED CONFIGURATION SYSTEM DEMO")
    print("=" * 60)
    print("This demo shows the new configuration system features:")
    print("1. Centralized YAML configuration")
    print("2. Easy model switching")
    print("3. Agent factory pattern")
    print("4. Dependency injection container")
    print("5. Parameter customization")
    print("=" * 60)
    print()
    
    # Run all demos
    await demo_configuration_loading()
    await demo_model_switching()
    await demo_agent_factory()
    await demo_container_integration()
    await demo_parameter_customization()
    
    print("🎉 DEMO COMPLETED!")
    print("=" * 60)
    print("Key Benefits Demonstrated:")
    print("✅ No hardcoded configurations in agent code")
    print("✅ Easy model switching without code changes")
    print("✅ Runtime parameter customization")
    print("✅ Factory pattern for consistent agent creation")
    print("✅ Dependency injection for loose coupling")
    print()
    print("Next Steps:")
    print("1. Customize config/agents_config.yaml for your needs")
    print("2. Use scripts/manage_config.py for configuration management")
    print("3. Create environment-specific configurations")
    print("4. Enjoy easy maintenance and experimentation!")


if __name__ == "__main__":
    asyncio.run(main())