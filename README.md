# LLM Agentic Framework

## Overview

The LLM Agentic Framework is a powerful tool that orchestrates multiple Large Language Models (LLMs) to accomplish complex, high-level objectives. By leveraging the unique strengths of different LLMs, the framework can efficiently break down tasks and distribute them to the most suitable model, creating a collaborative AI ecosystem that's greater than the sum of its parts.

At its core, the framework uses Claude Code as the orchestrator (or "conductor") that intelligently decomposes multimodal prompts into discrete tasks, then delegates these tasks to specialized LLM agents based on their individual capabilities, optimizing for performance, cost, and efficiency.

## Architecture

![LLM Agentic Framework Architecture](https://i.imgur.com/placeholder.png)

### Core Components

#### Orchestrator (Claude Code)

Claude Code serves as the central orchestrator of the framework. Its responsibilities include:

- Analyzing and understanding high-level objectives from multimodal prompts
- Breaking down complex objectives into smaller, manageable tasks
- Determining which LLM agent is best suited for each task based on capabilities
- Coordinating communication between agents
- Synthesizing individual outputs into a cohesive final result
- Handling error recovery and task reassignment when necessary

#### LLM Agents

The framework supports integration with various LLM agents, including but not limited to:

- **OpenAI Models (GPT-4, GPT-3.5)**: Excelling at general knowledge tasks and code generation
- **Google Gemini**: Strong at multimodal reasoning and image analysis
- **Perplexity**: Specialized in real-time information retrieval and web search integration for up-to-date knowledge
- **DeepSeek**: Optimized for code understanding, scientific reasoning, and specialized domain knowledge
- **Open Source Models** (Llama, Mistral, etc.): Available for specialized or private deployment scenarios
- **Specialized Models**: Task-specific models for unique requirements

#### Client Layer

A unified interface for interacting with different LLM APIs, handling:

- Authentication and API key management
- Rate limiting and retry logic
- Response parsing and error handling
- Context window management

#### Utilities

Supporting components including:

- Task router for matching tasks to appropriate agents
- Task scheduler for managing dependencies and parallel execution
- Memory module for maintaining context across interactions
- Logging and monitoring tools

## Installation

### Prerequisites

- Python 3.8+
- API keys for the LLMs you plan to use

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/windprince/llm-agentic-framework.git
   cd llm-agentic-framework
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Basic Example

```python
from llm_agentic_framework import AgenticFramework

# Initialize the framework
framework = AgenticFramework()

# Define a high-level objective
objective = """
Analyze the attached image of a financial statement, 
extract key metrics, and prepare a summary report with 
visualizations of the trends.
"""

# Add any attachments (images, documents, etc.)
framework.add_attachment("financial_statement.jpg")

# Execute the objective
result = framework.execute(objective)

# Access the results
print(result.summary)
result.save_report("financial_analysis.pdf")
```

### Advanced Configuration

```python
from llm_agentic_framework import AgenticFramework, AgentConfiguration

# Configure specific agents
config = {
    "claude": AgentConfiguration(
        model="claude-3-opus-20240229",
        temperature=0.7,
        max_tokens=4000
    ),
    "gemini": AgentConfiguration(
        model="gemini-pro-vision",
        temperature=0.2,
        priority_tasks=["image_analysis", "chart_generation"]
    )
}

# Initialize with custom configuration
framework = AgenticFramework(agent_config=config)

# Execute with execution preferences
result = framework.execute(
    objective="Design a marketing campaign based on this product data",
    execution_mode="parallel",
    max_budget_usd=5.0,
    timeout_seconds=300
)
```

## Extending the Framework

### Adding a New Agent

1. Create a new agent class in the `src/agents` directory:

```python
# src/agents/custom_agent.py
from llm_agentic_framework.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, api_key, model_name="default-model"):
        super().__init__(name="custom", capabilities=["specialized_task"])
        self.api_key = api_key
        self.model_name = model_name
        
    async def execute_task(self, task):
        # Implement task execution logic here
        result = await self._call_api(task.prompt, task.parameters)
        return self._parse_response(result)
        
    def _call_api(self, prompt, parameters):
        # Implement API call logic
        pass
        
    def _parse_response(self, raw_response):
        # Implement response parsing
        pass
```

2. Register the agent in the framework:

```python
# Usage
from llm_agentic_framework import AgenticFramework
from custom_agent import CustomAgent

framework = AgenticFramework()
framework.register_agent(
    CustomAgent(api_key="your-api-key", model_name="specialized-model")
)
```

### Creating Custom Task Types

```python
from llm_agentic_framework.tasks import TaskType, Task

# Define a new task type
class SpecializedTaskType(TaskType):
    name = "specialized_task"
    required_capabilities = ["specialized_capability"]
    priority = 5
    
# Create and execute tasks of this type
task = Task(
    type=SpecializedTaskType,
    prompt="Perform specialized analysis on this data",
    context={"data": [...]}
)

result = framework.execute_task(task)
```

## Contributing

We welcome contributions to the LLM Agentic Framework! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# llm-agentic-framework
# llm-agentic-framework
