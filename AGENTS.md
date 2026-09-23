# Project Instructions

## Project

This is a Python project for a government information retrieval agent
focused on Kazakhstan government websites.

## Workflow
- Do not implement code changes before approval
- Briefly describe the intended changes and reasoning behind them and ask for approval
- If a task is complex, decompose it into smaller parts
- Implement large changes step by step with approval on each step

## Architecture

- Use LangGraph for agent orchestration.
- Keep retrieval logic separate from agent logic.
- Do not hallucinate information that isn't supported by retrieved sources.
- Do not use external LLM APIs, everything is going to be served locally with ollama

## Code

- Python 3.12+
- Use type hints.
- Prefer small, testable functions.
- Use async where appropriate for I/O.
- Keep configuration in config.yaml.
- Keep API keys in system environment
- Do not hardcode API keys or credentials.
- You can install additional Python packages, but ask for permission first
- If something can be easily done with a Python package, use a package
- Use Pydantic for type-checking

## Retrieval

- For retrieval, use QA-data provided in data/egov_ru.xlsx
- Retrieval should be reliable and grounded

## Testing

- Add tests for retrieval and agent behavior.
- Do not modify tests merely to make failing tests pass.