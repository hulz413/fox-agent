from src.agent.cli import main

# Example usage:
#
# Interactive:
# python -m src.agent
#
# One-shot prompt:
# python -m src.agent -p "What is an AI agent?"
#
# Prompt + Planning:
# python -m src.agent -p "List files in src/tools and summarize them" --plan-mode auto
#
# Prompt + Planning + Memory
# python -m src.agent -p "What do you remember about this project?" --plan-mode disable --memory-mode auto
if __name__ == "__main__":
    main()
