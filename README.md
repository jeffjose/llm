# llm

Run LLMs locally on Linux.

## Setup

```bash
# Linux dependencies
sudo apt install build-essential

# Install
pnpm install
pnpm start
```

## Usage

### Chat with pre-configured models
```bash
pnpm start
# Select option 1 for chat mode
```

### Chat with custom GGUF file
```bash
# Method 1: Direct command line
pnpm start /path/to/your/model.gguf

# Method 2: Via menu
pnpm start
# Select option 3
# Enter path to GGUF file
```

### Download models
```bash
pnpm start
# Select option 2
```

### Other features
- Compare outputs from multiple models
- Benchmark model performance
- Run parallel or sequential inference

## Models

Pre-configured models include:
- Qwen (0.5B, 1.5B, 7B)
- Llama 3.2 (1B, 3B)
- Mistral 7B
- Gemma 2 (2B)
- DeepSeek R1 (1.5B)
- TinyLlama (1.1B)
- Phi-3 Mini (3.8B)
- SmolLM (135M, 360M)
- StableLM (1.6B)
- OpenHermes (0.77B)