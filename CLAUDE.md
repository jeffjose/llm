# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a local LLM inference tool that runs language models directly on Linux using llama.cpp bindings. It provides a CLI interface for downloading models, running single or multi-model inference, and interactive chat sessions with various open-source LLMs.

## Architecture

The codebase consists of three main components:

1. **multi-model.ts**: Core inference engine that manages model loading, context creation, and parallel/sequential inference across multiple models
2. **interactive-multi-model.ts**: Interactive CLI providing menus for chat mode, model downloads, benchmarking, and various inference modes
3. **download-models.ts**: Standalone model downloader with curated list of GGUF models from HuggingFace

Key architectural decisions:

- Uses node-llama-cpp for direct llama.cpp integration
- Models are stored in `models/` directory as GGUF files
- Supports both streaming and non-streaming inference
- Context management with automatic cleanup to support multiple models

## Common Commands

```bash
# Install dependencies
pnpm install

# Start interactive mode (main entry point)
pnpm start

# Download models
pnpm download

# Build TypeScript
pnpm build

# Development mode with auto-reload
pnpm dev
```

## Development Notes

- The project uses TypeScript with CommonJS module system
- Model configurations are embedded in the source files with metadata (size, architecture, capabilities)
- Chat sessions maintain context within a single conversation but clear between models
- Downloads support wget, curl, or Node.js HTTPS as fallbacks
- The tool automatically detects available models in the `models/` directory

## Best Practices

- Always use pnpm
- No need to build the app, the user will test it manually
