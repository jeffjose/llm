#!/bin/bash

# Test script for custom GGUF functionality

echo "Testing custom GGUF feature..."
echo ""
echo "Usage examples:"
echo ""
echo "1. Direct command line:"
echo "   pnpm start /path/to/your/model.gguf"
echo ""
echo "2. Via menu option 3:"
echo "   pnpm start"
echo "   Select option 3"
echo "   Enter path to GGUF file"
echo ""
echo "Example with a model from the models directory:"
if [ -f "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" ]; then
    echo "   pnpm start models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
else
    echo "   (No models found in models/ directory)"
fi