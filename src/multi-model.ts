import { getLlama, LlamaChatSession, Llama, LlamaModel, LlamaContext } from 'node-llama-cpp';
import path from 'path';
import { existsSync } from 'fs';

interface ModelConfig {
  name: string;
  displayName: string;
  enabled: boolean;
  filename: string;
  size: string;
  architecture: string;
  releaseDate: string;
  capabilities: string;
  bestFor: string;
}

interface InferenceResult {
  model: string;
  response: string;
  duration: number;
  tokensPerSecond?: number;
}

class MultiModelInference {
  private llama?: Llama;
  private loadedModels: Map<string, { model: LlamaModel, context: LlamaContext }> = new Map();
  models: ModelConfig[] = [
    // Tiny models (< 500MB)
    { 
      name: 'qwen2.5:0.5b', 
      displayName: 'Qwen2.5 0.5B', 
      enabled: false,
      filename: 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
      size: '0.4GB',
      architecture: 'Qwen2.5 (Transformer)',
      releaseDate: '2024',
      capabilities: 'Multilingual, basic reasoning',
      bestFor: 'Quick responses, testing, simple queries'
    },
    { 
      name: 'smollm:135m', 
      displayName: 'SmolLM 135M', 
      enabled: false,
      filename: 'SmolLM-135M-Instruct-Q8_0.gguf',
      size: '0.15GB',
      architecture: 'SmolLM (HuggingFace)',
      releaseDate: '2024',
      capabilities: 'Ultra-lightweight, basic chat',
      bestFor: 'Edge devices, simple tasks'
    },
    { 
      name: 'smollm:360m', 
      displayName: 'SmolLM 360M', 
      enabled: false,
      filename: 'SmolLM-360M-Instruct-Q4_K_M.gguf',
      size: '0.25GB',
      architecture: 'SmolLM (HuggingFace)',
      releaseDate: '2024',
      capabilities: 'Lightweight, improved reasoning',
      bestFor: 'Mobile apps, quick responses'
    },
    
    // Small models (500MB - 1GB)
    { 
      name: 'tinyllama', 
      displayName: 'TinyLlama 1.1B', 
      enabled: false,
      filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
      size: '0.7GB',
      architecture: 'Llama architecture',
      releaseDate: '2024',
      capabilities: 'Chat-optimized, efficient',
      bestFor: 'Conversational AI, low-resource environments'
    },
    { 
      name: 'stablelm:1.6b', 
      displayName: 'StableLM Zephyr 1.6B', 
      enabled: false,
      filename: 'stablelm-zephyr-1_6b-Q4_K_M.gguf',
      size: '0.95GB',
      architecture: 'StableLM (Stability AI)',
      releaseDate: '2024',
      capabilities: 'Instruction following, coding',
      bestFor: 'Code completion, technical tasks'
    },
    { 
      name: 'openhermes:0.77b', 
      displayName: 'OpenHermes 0.77B', 
      enabled: false,
      filename: 'openhermes-0.77b-v3.5-Q4_K_M.gguf',
      size: '0.5GB',
      architecture: 'Hermes (Nous Research)',
      releaseDate: '2024',
      capabilities: 'Function calling, structured output',
      bestFor: 'API integration, tool use'
    },
    { 
      name: 'phi3:mini', 
      displayName: 'Phi-3 Mini 3.8B', 
      enabled: false,
      filename: 'Phi-3-mini-4k-instruct-q4.gguf',
      size: '2.3GB',
      architecture: 'Phi-3 (Microsoft)',
      releaseDate: '2024',
      capabilities: 'Strong reasoning, coding',
      bestFor: 'Code generation, logical reasoning, STEM'
    },
    
    // Medium models (1GB - 3GB)
    { 
      name: 'llama3.2:1b', 
      displayName: 'Llama 3.2 1B', 
      enabled: false,
      filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
      size: '0.8GB',
      architecture: 'Llama 3.2 (Meta)',
      releaseDate: '2024',
      capabilities: 'Latest Meta model, instruction-tuned',
      bestFor: 'General tasks, following instructions'
    },
    { 
      name: 'llama3.2:3b', 
      displayName: 'Llama 3.2 3B', 
      enabled: false,
      filename: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
      size: '2.0GB',
      architecture: 'Llama 3.2 (Meta)',
      releaseDate: '2024',
      capabilities: 'Enhanced reasoning, longer context',
      bestFor: 'Complex conversations, detailed responses'
    },
    { 
      name: 'gemma2:2b', 
      displayName: 'Gemma 2 2B', 
      enabled: false,
      filename: 'gemma-2-2b-it-Q4_K_M.gguf',
      size: '1.6GB',
      architecture: 'Gemma 2 (Google)',
      releaseDate: '2024',
      capabilities: 'Efficient, safety-tuned',
      bestFor: 'Safe content generation, general chat'
    },
    { 
      name: 'qwen2.5:1.5b', 
      displayName: 'Qwen2.5 1.5B', 
      enabled: false,
      filename: 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf',
      size: '1.0GB',
      architecture: 'Qwen2.5 (Alibaba)',
      releaseDate: '2024',
      capabilities: 'Strong multilingual, math',
      bestFor: 'Multilingual tasks, mathematical reasoning'
    },
    { 
      name: 'qwen3:4b', 
      displayName: 'Qwen3 4B', 
      enabled: false,
      filename: 'Qwen3-4B-Q4_K_M.gguf',
      size: '2.4GB',
      architecture: 'Qwen3 (Alibaba)',
      releaseDate: '2024',
      capabilities: 'Advanced reasoning, thinking mode',
      bestFor: 'Complex reasoning, step-by-step thinking'
    },
    
    // Large models (3GB+)
    { 
      name: 'mistral:7b', 
      displayName: 'Mistral 7B', 
      enabled: false,
      filename: 'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
      size: '4.4GB',
      architecture: 'Mistral (v0.3)',
      releaseDate: '2024',
      capabilities: 'State-of-the-art 7B performance',
      bestFor: 'Professional use, creative writing, analysis'
    },
    { 
      name: 'qwen2.5:7b', 
      displayName: 'Qwen2.5 7B', 
      enabled: false,
      filename: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
      size: '4.7GB',
      architecture: 'Qwen2.5 (Alibaba)',
      releaseDate: '2024',
      capabilities: 'Top multilingual, coding, math',
      bestFor: 'Professional multilingual tasks, coding'
    },
    { 
      name: 'deepseek-r1:1.5b', 
      displayName: 'DeepSeek R1 1.5B', 
      enabled: false,
      filename: 'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
      size: '1.0GB',
      architecture: 'DeepSeek R1 (Distilled)',
      releaseDate: '2025',
      capabilities: 'Reasoning-focused, distilled from larger model',
      bestFor: 'Step-by-step reasoning, problem solving'
    }
  ];

  constructor() {}

  async initialize() {
    this.llama = await getLlama();
  }

  // Enable/disable specific models
  setModelEnabled(modelName: string, enabled: boolean) {
    const model = this.models.find(m => m.name === modelName);
    if (model) {
      model.enabled = enabled;
      // Clear the model from cache if disabling
      if (!enabled && this.loadedModels.has(modelName)) {
        const { context, model: llamaModel } = this.loadedModels.get(modelName)!;
        context.dispose();
        llamaModel.dispose();
        this.loadedModels.delete(modelName);
      }
    }
  }

  // Get model configuration by name
  public getModelConfig(modelName: string): ModelConfig | undefined {
    return this.models.find(m => m.name === modelName);
  }

  // Check which models are available locally
  async getAvailableModels(): Promise<string[]> {
    const available: string[] = [];
    for (const model of this.models) {
      const modelPath = path.join(process.cwd(), 'models', model.filename);
      if (existsSync(modelPath)) {
        available.push(model.name);
      }
    }
    return available;
  }

  // Load a model if not already loaded
  public async loadModel(modelConfig: ModelConfig) {
    if (!this.llama) throw new Error('Llama not initialized');
    
    if (this.loadedModels.has(modelConfig.name)) {
      return this.loadedModels.get(modelConfig.name)!;
    }

    const modelPath = path.join(process.cwd(), 'models', modelConfig.filename);
    if (!existsSync(modelPath)) {
      throw new Error(`Model file not found: ${modelPath}`);
    }

    console.log(`Loading ${modelConfig.displayName}...`);
    const model = await this.llama.loadModel({
      modelPath,
      gpuLayers: 0
    });

    const context = await model.createContext({
      contextSize: 2048, // Increased context size
      sequences: 1 // Explicitly set sequences
    });

    this.loadedModels.set(modelConfig.name, { model, context });
    return { model, context };
  }

  // Load a custom model from a specific path
  public async loadCustomModel(modelPath: string) {
    if (!this.llama) throw new Error('Llama not initialized');
    
    if (!existsSync(modelPath)) {
      throw new Error(`Model file not found: ${modelPath}`);
    }

    const modelName = `custom_${path.basename(modelPath)}`;
    
    // Check if already loaded
    if (this.loadedModels.has(modelName)) {
      return this.loadedModels.get(modelName)!;
    }

    console.log(`Loading custom model: ${path.basename(modelPath)}...`);
    const model = await this.llama.loadModel({
      modelPath,
      gpuLayers: 0
    });

    const context = await model.createContext({
      contextSize: 1024 // Smaller context for multi-model usage
    });

    this.loadedModels.set(modelName, { model, context });
    return { model, context };
  }

  // Run inference on a single model
  async inferSingle(
    modelName: string, 
    prompt: string, 
    stream: boolean = false
  ): Promise<InferenceResult> {
    const modelConfig = this.models.find(m => m.name === modelName);
    if (!modelConfig) {
      throw new Error(`Unknown model: ${modelName}`);
    }

    const startTime = Date.now();
    
    try {
      const { context } = await this.loadModel(modelConfig);
      
      // Get a sequence from the context
      const sequence = context.getSequence();
      const session = new LlamaChatSession({
        contextSequence: sequence
      });

    if (stream) {
      let response = '';
      await session.prompt(prompt, {
        onTextChunk: (chunk) => {
          response += chunk;
          process.stdout.write(chunk);
        }
      });

      sequence.clearHistory();
      return {
        model: modelName,
        response,
        duration: Date.now() - startTime
      };
    }

    const response = await session.prompt(prompt);
    sequence.clearHistory();
    
    return {
      model: modelName,
      response,
      duration: Date.now() - startTime
    };
    } catch (error: any) {
      console.error(`Error in inferSingle for ${modelName}:`, error.message);
      throw error;
    }
  }

  // Run inference on multiple models in parallel
  async inferParallel(prompt: string): Promise<InferenceResult[]> {
    const enabledModels = this.models.filter(m => m.enabled);
    
    console.log(`🚀 Running inference on ${enabledModels.length} models in parallel...\n`);

    // Pre-load all models to avoid race conditions
    const loadPromises = enabledModels.map(async model => {
      try {
        await this.loadModel(model);
        return true;
      } catch (err) {
        console.error(`Failed to load ${model.name}: ${err}`);
        return false;
      }
    });
    
    await Promise.all(loadPromises);

    // Run inference with better error handling
    const promises = enabledModels.map(async model => {
      try {
        // Create a separate context for each parallel inference
        const modelPath = path.join(process.cwd(), 'models', model.filename);
        if (!existsSync(modelPath)) {
          throw new Error(`Model file not found: ${modelPath}`);
        }

        // Use cached model if available
        if (this.loadedModels.has(model.name)) {
          return await this.inferSingle(model.name, prompt);
        }

        // Otherwise load a fresh instance
        const llamaModel = await this.llama!.loadModel({
          modelPath,
          gpuLayers: 0
        });

        const context = await llamaModel.createContext({
          contextSize: 2048,
          sequences: 1
        });

        const sequence = context.getSequence();
        const session = new LlamaChatSession({
          contextSequence: sequence
        });

        const startTime = Date.now();
        const response = await session.prompt(prompt);
        
        // Clean up immediately
        sequence.clearHistory();
        await context.dispose();
        await llamaModel.dispose();

        return {
          model: model.name,
          response,
          duration: Date.now() - startTime
        };
      } catch (err: any) {
        return {
          model: model.name,
          response: `Error: ${err.message}`,
          duration: 0
        };
      }
    });

    return Promise.all(promises);
  }

  // Run inference on models sequentially with streaming
  async inferSequential(prompt: string): Promise<InferenceResult[]> {
    const enabledModels = this.models.filter(m => m.enabled);
    const results: InferenceResult[] = [];

    for (const model of enabledModels) {
      console.log(`\n📝 ${model.displayName}:`);
      console.log('─'.repeat(50));
      
      try {
        const result = await this.inferSingle(model.name, prompt, true);
        results.push(result);
        console.log(`\n⏱️  Time: ${result.duration}ms`);
      } catch (err: any) {
        console.log(`❌ Error: ${err.message}`);
        results.push({
          model: model.name,
          response: `Error: ${err.message}`,
          duration: 0
        });
      }
    }

    return results;
  }

  // Compare outputs side by side
  displayComparison(results: InferenceResult[], compact: boolean = false) {
    console.log('\n📊 Model Comparison:');
    console.log('═'.repeat(80));
    
    if (compact) {
      // Compact format for short responses with inline timings
      const maxModelLength = Math.max(...results.map(r => {
        const model = this.models.find(m => m.name === r.model);
        return (model?.displayName || r.model).length;
      }));
      
      console.log();
      
      // Sort by duration (fastest first)
      const sortedResults = [...results].sort((a, b) => {
        // Put errors at the bottom
        if (a.duration === 0) return 1;
        if (b.duration === 0) return -1;
        return a.duration - b.duration;
      });
      
      sortedResults.forEach(result => {
        const model = this.models.find(m => m.name === result.model);
        const modelName = model?.displayName || result.model;
        const paddedName = modelName.padEnd(maxModelLength);
        
        // Format timing
        const timing = result.duration > 0 
          ? `[${result.duration.toString().padStart(5)}ms]`
          : '[  ERROR]';
        
        // Format response - handle multi-line responses
        const responseLines = result.response.trim().split('\n');
        const firstLine = responseLines[0];
        
        console.log(`${timing} [${paddedName}]: ${firstLine}`);
        
        // Print additional lines with proper indentation
        if (responseLines.length > 1) {
          const indent = ' '.repeat(timing.length + 3 + paddedName.length + 3);
          responseLines.slice(1).forEach(line => {
            if (line.trim()) {  // Skip empty lines
              console.log(`${indent}${line}`);
            }
          });
        }
        console.log();  // Empty line between responses
      });
    } else {
      // Verbose format for longer responses
      for (const result of results) {
        const model = this.models.find(m => m.name === result.model);
        console.log(`\n🤖 ${model?.displayName || result.model}`);
        console.log(`⏱️  Duration: ${result.duration}ms`);
        if (result.tokensPerSecond) {
          console.log(`⚡ Speed: ${result.tokensPerSecond} tokens/sec`);
        }
        console.log('─'.repeat(40));
        console.log(result.response.trim());
      }
    }
    
    console.log('\n' + '═'.repeat(80));
  }

  // List all configured models
  listModels() {
    console.log('\n📋 Configured Models:');
    this.models.forEach((model, index) => {
      const status = model.enabled ? '✅' : '❌';
      console.log(`${index + 1}. ${status} ${model.displayName} (${model.name})`);
    });
  }

  // Clean up loaded models
  async dispose() {
    for (const [name, { context, model }] of this.loadedModels) {
      try {
        // Clear history before disposal
        try {
          const sequence = context.getSequence();
          sequence.clearHistory();
        } catch (e) {
          // Sequence might already be disposed
        }
        await context.dispose();
        await model.dispose();
      } catch (error) {
        console.error(`Error disposing model ${name}:`, error);
      }
    }
    this.loadedModels.clear();
  }
}

// Example usage
async function main() {
  const multiModel = new MultiModelInference();
  await multiModel.initialize();

  // Check available models
  console.log('🔍 Checking available models on system...');
  const available = await multiModel.getAvailableModels();
  console.log('Available:', available);

  // Configure which models to use
  multiModel.setModelEnabled('qwen2.5:0.5b', true);
  // Uncomment to enable more models:
  // multiModel.setModelEnabled('tinyllama', true);

  multiModel.listModels();

  const prompt = 'Explain what TypeScript is in 2 sentences.';
  console.log(`\n📝 Prompt: "${prompt}"\n`);

  // Example 1: Parallel inference (faster, no streaming)
  console.log('\n🔄 PARALLEL INFERENCE:');
  const parallelResults = await multiModel.inferParallel(prompt);
  multiModel.displayComparison(parallelResults);

  // Example 2: Sequential inference (with streaming)
  console.log('\n\n🔄 SEQUENTIAL INFERENCE (with streaming):');
  await multiModel.inferSequential(prompt);

  // Example 3: Single model selection
  console.log('\n\n🎯 SINGLE MODEL INFERENCE:');
  const singleResult = await multiModel.inferSingle('qwen2.5:0.5b', 'What is 2+2?', true);
  console.log(`\nDuration: ${singleResult.duration}ms`);

  // Clean up
  await multiModel.dispose();
}

export { MultiModelInference, ModelConfig };