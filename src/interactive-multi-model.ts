import { MultiModelInference } from './multi-model';
import * as readline from 'readline';
import { LlamaChatSession, LlamaContext } from 'node-llama-cpp';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import https from 'https';
import { pipeline } from 'stream/promises';

// Interactive CLI for multi-model inference
class InteractiveMultiModel {
  private multiModel: MultiModelInference;
  private rl: readline.Interface;
  private currentChatSession?: { session: LlamaChatSession; context: LlamaContext; modelName: string };

  constructor() {
    this.multiModel = new MultiModelInference();
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  private parseSizeToBytes(sizeStr: string): number {
    const match = sizeStr.match(/^([\d.]+)\s*(MB|GB)$/i);
    if (!match) return 0;
    
    const value = parseFloat(match[1]);
    const unit = match[2].toUpperCase();
    
    if (unit === 'MB') return Math.floor(value * 1024 * 1024);
    if (unit === 'GB') return Math.floor(value * 1024 * 1024 * 1024);
    return 0;
  }

  private displayCompactModel(config: any, index: number) {
    console.log(`${index}. ${config.displayName} (${config.size}) - ${config.bestFor}`);
  }

  async start() {
    console.log('🤖 Multi-Model LLM Inference Tool');
    console.log('═'.repeat(50));
    
    // Initialize the multi-model system
    await this.multiModel.initialize();
    
    // Check available models
    const available = await this.multiModel.getAvailableModels();
    console.log('\n📦 Available models:', available.join(', '));
    
    // Check if launched from download-models with a specific model
    const autoChatModel = process.env.AUTO_CHAT_MODEL;
    if (autoChatModel && available.includes(autoChatModel)) {
      console.log(`\n🚀 Auto-starting chat with ${autoChatModel}...`);
      await this.chatModeWithModel(autoChatModel);
      return;
    }
    
    await this.showMenu();
  }

  async showMenu() {
    console.log('\n📋 MAIN MENU:');
    console.log('1. 💬 Chat mode (with context)');
    console.log('2. 📥 Download models');
    console.log('3. 🚀 Parallel inference (all models)');
    console.log('4. 📝 Sequential inference (streaming)');
    console.log('5. 🎯 Single model inference');
    console.log('6. 📊 Benchmark models');
    console.log('7. Exit');

    const choice = await this.question('\nSelect option (1-7): ');

    switch (choice) {
      case '1':
        await this.chatMode();
        break;
      case '2':
        await this.downloadModels();
        break;
      case '3':
        await this.runParallel();
        break;
      case '4':
        await this.runSequential();
        break;
      case '5':
        await this.runSingle();
        break;
      case '6':
        await this.benchmark();
        break;
      case '7':
        await this.multiModel.dispose();
        this.rl.close();
        return;
      default:
        console.log('Invalid option');
    }

    await this.showMenu();
  }

  async downloadModels() {
    const modelsDir = path.join(process.cwd(), 'models');
    
    // Create models directory if it doesn't exist
    if (!fs.existsSync(modelsDir)) {
      fs.mkdirSync(modelsDir, { recursive: true });
      console.log(`\n📁 Created models directory: ${modelsDir}`);
    }

    // Model download information
    const downloadInfo = [
      {
        model: this.multiModel.getModelConfig('qwen2.5:0.5b')!,
        url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf'
      },
      {
        model: this.multiModel.getModelConfig('smollm:135m')!,
        url: 'https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/resolve/main/smollm-135m-instruct-q8_0-gguf.gguf'
      },
      {
        model: this.multiModel.getModelConfig('smollm:360m')!,
        url: 'https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct-GGUF/resolve/main/smollm-360m-instruct-q4_k_m.gguf'
      },
      {
        model: this.multiModel.getModelConfig('tinyllama')!,
        url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('stablelm:1.6b')!,
        url: 'https://huggingface.co/TheBloke/stablelm-zephyr-1.6b-GGUF/resolve/main/stablelm-zephyr-1.6b.Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('openhermes:0.77b')!,
        url: 'https://huggingface.co/bartowski/OpenHermes-0.77B-GGUF/resolve/main/OpenHermes-0.77B-Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('llama3.2:1b')!,
        url: 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('llama3.2:3b')!,
        url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('gemma2:2b')!,
        url: 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('qwen2.5:1.5b')!,
        url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf'
      },
      {
        model: this.multiModel.getModelConfig('phi3:mini')!,
        url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf'
      },
      {
        model: this.multiModel.getModelConfig('mistral:7b')!,
        url: 'https://huggingface.co/MistralAI/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('qwen2.5:7b')!,
        url: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf'
      },
      {
        model: this.multiModel.getModelConfig('deepseek-r1:1.5b')!,
        url: 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf'
      },
      {
        model: this.multiModel.getModelConfig('qwen3:4b')!,
        url: 'https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf'
      }
    ];

    console.log('\n📥 DOWNLOAD MODELS');
    console.log('═'.repeat(50));
    
    // Check how many models are already downloaded
    const downloadedCount = downloadInfo.filter(info => 
      fs.existsSync(path.join(modelsDir, info.model.filename))
    ).length;
    
    console.log(`\n📊 Status: ${downloadedCount}/${downloadInfo.length} models downloaded\n`);
    
    // Helper function to parse size
    const getSizeInMB = (sizeStr: string): number => {
      if (sizeStr.includes('GB')) {
        return parseFloat(sizeStr) * 1024;
      }
      return parseInt(sizeStr);
    };

    // Sort models by size
    const sortedInfo = [...downloadInfo].sort((a, b) => 
      getSizeInMB(a.model.size) - getSizeInMB(b.model.size)
    );

    // Display models organized by size categories with sequential numbering
    let lastCategory = '';
    sortedInfo.forEach((info, index) => {
      const sizeInMB = getSizeInMB(info.model.size);
      let category = '';
      
      if (sizeInMB < 500) {
        category = '🔸 Tiny Models (< 500MB):';
      } else if (sizeInMB < 1024) {
        category = '🔹 Small Models (500MB - 1GB):';
      } else if (sizeInMB < 3072) {
        category = '🔷 Medium Models (1GB - 3GB):';
      } else {
        category = '🔶 Large Models (3GB+):';
      }
      
      if (category !== lastCategory) {
        console.log(lastCategory ? '\n' + category : category);
        lastCategory = category;
      }
      
      // Use sequential numbering for display
      this.displayModelStatus(info, index + 1, modelsDir);
    });

    console.log('\n0. Back to main menu');
    
    const choice = await this.question('\nSelect model to download (0 to go back): ');
    
    if (choice === '0') {
      return;
    }

    const index = parseInt(choice) - 1;
    if (index >= 0 && index < sortedInfo.length) {
      const info = sortedInfo[index];
      const filePath = path.join(modelsDir, info.model.filename);
      
      if (fs.existsSync(filePath)) {
        console.log('\n✅ This model is already downloaded!');
        const overwrite = await this.question('Do you want to re-download it? (y/n): ');
        if (overwrite.toLowerCase() !== 'y') {
          await this.downloadModels();
          return;
        }
      }

      console.log(`\n📦 Model: ${info.model.displayName}`);
      console.log(`📏 Size: ${info.model.size}`);
      console.log(`📂 File: ${info.model.filename}`);
      
      const downloadChoice = await this.question('\nDownload now? (y/n): ');
      
      if (downloadChoice.toLowerCase() === 'y') {
        await this.downloadModel(info.url, filePath, info.model.displayName);
      } else {
        console.log('\n📥 Manual download commands:');
        console.log(`wget -O "${filePath}" "${info.url}"`);
        console.log(`curl -L -o "${filePath}" "${info.url}"`);
        await this.question('\nPress Enter to continue...');
      }
    } else {
      console.log('❌ Invalid selection');
    }
    
    await this.downloadModels();
  }


  async runParallel() {
    // Get available models
    const available = await this.multiModel.getAvailableModels();
    if (available.length === 0) {
      console.log('❌ No models available. Please download models first.');
      return;
    }
    
    console.log('\n📦 Available models for parallel inference:');
    available.forEach((model, index) => {
      console.log(`${index + 1}. ${model}`);
    });
    
    // Enable all available models for parallel inference
    available.forEach(modelName => {
      this.multiModel.setModelEnabled(modelName, true);
    });
    
    const prompt = await this.question('\nEnter prompt: ');
    console.log('\n🚀 Running parallel inference on all available models...\n');
    
    const results = await this.multiModel.inferParallel(prompt);
    
    // Always use compact format with inline timings for parallel inference
    this.multiModel.displayComparison(results, true);
    
    // Reset to default enabled state
    this.multiModel.models.forEach(model => {
      model.enabled = model.name === 'qwen2.5:0.5b';
    });
  }

  async runSequential() {
    // Get available models
    const available = await this.multiModel.getAvailableModels();
    if (available.length === 0) {
      console.log('❌ No models available. Please download models first.');
      return;
    }
    
    console.log('\n📦 Available models for sequential inference:');
    available.forEach((model, index) => {
      console.log(`${index + 1}. ${model}`);
    });
    
    // Enable all available models for sequential inference
    available.forEach(modelName => {
      this.multiModel.setModelEnabled(modelName, true);
    });
    
    const prompt = await this.question('\nEnter prompt: ');
    console.log('\n🚀 Running sequential inference on all available models...\n');
    
    await this.multiModel.inferSequential(prompt);
    
    // Reset to default enabled state
    this.multiModel.models.forEach(model => {
      model.enabled = model.name === 'qwen2.5:0.5b';
    });
  }

  async runSingle() {
    const model = await this.question('\nEnter model name (e.g., qwen2.5:0.5b): ');
    const prompt = await this.question('Enter prompt: ');
    
    console.log('\n📝 Running inference...\n');
    const result = await this.multiModel.inferSingle(model, prompt, true);
    console.log(`\n⏱️  Duration: ${result.duration}ms`);
  }

  async benchmark() {
    console.log('\n📊 Running benchmark...\n');
    
    const prompts = [
      'What is 2+2?',
      'Write a haiku about coding',
      'Explain recursion in one sentence'
    ];

    for (const prompt of prompts) {
      console.log(`\n📝 Prompt: "${prompt}"`);
      const results = await this.multiModel.inferParallel(prompt);
      
      // Sort by speed
      results.sort((a, b) => a.duration - b.duration);
      
      console.log('\n🏆 Speed Ranking:');
      results.forEach((result, index) => {
        console.log(`${index + 1}. ${result.model}: ${result.duration}ms`);
      });
    }
  }

  async chatModeWithModel(modelName: string) {
    console.log('\n💬 CHAT MODE');
    console.log('═'.repeat(50));
    
    const modelConfig = this.multiModel.getModelConfig(modelName);
    if (modelConfig) {
      console.log(`\n📦 Model: ${modelConfig.displayName} (${modelConfig.size})`);
      console.log(`📅 Released: ${modelConfig.releaseDate} | 🏗️ ${modelConfig.architecture}`);
      console.log(`💡 Capabilities: ${modelConfig.capabilities}`);
      console.log(`✨ Best for: ${modelConfig.bestFor}`);
    }
    
    console.log(`\n✅ Starting chat with ${modelName}`);
    console.log('💡 Commands: /clear (new conversation), /model (change model), /exit (return to menu)');
    console.log('─'.repeat(50));

    // Initialize chat session
    await this.initializeChatSession(modelName);

    // Chat loop
    while (true) {
      const input = await this.question('\n👤 You: ');
      
      // Handle commands
      if (input.startsWith('/')) {
        const command = input.toLowerCase();
        
        if (command === '/exit') {
          console.log('👋 Exiting chat mode...');
          if (this.currentChatSession) {
            await this.currentChatSession.context.dispose();
            this.currentChatSession = undefined;
          }
          break;
        } else if (command === '/clear') {
          console.log('🔄 Starting new conversation...');
          if (this.currentChatSession) {
            // Get the sequence and clear its history
            const sequence = this.currentChatSession.context.getSequence();
            sequence.clearHistory();
            
            // Create a new chat session with cleared history
            this.currentChatSession.session = new LlamaChatSession({
              contextSequence: sequence,
              systemPrompt: 'You are a helpful AI assistant. Provide clear, concise, and helpful responses.'
            });
          }
          console.log('✅ Conversation cleared!');
          continue;
        } else if (command === '/model') {
          // Clean up current session
          if (this.currentChatSession) {
            await this.currentChatSession.context.dispose();
            this.currentChatSession = undefined;
          }
          // Restart model selection
          await this.chatMode();
          return;
        } else {
          console.log('❌ Unknown command. Available: /clear, /model, /exit');
          continue;
        }
      }

      // Generate response
      console.log('\n🤖 Assistant: ', '');
      try {
        await this.currentChatSession!.session.prompt(input, {
          onTextChunk: (chunk) => {
            process.stdout.write(chunk);
          }
        });
        console.log(); // New line after response
      } catch (error: any) {
        console.log(`\n❌ Error: ${error.message}`);
        console.log('💡 Try /clear to start a new conversation or /model to switch models.');
      }
    }
  }

  async chatMode() {
    console.log('\n💬 CHAT MODE');
    console.log('═'.repeat(50));
    
    // Select model for chat
    const available = await this.multiModel.getAvailableModels();
    if (available.length === 0) {
      console.log('❌ No models available. Please download a model first.');
      return;
    }

    // Get model configs and sort by size
    const modelConfigs = available
      .map(modelName => this.multiModel.getModelConfig(modelName))
      .filter(config => config !== undefined) as any[];
    
    // Sort by size (convert to bytes for proper sorting)
    modelConfigs.sort((a, b) => {
      const sizeA = this.parseSizeToBytes(a.size);
      const sizeB = this.parseSizeToBytes(b.size);
      return sizeA - sizeB;
    });
    
    console.log('\n📦 Available models for chat:');
    
    let displayIndex = 1;
    
    // Group by size categories
    const tinyModels = modelConfigs.filter(config => 
      config.size.includes('MB') && parseInt(config.size) < 500
    );
    if (tinyModels.length > 0) {
      console.log('\n🔸 Tiny Models (< 500MB):');
      tinyModels.forEach(config => {
        this.displayCompactModel(config, displayIndex++);
      });
    }
    
    const smallModels = modelConfigs.filter(config => {
      const size = parseInt(config.size);
      return config.size.includes('MB') && size >= 500;
    });
    if (smallModels.length > 0) {
      console.log('\n🔹 Small Models (500MB - 1GB):');
      smallModels.forEach(config => {
        this.displayCompactModel(config, displayIndex++);
      });
    }
    
    const mediumModels = modelConfigs.filter(config => {
      const size = parseFloat(config.size);
      return config.size.includes('GB') && size >= 1 && size < 3;
    });
    if (mediumModels.length > 0) {
      console.log('\n🔷 Medium Models (1GB - 3GB):');
      mediumModels.forEach(config => {
        this.displayCompactModel(config, displayIndex++);
      });
    }
    
    const largeModels = modelConfigs.filter(config => {
      const size = parseFloat(config.size);
      return config.size.includes('GB') && size >= 3;
    });
    if (largeModels.length > 0) {
      console.log('\n🔶 Large Models (3GB+):');
      largeModels.forEach(config => {
        this.displayCompactModel(config, displayIndex++);
      });
    }
    
    console.log();

    const modelChoice = await this.question('\nSelect model number: ');
    const modelIndex = parseInt(modelChoice) - 1;
    
    if (modelIndex < 0 || modelIndex >= modelConfigs.length) {
      console.log('❌ Invalid selection');
      return;
    }

    const selectedModel = modelConfigs[modelIndex].name;
    await this.chatModeWithModel(selectedModel);
  }

  private async initializeChatSession(modelName: string) {
    try {
      // Get the model configuration
      const modelConfig = this.multiModel.getModelConfig(modelName);
      if (!modelConfig) {
        throw new Error(`Model ${modelName} not found`);
      }

      // Load the model and create context
      const { model, context } = await this.multiModel.loadModel(modelConfig);
      
      // Create a new sequence for chat
      const sequence = context.getSequence();
      
      // Create chat session with system prompt
      const session = new LlamaChatSession({
        contextSequence: sequence,
        systemPrompt: 'You are a helpful AI assistant. Provide clear, concise, and helpful responses.'
      });

      this.currentChatSession = {
        session,
        context,
        modelName
      };

      console.log(`\n✅ Chat session initialized with ${modelName}`);
    } catch (error: any) {
      console.error(`❌ Failed to initialize chat: ${error.message}`);
      throw error;
    }
  }

  private displayModelStatus(info: any, index: number, modelsDir: string) {
    const filePath = path.join(modelsDir, info.model.filename);
    const exists = fs.existsSync(filePath);
    const status = exists ? '✅' : '  ';
    console.log(`${status} ${index}. ${info.model.displayName} (${info.model.size}) - ${info.model.bestFor}`);
  }

  private async downloadModel(url: string, filePath: string, modelName: string): Promise<void> {
    console.log(`\n⏳ Downloading ${modelName}...`);
    console.log('This may take a few minutes depending on your connection speed.\n');

    try {
      // Try using system wget first (fastest)
      const wgetAvailable = await this.checkCommand('wget');
      if (wgetAvailable) {
        await this.downloadWithWget(url, filePath);
        console.log(`\n✅ Successfully downloaded ${modelName}!`);
        return;
      }

      // Try curl as fallback
      const curlAvailable = await this.checkCommand('curl');
      if (curlAvailable) {
        await this.downloadWithCurl(url, filePath);
        console.log(`\n✅ Successfully downloaded ${modelName}!`);
        return;
      }

      // Use Node.js HTTPS as last resort
      console.log('Using built-in downloader (this may be slower)...');
      await this.downloadWithNode(url, filePath);
      console.log(`\n✅ Successfully downloaded ${modelName}!`);
    } catch (error: any) {
      console.error(`\n❌ Download failed: ${error.message}`);
      console.log('Please try downloading manually with the commands provided.');
    }
  }

  private checkCommand(command: string): Promise<boolean> {
    return new Promise((resolve) => {
      const proc = spawn(command, ['--version'], { stdio: 'ignore' });
      proc.on('error', () => resolve(false));
      proc.on('exit', (code) => resolve(code === 0));
    });
  }

  private downloadWithWget(url: string, filePath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const wget = spawn('wget', [
        '--show-progress',
        '-O', filePath,
        url
      ], { stdio: 'inherit' });

      wget.on('error', reject);
      wget.on('exit', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`wget exited with code ${code}`));
      });
    });
  }

  private downloadWithCurl(url: string, filePath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const curl = spawn('curl', [
        '-L',
        '--progress-bar',
        '-o', filePath,
        url
      ], { stdio: 'inherit' });

      curl.on('error', reject);
      curl.on('exit', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`curl exited with code ${code}`));
      });
    });
  }

  private async downloadWithNode(url: string, filePath: string): Promise<void> {
    const file = fs.createWriteStream(filePath);
    
    return new Promise((resolve, reject) => {
      https.get(url, { 
        headers: { 'User-Agent': 'llm-tool/1.0' }
      }, (response) => {
        if (response.statusCode === 302 || response.statusCode === 301) {
          // Handle redirect
          file.close();
          fs.unlinkSync(filePath);
          this.downloadWithNode(response.headers.location!, filePath)
            .then(resolve)
            .catch(reject);
          return;
        }

        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }

        const totalSize = parseInt(response.headers['content-length'] || '0');
        let downloaded = 0;
        let lastPercent = 0;

        response.on('data', (chunk) => {
          downloaded += chunk.length;
          const percent = Math.floor((downloaded / totalSize) * 100);
          
          if (percent !== lastPercent && percent % 5 === 0) {
            process.stdout.write(`\rProgress: ${percent}%`);
            lastPercent = percent;
          }
        });

        pipeline(response, file)
          .then(() => {
            console.log('\rProgress: 100%');
            resolve();
          })
          .catch(reject);
      }).on('error', reject);
    });
  }

  question(prompt: string): Promise<string> {
    return new Promise((resolve) => {
      this.rl.question(prompt, resolve);
    });
  }
}

// Check for command line arguments
async function main() {
  const tool = new InteractiveMultiModel();
  
  // Check if a model path was provided as argument
  const modelPath = process.argv[2];
  if (modelPath) {
    // If it's a model file path, download it directly
    if (modelPath.endsWith('.gguf')) {
      const modelName = path.basename(modelPath);
      const modelsDir = path.join(process.cwd(), 'models');
      
      // Ensure models directory exists
      if (!fs.existsSync(modelsDir)) {
        fs.mkdirSync(modelsDir, { recursive: true });
      }
      
      const targetPath = path.join(modelsDir, modelName);
      
      // Check if model already exists
      if (fs.existsSync(targetPath)) {
        console.log(`✅ Model ${modelName} already exists`);
        process.exit(0);
      }
      
      console.log(`📥 Downloading ${modelName}...`);
      console.log('This feature requires the model to be available via HTTP URL.');
      console.log('Please use the interactive menu to download models from the curated list.');
      process.exit(1);
    }
  }
  
  // Start interactive mode
  await tool.start();
}

main().catch(console.error);