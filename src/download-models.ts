#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';
import https from 'https';
import { pipeline } from 'stream/promises';

interface ModelInfo {
  name: string;
  displayName: string;
  filename: string;
  size: string;
  url: string;
  description: string;
  architecture?: string;
  releaseDate?: string;
}

const MODELS: ModelInfo[] = [
  // Tiny models (< 500MB)
  {
    name: 'qwen2.5:0.5b',
    displayName: 'Qwen2.5 0.5B',
    filename: 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    size: '397MB',
    url: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    description: 'Smallest, fastest model. Good for testing and basic tasks.',
    architecture: 'Qwen2.5 (Transformer)',
    releaseDate: '2024'
  },
  
  // Small models (500MB - 1GB)  
  {
    name: 'tinyllama',
    displayName: 'TinyLlama 1.1B',
    filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    size: '669MB',
    url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    description: 'Efficient small model with good quality for its size.'
  },
  {
    name: 'phi3:mini',
    displayName: 'Phi-3 Mini 3.8B',
    filename: 'Phi-3-mini-4k-instruct-q4.gguf',
    size: '2.3GB',
    url: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
    description: 'Microsoft\'s efficient model with strong reasoning.'
  },
  
  // Medium models (1GB - 3GB)
  {
    name: 'llama3.2:1b',
    displayName: 'Llama 3.2 1B',
    filename: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    size: '760MB',
    url: 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    description: 'Meta\'s latest small model with improved performance.'
  },
  {
    name: 'llama3.2:3b',
    displayName: 'Llama 3.2 3B',
    filename: 'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    size: '2.0GB',
    url: 'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    description: 'Balanced model with good quality and reasonable size.'
  },
  {
    name: 'gemma2:2b',
    displayName: 'Gemma 2 2B',
    filename: 'gemma-2-2b-it-Q4_K_M.gguf',
    size: '1.6GB',
    url: 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
    description: 'Google\'s efficient model optimized for chat.'
  },
  {
    name: 'qwen2.5:1.5b',
    displayName: 'Qwen2.5 1.5B',
    filename: 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf',
    size: '1.0GB',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    description: 'Alibaba\'s multilingual model with good performance.'
  },
  
  // Large models (3GB+)
  {
    name: 'mistral:7b',
    displayName: 'Mistral 7B v0.3',
    filename: 'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
    size: '4.4GB',
    url: 'https://huggingface.co/MistralAI/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/mistral-7b-instruct-v0.3.Q4_K_M.gguf',
    description: 'High-quality 7B model with excellent performance.'
  },
  {
    name: 'qwen2.5:7b',
    displayName: 'Qwen2.5 7B',
    filename: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    size: '4.7GB',
    url: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf',
    description: 'Large multilingual model with state-of-the-art performance.'
  },
  {
    name: 'deepseek-r1:1.5b',
    displayName: 'DeepSeek R1 1.5B',
    filename: 'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
    size: '1.0GB',
    url: 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
    description: 'DeepSeek\'s reasoning model distilled to 1.5B parameters.'
  },
  {
    name: 'qwen3:4b',
    displayName: 'Qwen3 4B',
    filename: 'Qwen3-4B-Q4_K_M.gguf',
    size: '2.4GB',
    url: 'https://huggingface.co/bartowski/Qwen_Qwen3-4B-GGUF/resolve/main/Qwen_Qwen3-4B-Q4_K_M.gguf',
    description: 'Latest Qwen3 model with thinking mode for complex reasoning.'
  }
];

class ModelDownloader {
  private rl: readline.Interface;
  private modelsDir: string;

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    this.modelsDir = path.join(process.cwd(), 'models');
  }

  async start() {
    console.log('🤖 LLM Model Downloader');
    console.log('═'.repeat(50));
    
    // Create models directory if it doesn't exist
    if (!fs.existsSync(this.modelsDir)) {
      fs.mkdirSync(this.modelsDir, { recursive: true });
      console.log(`\n📁 Created models directory: ${this.modelsDir}`);
    }
    
    await this.showMenu();
  }


  displayModel(model: ModelInfo, index: number) {
    const exists = fs.existsSync(path.join(this.modelsDir, model.filename));
    const status = exists ? '✅' : '  ';
    console.log(`${status} ${index}. ${model.displayName} (${model.size}) - ${model.description}`);
    if (model.architecture && model.releaseDate) {
      console.log(`      📅 ${model.releaseDate} | 🏗️ ${model.architecture}`);
    }
  }

  async handleModelSelection(model: ModelInfo) {
    const filePath = path.join(this.modelsDir, model.filename);
    
    console.log('\n' + '─'.repeat(80));
    console.log(`📦 ${model.displayName}`);
    console.log(`📏 Size: ${model.size}`);
    console.log(`📝 Description: ${model.description}`);
    console.log(`📂 File: ${model.filename}`);
    
    if (fs.existsSync(filePath)) {
      console.log('\n✅ This model is already downloaded!');
      const overwrite = await this.question('Do you want to re-download it? (y/n): ');
      if (overwrite.toLowerCase() !== 'y') {
        await this.showPostDownloadOptions(model);
        return;
      }
    }
    
    const downloadChoice = await this.question('\nDownload now? (y/n): ');
    
    if (downloadChoice.toLowerCase() === 'y') {
      const success = await this.downloadModel(model.url, filePath, model.displayName);
      if (success) {
        await this.showPostDownloadOptions(model);
      }
    } else {
      console.log('\n📥 Manual download commands:\n');
      console.log('Option 1 - Using wget:');
      console.log(`wget -O "${filePath}" "${model.url}"`);
      console.log('\nOption 2 - Using curl:');
      console.log(`curl -L -o "${filePath}" "${model.url}"`);
      await this.question('\nPress Enter to continue...');
    }
  }

  private async downloadModel(url: string, filePath: string, modelName: string): Promise<boolean> {
    console.log(`\n⏳ Downloading ${modelName}...`);
    console.log('This may take a few minutes depending on your connection speed.\n');

    try {
      // Try using system wget first (fastest)
      const wgetAvailable = await this.checkCommand('wget');
      if (wgetAvailable) {
        await this.downloadWithWget(url, filePath);
        console.log(`\n✅ Successfully downloaded ${modelName}!`);
        return true;
      }

      // Try curl as fallback
      const curlAvailable = await this.checkCommand('curl');
      if (curlAvailable) {
        await this.downloadWithCurl(url, filePath);
        console.log(`\n✅ Successfully downloaded ${modelName}!`);
        return true;
      }

      // Use Node.js HTTPS as last resort
      console.log('Using built-in downloader (this may be slower)...');
      await this.downloadWithNode(url, filePath);
      console.log(`\n✅ Successfully downloaded ${modelName}!`);
      return true;
    } catch (error: any) {
      console.error(`\n❌ Download failed: ${error.message}`);
      console.log('Please try downloading manually with the commands provided.');
      return false;
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

  async showPostDownloadOptions(model: ModelInfo) {
    console.log('\n' + '═'.repeat(50));
    console.log('What would you like to do next?');
    console.log('\n1. 💬 Chat with this model');
    console.log('2. 📥 Download another model');
    console.log('3. 🚪 Exit');
    
    const choice = await this.question('\nSelect option (1-3): ');
    
    switch (choice) {
      case '1':
        console.log('\n🚀 Launching chat mode...');
        this.rl.close();
        // Launch the interactive multi-model tool in chat mode
        spawn('node', ['src/interactive-multi-model.ts'], { 
          stdio: 'inherit',
          env: { ...process.env, AUTO_CHAT_MODEL: model.name }
        });
        break;
      case '2':
        // Continue with menu
        break;
      case '3':
        this.rl.close();
        process.exit(0);
        break;
      default:
        console.log('❌ Invalid selection');
        await this.showPostDownloadOptions(model);
    }
  }

  async showMenu() {
    console.log('\n📋 Available Models:\n');
    
    // Group by size
    console.log('🔸 Tiny Models (< 500MB):');
    MODELS.filter(m => m.size.includes('MB') && parseInt(m.size) < 500).forEach((model, i) => {
      this.displayModel(model, i + 1);
    });
    
    console.log('\n🔹 Small Models (500MB - 1GB):');
    MODELS.filter(m => {
      const size = parseInt(m.size);
      return m.size.includes('MB') && size >= 500 && size < 1000;
    }).forEach((model, i) => {
      this.displayModel(model, i + 1);
    });
    
    console.log('\n🔷 Medium Models (1GB - 3GB):');
    MODELS.filter(m => {
      const size = parseFloat(m.size);
      return m.size.includes('GB') && size >= 1 && size < 3;
    }).forEach((model, i) => {
      this.displayModel(model, i + 1);
    });
    
    console.log('\n🔶 Large Models (3GB+):');
    MODELS.filter(m => {
      const size = parseFloat(m.size);
      return m.size.includes('GB') && size >= 3;
    }).forEach((model, i) => {
      this.displayModel(model, i + 1);
    });
    
    console.log('\n0. Exit');
    
    const choice = await this.question('\nSelect model number to download (0 to exit): ');
    
    if (choice === '0') {
      this.rl.close();
      process.exit(0);
      return;
    }
    
    const modelIndex = parseInt(choice) - 1;
    if (modelIndex >= 0 && modelIndex < MODELS.length) {
      await this.handleModelSelection(MODELS[modelIndex]);
    } else {
      console.log('❌ Invalid selection');
    }
    
    await this.showMenu();
  }

  question(prompt: string): Promise<string> {
    return new Promise((resolve) => {
      this.rl.question(prompt, resolve);
    });
  }
}

// Run the downloader
const downloader = new ModelDownloader();
downloader.start().catch(console.error);