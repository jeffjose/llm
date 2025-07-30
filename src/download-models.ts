#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { spawn } from 'child_process';
import https from 'https';
import { pipeline } from 'stream/promises';

// Helper function to parse size strings to bytes
function parseSizeToBytes(sizeStr: string): number | undefined {
  const match = sizeStr.match(/^([\d.]+)\s*(MB|GB)$/i);
  if (!match) return undefined;
  
  const value = parseFloat(match[1]);
  const unit = match[2].toUpperCase();
  
  if (unit === 'MB') return Math.floor(value * 1024 * 1024);
  if (unit === 'GB') return Math.floor(value * 1024 * 1024 * 1024);
  return undefined;
}

interface ModelInfo {
  name: string;
  displayName: string;
  filename: string;
  size: string;
  url: string;
  fallbackUrl?: string;  // Alternative URL if primary fails
  description: string;
  architecture?: string;
  releaseDate?: string;
  expectedSizeBytes?: number;
}

const MODELS: ModelInfo[] = [
  // Tiny models (< 500MB)
  {
    name: 'smollm:135m',
    displayName: 'SmolLM 135M',
    filename: 'SmolLM-135M-Instruct-Q8_0.gguf',
    size: '150MB',
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/resolve/main/smollm-135m-instruct-q8_0-gguf.gguf',
    description: 'Ultra-lightweight model for edge devices.',
    architecture: 'SmolLM (HuggingFace)',
    releaseDate: '2024'
  },
  {
    name: 'smollm:360m',
    displayName: 'SmolLM 360M',
    filename: 'SmolLM-360M-Instruct-Q4_K_M.gguf',
    size: '250MB',
    url: 'https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct-GGUF/resolve/main/smollm-360m-instruct-q4_k_m.gguf',
    description: 'Lightweight model with improved reasoning.',
    architecture: 'SmolLM (HuggingFace)',
    releaseDate: '2024'
  },
  {
    name: 'qwen2.5:0.5b',
    displayName: 'Qwen2.5 0.5B',
    filename: 'Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    size: '397MB',
    url: 'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf',
    fallbackUrl: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    description: 'Smallest, fastest model. Good for testing and basic tasks.',
    architecture: 'Qwen2.5 (Transformer)',
    releaseDate: '2024',
    expectedSizeBytes: 416611552  // 397MB
  },
  
  // Small models (500MB - 1GB)
  {
    name: 'openhermes:0.77b',
    displayName: 'OpenHermes 0.77B',
    filename: 'openhermes-0.77b-v3.5-Q4_K_M.gguf',
    size: '500MB',
    url: 'https://huggingface.co/bartowski/OpenHermes-0.77B-GGUF/resolve/main/OpenHermes-0.77B-Q4_K_M.gguf',
    description: 'Function calling and structured output specialist.',
    architecture: 'Hermes (Nous Research)',
    releaseDate: '2024'
  },
  {
    name: 'tinyllama',
    displayName: 'TinyLlama 1.1B',
    filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    size: '669MB',
    url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    description: 'Efficient small model with good quality for its size.'
  },
  {
    name: 'stablelm:1.6b',
    displayName: 'StableLM Zephyr 1.6B',
    filename: 'stablelm-zephyr-1_6b-Q4_K_M.gguf',
    size: '950MB',
    url: 'https://huggingface.co/TheBloke/stablelm-zephyr-1.6b-GGUF/resolve/main/stablelm-zephyr-1.6b.Q4_K_M.gguf',
    description: 'Instruction following and coding specialist.',
    architecture: 'StableLM (Stability AI)',
    releaseDate: '2024'
  },
  {
    name: 'phi3:mini',
    displayName: 'Phi-3 Mini 3.8B',
    filename: 'Phi-3-mini-4k-instruct-q4.gguf',
    size: '2.3GB',
    url: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf',
    fallbackUrl: 'https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf',
    description: 'Microsoft\'s efficient model with strong reasoning.',
    expectedSizeBytes: 2469606195  // 2.3GB
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
    description: 'Balanced model with good quality and reasonable size.',
    expectedSizeBytes: 2147483648  // 2.0GB
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
    url: 'https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf',
    fallbackUrl: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    description: 'Alibaba\'s multilingual model with good performance.',
    expectedSizeBytes: 1073741824  // 1.0GB
  },
  
  // Large models (3GB+)
  {
    name: 'mistral:7b',
    displayName: 'Mistral 7B v0.2',
    filename: 'mistral-7b-instruct-v0.3.Q4_K_M.gguf',
    size: '4.1GB',
    url: 'https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf',
    description: 'High-quality 7B model with excellent performance.',
    expectedSizeBytes: 4368439584  // 4.1GB
  },
  {
    name: 'qwen2.5:7b',
    displayName: 'Qwen2.5 7B',
    filename: 'Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    size: '4.7GB',
    url: 'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
    description: 'Large multilingual model with state-of-the-art performance.',
    expectedSizeBytes: 4683074240
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
    
    // Check for incomplete downloads
    await this.checkIncompleteDownloads();
    
    await this.showMenu();
  }

  async checkIncompleteDownloads() {
    const files = fs.readdirSync(this.modelsDir);
    const incompleteFiles: string[] = [];
    
    // Check for .tmp files
    files.forEach(file => {
      if (file.endsWith('.tmp')) {
        incompleteFiles.push(file);
      }
    });
    
    // Check for empty model files
    MODELS.forEach(model => {
      const filePath = path.join(this.modelsDir, model.filename);
      if (fs.existsSync(filePath)) {
        const stats = fs.statSync(filePath);
        if (stats.size === 0) {
          incompleteFiles.push(model.filename);
        }
      }
    });
    
    if (incompleteFiles.length > 0) {
      console.log('\n⚠️  Found incomplete downloads:');
      incompleteFiles.forEach(file => {
        console.log(`   - ${file}`);
      });
      
      const cleanup = await this.question('\nClean up incomplete downloads? (y/n): ');
      if (cleanup.toLowerCase() === 'y') {
        incompleteFiles.forEach(file => {
          const filePath = path.join(this.modelsDir, file);
          if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
            console.log(`   🗑️  Deleted: ${file}`);
          }
        });
        console.log('\n✅ Cleanup complete!');
      }
    }
  }


  displayModel(model: ModelInfo, index: number) {
    const filePath = path.join(this.modelsDir, model.filename);
    const exists = fs.existsSync(filePath);
    let status = exists ? '✅' : '  ';
    
    // Check if file size is correct for downloaded models
    if (exists && model.expectedSizeBytes) {
      const stats = fs.statSync(filePath);
      if (stats.size === 0 || Math.abs(stats.size - model.expectedSizeBytes) > model.expectedSizeBytes * 0.1) {
        status = '⚠️ '; // Warning emoji for incomplete downloads
      }
    }
    
    console.log(`${status} ${index}. ${model.displayName} (${model.size}) - ${model.description}`);
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
      const success = await this.downloadModel(model.url, filePath, model.displayName, model.expectedSizeBytes, model.fallbackUrl);
      if (success) {
        await this.showPostDownloadOptions(model);
      }
    } else {
      console.log('\n📥 Manual download commands:\n');
      console.log('Option 1 - Using wget:');
      console.log(`wget -O "${filePath}" "${model.url}"`);
      if (model.fallbackUrl) {
        console.log('\nAlternative URL (if primary fails):');
        console.log(`wget -O "${filePath}" "${model.fallbackUrl}"`);
      }
      console.log('\nOption 2 - Using curl:');
      console.log(`curl -L -o "${filePath}" "${model.url}"`);
      await this.question('\nPress Enter to continue...');
    }
  }

  private async downloadModel(url: string, filePath: string, modelName: string, expectedSize?: number, fallbackUrl?: string): Promise<boolean> {
    console.log(`\n⏳ Downloading ${modelName}...`);
    console.log('This may take a few minutes depending on your connection speed.\n');

    const tempFilePath = `${filePath}.tmp`;
    let attempts = 0;
    const maxAttempts = 3;

    while (attempts < maxAttempts) {
      try {
        // Clean up any incomplete downloads
        if (fs.existsSync(tempFilePath)) {
          fs.unlinkSync(tempFilePath);
        }

        let downloadUrl = url;
        let usingFallback = false;
        
        try {
          // Try using system wget first (fastest)
          const wgetAvailable = await this.checkCommand('wget');
          if (wgetAvailable) {
            await this.downloadWithWget(downloadUrl, tempFilePath);
          } else {
            // Try curl as fallback
            const curlAvailable = await this.checkCommand('curl');
            if (curlAvailable) {
              await this.downloadWithCurl(downloadUrl, tempFilePath);
            } else {
              // Use Node.js HTTPS as last resort
              console.log('Using built-in downloader (this may be slower)...');
              await this.downloadWithNode(downloadUrl, tempFilePath);
            }
          }
        } catch (downloadError: any) {
          // Check if it's an authentication error
          if ((downloadError.message.includes('401') || downloadError.message.includes('403')) && fallbackUrl && !usingFallback) {
            console.log('\n⚠️  Primary URL requires authentication, trying alternative source...');
            downloadUrl = fallbackUrl;
            usingFallback = true;
            
            // Retry with fallback URL
            const wgetAvailable = await this.checkCommand('wget');
            if (wgetAvailable) {
              await this.downloadWithWget(downloadUrl, tempFilePath);
            } else {
              const curlAvailable = await this.checkCommand('curl');
              if (curlAvailable) {
                await this.downloadWithCurl(downloadUrl, tempFilePath);
              } else {
                await this.downloadWithNode(downloadUrl, tempFilePath);
              }
            }
          } else {
            throw downloadError;
          }
        }

        // Verify download
        if (!fs.existsSync(tempFilePath)) {
          throw new Error('Download file not found after completion');
        }

        const fileStats = fs.statSync(tempFilePath);
        if (fileStats.size === 0) {
          throw new Error('Downloaded file is empty');
        }

        if (expectedSize && Math.abs(fileStats.size - expectedSize) > expectedSize * 0.01) {
          // Allow 1% tolerance for size differences
          console.warn(`⚠️  File size differs from expected. Expected: ${(expectedSize / 1024 / 1024 / 1024).toFixed(2)} GB, Got: ${(fileStats.size / 1024 / 1024 / 1024).toFixed(2)} GB`);
          console.log('This might be due to model updates. Proceeding anyway...');
        }

        // Move temp file to final location
        fs.renameSync(tempFilePath, filePath);
        console.log(`\n✅ Successfully downloaded ${modelName}!`);
        console.log(`📏 File size: ${(fileStats.size / 1024 / 1024 / 1024).toFixed(2)} GB`);
        return true;
      } catch (error: any) {
        attempts++;
        console.error(`\n❌ Download attempt ${attempts} failed: ${error.message}`);
        
        // Clean up failed download
        if (fs.existsSync(tempFilePath)) {
          fs.unlinkSync(tempFilePath);
        }

        if (attempts < maxAttempts) {
          console.log(`\n🔄 Retrying download (attempt ${attempts + 1}/${maxAttempts})...`);
          await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before retry
        } else {
          console.log('\n❌ Download failed after all attempts.');
          console.log('Please try downloading manually with the commands provided.');
          return false;
        }
      }
    }
    return false;
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
        '--continue',  // Resume partial downloads
        '--retry-connrefused',  // Retry on connection refused
        '--tries=3',  // Number of retries
        '-O', filePath,
        url
      ], { stdio: 'inherit' });

      wget.on('error', reject);
      wget.on('exit', (code) => {
        if (code === 0) {
          resolve();
        } else if (code === 6) {
          reject(new Error('401 Unauthorized - Authentication required'));
        } else if (code === 8) {
          reject(new Error('404 Not Found - File not available'));
        } else {
          reject(new Error(`wget exited with code ${code}`));
        }
      });
    });
  }

  private downloadWithCurl(url: string, filePath: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const curl = spawn('curl', [
        '-L',  // Follow redirects
        '--progress-bar',
        '--retry', '3',  // Retry 3 times
        '--retry-delay', '2',  // Wait 2 seconds between retries
        '--retry-max-time', '120',  // Max time for retries
        '--fail',  // Fail on HTTP errors
        '-C', '-',  // Continue partial downloads
        '-o', filePath,
        url
      ], { stdio: 'inherit' });

      curl.on('error', reject);
      curl.on('exit', (code) => {
        if (code === 0) {
          resolve();
        } else if (code === 22) {
          reject(new Error('401 Unauthorized - Authentication required'));
        } else {
          reject(new Error(`curl exited with code ${code}`));
        }
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
          if (response.statusCode === 401 || response.statusCode === 403) {
            reject(new Error(`${response.statusCode} - Authentication required`));
          } else {
            reject(new Error(`HTTP ${response.statusCode}`));
          }
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
    
    // Show models with proper numbering
    console.log('🔸 Tiny Models (< 500MB):');
    MODELS.forEach((model, index) => {
      if (model.size.includes('MB') && parseInt(model.size) < 500) {
        this.displayModel(model, index + 1);
      }
    });
    
    console.log('\n🔹 Small Models (500MB - 1GB):');
    MODELS.forEach((model, index) => {
      const size = parseInt(model.size);
      if (model.size.includes('MB') && size >= 500) {
        this.displayModel(model, index + 1);
      }
    });
    
    console.log('\n🔷 Medium Models (1GB - 3GB):');
    MODELS.forEach((model, index) => {
      const size = parseFloat(model.size);
      if (model.size.includes('GB') && size >= 1 && size < 3) {
        this.displayModel(model, index + 1);
      }
    });
    
    console.log('\n🔶 Large Models (3GB+):');
    MODELS.forEach((model, index) => {
      const size = parseFloat(model.size);
      if (model.size.includes('GB') && size >= 3) {
        this.displayModel(model, index + 1);
      }
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