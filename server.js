import express from "express";
import * as faceapi from "face-api.js";
import canvas from "canvas";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";

const { Canvas, Image, ImageData } = canvas;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Increase timeout slightly
app.use((req, res, next) => {
  res.setTimeout(45000, () => {
    console.log('⚠️  Request timeout after 45s');
    if (!res.headersSent) {
      res.status(408).json({
        success: false,
        error: "Request Timeout",
        message: "Processing took too long"
      });
    }
  });
  next();
});

app.use(express.json({ limit: "10mb" }));

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const MODEL_PATH = path.join(__dirname, "models");
let modelsLoaded = false;

async function loadModels() {
  console.log('Loading models from:', MODEL_PATH);
  
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
  
  modelsLoaded = true;
  console.log("✅ Face-api models loaded successfully");
}

// CRITICAL: Much smaller input size for speed
const detectionOptions = new faceapi.TinyFaceDetectorOptions({
  inputSize: 128, // REDUCED from 320 - huge speed boost
  scoreThreshold: 0.5 // Slightly lower to catch more faces
});

// AGGRESSIVE image resizing - this is the key to speed
function resizeImage(img, maxSize = 128) { // REDUCED from 416
  const canvas = new Canvas(maxSize, maxSize);
  const ctx = canvas.getContext('2d');
  
  let width = img.width;
  let height = img.height;
  
  // More aggressive scaling
  const scale = Math.min(maxSize / width, maxSize / height);
  
  width = Math.floor(width * scale);
  height = Math.floor(height * scale);
  
  canvas.width = width;
  canvas.height = height;
  
  // Use lower quality but faster rendering
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'low'; // Fast rendering
  ctx.drawImage(img, 0, 0, width, height);
  
  return canvas;
}

// Load from URL
async function loadImageFromUrl(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 10000);
  
  try {
    const response = await fetch(url, { 
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const buffer = await response.arrayBuffer();
    const img = new Image();
    
    return new Promise((resolve, reject) => {
      const loadTimeout = setTimeout(() => {
        reject(new Error('Image decode timeout'));
      }, 3000);
      
      img.onload = () => {
        clearTimeout(loadTimeout);
        resolve(resizeImage(img));
      };
      
      img.onerror = () => {
        clearTimeout(loadTimeout);
        reject(new Error('Failed to decode URL image'));
      };
      
      img.src = Buffer.from(buffer);
    });
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('URL fetch timeout');
    }
    throw error;
  }
}

// Load from base64
async function loadImageFromBase64(base64String) {
  try {
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');
    
    // For very large base64, decode in chunks (memory optimization)
    const buffer = Buffer.from(base64Data, 'base64');
    const img = new Image();
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Base64 decode timeout'));
      }, 3000);
      
      img.onload = () => {
        clearTimeout(timeout);
        resolve(resizeImage(img));
      };
      
      img.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Invalid base64 image'));
      };
      
      img.src = buffer;
    });
  } catch (error) {
    throw new Error(`Base64 error: ${error.message}`);
  }
}

// Optimized face comparison
async function compareFaces(imageUrl, base64Image) {
  const startTime = Date.now();
  
  try {
    // Load images in parallel
    console.log('⏳ Loading images...');
    const loadStart = Date.now();
    const [img1, img2] = await Promise.all([
      loadImageFromUrl(imageUrl),
      loadImageFromBase64(base64Image)
    ]);
    console.log(`✓ Images loaded: ${Date.now() - loadStart}ms`);

    // Detect faces in parallel
    console.log('⏳ Detecting faces...');
    const detectStart = Date.now();
    
    const [detection1, detection2] = await Promise.all([
      faceapi
        .detectSingleFace(img1, detectionOptions)
        .withFaceLandmarks(true)
        .withFaceDescriptor(),
      faceapi
        .detectSingleFace(img2, detectionOptions)
        .withFaceLandmarks(true)
        .withFaceDescriptor()
    ]);
    
    console.log(`✓ Faces detected: ${Date.now() - detectStart}ms`);

    if (!detection1 || !detection2) {
      return { 
        success: false,
        match: false, 
        message: !detection1 && !detection2 
          ? "No faces detected in both images"
          : !detection1 
          ? "No face detected in URL image"
          : "No face detected in base64 image",
        processingTimeMs: Date.now() - startTime
      };
    }

    // Compare
    const distance = faceapi.euclideanDistance(
      detection1.descriptor, 
      detection2.descriptor
    );
    
    const threshold = 0.5;
    const match = distance < threshold;
    const similarity = Math.max(0, Math.min(100, (1 - distance) * 100));
    
    return { 
      success: true,
      match, 
      distance: parseFloat(distance.toFixed(4)),
      similarity: parseFloat(similarity.toFixed(2)),
      threshold,
      confidence: match ? 'high' : distance < 0.7 ? 'medium' : 'low',
      processingTimeMs: Date.now() - startTime
    };
    
  } catch (error) {
    throw error;
  }
}

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: modelsLoaded ? 'ready' : 'loading',
    timestamp: new Date().toISOString(),
    uptime: Math.floor(process.uptime())
  });
});

// Compare endpoint
app.post("/compare", async (req, res) => {
  const requestId = Date.now();
  console.log(`\n🔍 [${requestId}] Comparison request`);
  
  const startTime = Date.now();
  
  try {
    if (!modelsLoaded) {
      return res.status(503).json({
        success: false,
        error: "Models still loading",
        message: "Models still loading"
      });
    }
    
    const { imageUrl, base64Image } = req.body;

    if (!imageUrl || !base64Image) {
      return res.status(400).json({ 
        success: false,
        error: "Missing imageUrl or base64Image",
        message: "Missing imageUrl or base64Image"
      });
    }

    if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) {
      return res.status(400).json({ 
        success: false,
        error: "Invalid image URL  format",
        message: "Invalid URL format"
      });
    }

    console.log(`📊 URL: ${imageUrl.substring(0, 60)}...`);
    console.log(`📊 Base64: ${(base64Image.length / 1024).toFixed(0)}KB`);

    const result = await compareFaces(imageUrl, base64Image);
    
    const totalTime = Date.now() - startTime;
    
    console.log(`✅ [${requestId}] ${result.match ? 'MATCH ✓' : 'NO MATCH ✗'} (${totalTime}ms)\n`);
    
    res.json({
      ...result,
      processingTimeMs: totalTime
    });
    
  } catch (error) {
    const totalTime = Date.now() - startTime;
    console.error(`❌ [${requestId}] Error: ${error.message} (${totalTime}ms)\n`);
    
    if (!res.headersSent) {
      res.status(500).json({ 
        success: false,
        error: error.message,
        message: error.message,
        processingTimeMs: totalTime
      });
    }
  }
});
app.post("/detect", async (req, res) => {
  const requestId = Date.now();
  console.log(`\n🔍 [${requestId}] face detect request`);
  
  const startTime = Date.now();
  
  try {
    if (!modelsLoaded) {
      return res.status(503).json({
        success: false,
        error: "Models still loading"
      });
    }
    
    const { base64Image } = req.body;

    if (!base64Image) {
      return res.status(400).json({ 
        success: false,
        error: "Missing  base64Image image",
        message: "Missing  base64Image image"
      });
    }

    console.log(`📊 Base64: ${(base64Image.length / 1024).toFixed(0)}KB`);

     console.log('⏳ Loading image...');
    const loadStart = Date.now();
    const [img1] = await Promise.all([
      loadImageFromBase64(base64Image)
    ]);
    console.log(`✓ Image loaded: ${Date.now() - loadStart}ms`);

    // Detect faces in parallel
    console.log('⏳ Detecting face...');
    
    const detection = await faceapi
        .detectSingleFace(img1, detectionOptions)
        .withFaceLandmarks(true)
        .withFaceDescriptor()
      
    
    const totalTime = Date.now() - startTime;
    
    console.log(`✅ [${requestId}] ${detection ? 'Found ✓' : 'No face found'} (${totalTime}ms)\n`);
    
    res.json({
    faceFound:detection?true:false,
      processingTimeMs: totalTime
    });
    
  } catch (error) {
    const totalTime = Date.now() - startTime;
    console.error(`❌ [${requestId}] Error: ${error.message} (${totalTime}ms)\n`);
    
    if (!res.headersSent) {
      res.status(500).json({ 
        success: false,
        error: error.message,
        message: error.message,
        processingTimeMs: totalTime
      });
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    message:"url not found",
    error: 'Not Found',
    path: req.url
  });
});

// Start server
const PORT = process.env.PORT || 4000;
const server = app.listen(PORT, '0.0.0.0', async () => {
  console.log('\n🚀 Face Comparison API');
  console.log(`   Port: ${PORT}`);
  
  try {
    await loadModels();
    console.log('   Status: ✅ Ready\n');
  } catch (error) {
    console.error('❌ Failed to start:', error);
    process.exit(1);
  }
});

server.timeout = 50000;
server.keepAliveTimeout = 55000;
server.headersTimeout = 56000;

const shutdown = () => {
  console.log('Shutting down...');
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 10000);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);