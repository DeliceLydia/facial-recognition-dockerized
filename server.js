import express from "express";
import * as faceapi from "face-api.js";
import canvas from "canvas";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";
import multer from 'multer';

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB max
});

const { Canvas, Image, ImageData } = canvas;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Set request timeout middleware FIRST
app.use((req, res, next) => {
  res.setTimeout(30000, () => {
    console.log('⚠️  Request timeout after 30s');
    if (!res.headersSent) {
      res.status(408).json({
        success: false,
        error: "Request Timeout",
        message: "Processing took too long (>30s)"
      });
    }
  });
  next();
});

app.use(express.json({ limit: "10mb" }));

// Monkey patch face-api for Node.js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

// Models directory
const MODEL_PATH = path.join(__dirname, "models");
let modelsLoaded = false;

// Load models
async function loadModels() {
  console.log('Loading models from:', MODEL_PATH);
  
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceLandmark68TinyNet.loadFromDisk(MODEL_PATH);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_PATH);
  
  modelsLoaded = true;
  console.log("✅ Face-api models loaded successfully");
}

// Detection options - REDUCED for speed
const detectionOptions = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320, // Reduced from 416 for faster processing
  scoreThreshold: 0.5
});

// Resize image - MORE AGGRESSIVE REDUCTION
function resizeImage(img, maxSize = 416) { // Reduced from 512
  const canvas = new Canvas(maxSize, maxSize);
  const ctx = canvas.getContext('2d');
  
  let width = img.width;
  let height = img.height;
  
  // Calculate scale
  const scale = Math.min(maxSize / width, maxSize / height);
  
  width = Math.floor(width * scale);
  height = Math.floor(height * scale);
  
  canvas.width = width;
  canvas.height = height;
  
  ctx.drawImage(img, 0, 0, width, height);
  
  return canvas;
}

// Load image from URL with timeout
async function loadImageFromUrl(url) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout
  
  try {
    console.log('  ⏳ Fetching URL image...');
    const response = await fetch(url, { 
      signal: controller.signal,
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const buffer = await response.arrayBuffer();
    console.log(`  ✓ URL image fetched (${(buffer.byteLength / 1024).toFixed(0)}KB)`);
    
    const img = new Image();
    
    return new Promise((resolve, reject) => {
      const loadTimeout = setTimeout(() => {
        reject(new Error('Image decode timeout'));
      }, 5000);
      
      img.onload = () => {
        clearTimeout(loadTimeout);
        console.log(`  ✓ URL image decoded (${img.width}x${img.height})`);
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
      throw new Error('URL fetch timeout (15s)');
    }
    throw new Error(`URL image error: ${error.message}`);
  }
}

// Load image from base64 with timeout
async function loadImageFromBase64(base64String) {
  try {
    console.log('  ⏳ Decoding base64 image...');
    
    // Remove data URI prefix
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');
    console.log(`  ℹ️  Base64 size: ${(base64Data.length / 1024).toFixed(0)}KB`);
    
    const buffer = Buffer.from(base64Data, 'base64');
    console.log(`  ✓ Base64 decoded to buffer (${(buffer.length / 1024).toFixed(0)}KB)`);
    
    const img = new Image();
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Base64 image decode timeout (5s)'));
      }, 5000);
      
      img.onload = () => {
        clearTimeout(timeout);
        console.log(`  ✓ Base64 image decoded (${img.width}x${img.height})`);
        resolve(resizeImage(img));
      };
      
      img.onerror = () => {
        clearTimeout(timeout);
        reject(new Error('Invalid base64 image data'));
      };
      
      img.src = buffer;
    });
  } catch (error) {
    throw new Error(`Base64 error: ${error.message}`);
  }
}

// Compare faces with proper timeout
async function compareFaces(imageUrl, base64Image) {
  return new Promise(async (resolve, reject) => {
    // Overall timeout for comparison
    const overallTimeout = setTimeout(() => {
      reject(new Error('Face comparison timeout (25s)'));
    }, 25000);
    
    try {
      const startTime = Date.now();
      
      // Step 1: Load images
      console.log('📥 Step 1: Loading images...');
      const [img1, img2] = await Promise.all([
        loadImageFromUrl(imageUrl),
        loadImageFromBase64(base64Image)
      ]);
      console.log(`✓ Images loaded in ${Date.now() - startTime}ms`);

      // Step 2: Detect faces with timeout
      console.log('🔍 Step 2: Detecting faces...');
      const detectStart = Date.now();
      
      const detectionPromise = Promise.all([
        faceapi
          .detectSingleFace(img1, detectionOptions)
          .withFaceLandmarks(true)
          .withFaceDescriptor(),
        faceapi
          .detectSingleFace(img2, detectionOptions)
          .withFaceLandmarks(true)
          .withFaceDescriptor()
      ]);
      
      // Race between detection and timeout
      const detectionTimeout = new Promise((_, timeoutReject) => {
        setTimeout(() => timeoutReject(new Error('Face detection timeout (20s)')), 20000);
      });
      
      const [detection1, detection2] = await Promise.race([
        detectionPromise,
        detectionTimeout
      ]);
      
      console.log(`✓ Faces detected in ${Date.now() - detectStart}ms`);
      
      clearTimeout(overallTimeout);

      // Step 3: Check results
      if (!detection1 || !detection2) {
        resolve({ 
          success: false,
          match: false, 
          message: !detection1 && !detection2 
            ? "No faces detected in both images"
            : !detection1 
            ? "No face detected in URL image"
            : "No face detected in uploaded image",
          processingTimeMs: Date.now() - startTime
        });
        return;
      }

      // Step 4: Calculate similarity
      console.log('📊 Step 3: Comparing faces...');
      const distance = faceapi.euclideanDistance(
        detection1.descriptor, 
        detection2.descriptor
      );
      
      const threshold = 0.5;
      const match = distance < threshold;
      const similarity = Math.max(0, Math.min(100, (1 - distance) * 100));
      
      const totalTime = Date.now() - startTime;
      console.log(`✅ Comparison complete in ${totalTime}ms`);
      
      resolve({ 
        success: true,
        match, 
        distance: parseFloat(distance.toFixed(4)),
        similarity: parseFloat(similarity.toFixed(2)),
        threshold,
        confidence: match ? 'high' : distance < 0.7 ? 'medium' : 'low',
        processingTimeMs: totalTime
      });
      
    } catch (error) {
      clearTimeout(overallTimeout);
      reject(error);
    }
  });
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: modelsLoaded ? 'ready' : 'loading',
    timestamp: new Date().toISOString(),
    service: 'face-comparison-api',
    uptime: Math.floor(process.uptime())
  });
});

// Compare faces endpoint
app.post("/compare",  async (req, res) => {
  const requestId = Date.now();
  console.log(`🔍 [${requestId}] New comparison request`);
  
  const startTime = Date.now();
  
  try {
    // Check if models loaded
    if (!modelsLoaded) {
      return res.status(503).json({
        success: false,
        error: "Service Unavailable",
        message: "Models are still loading. Try again in a moment."
      });
    }
    
    const { imageUrl, base64Image } = req.body;
    
    

    // Validate inputs
    if (!imageUrl ||!base64Image) {
      return res.status(400).json({ 
        success: false,
        error: "Missing required field",
        message: "'imageUrl and base64Image' is required" 
      });
    }

    // Validate URL format
    if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://')) {
      return res.status(400).json({ 
        success: false,
        error: "Invalid URL",
        message: "imageUrl must start with http:// or https://" 
      });
    }

    console.log(` Request Details:`);
    console.log(`   URL: ${imageUrl}`);
    console.log(`   Base64 length: ${base64Image.length.toLocaleString()} chars`);

    // Perform comparison
    const result = await compareFaces(imageUrl, base64Image);
    
    const processingTime = Date.now() - startTime;
    
    console.log(` [${requestId}] Result: ${result.match ? 'MATCH' : ' NO MATCH'}`);
    console.log(`  Total time: ${processingTime}ms`);

    
    res.json({
      ...result,
      processingTimeMs: processingTime
    });
    
  } catch (error) {
    const processingTime = Date.now() - startTime;
    console.error(` [${requestId}] Error: ${error.message}`);
    console.error(`  Failed after: ${processingTime}ms`);
    
    if (!res.headersSent) {
      res.status(500).json({ 
        success: false,
        error: "Face comparison failed",
        message: error.message,
        processingTimeMs: processingTime
      });
    }
  }
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Cannot ${req.method} ${req.url}`
  });
});

// Start server
const PORT = process.env.PORT || 4000;
const HOST = '0.0.0.0';

const server = app.listen(PORT, HOST, async () => {
  console.log(' Face Comparison API Starting...  ');

  
  try {
    await loadModels();
    console.log(`   Server: http://localhost:${PORT} `);
    console.log(`   Health: GET /health                `);
    console.log(`  Compare: POST /compare (multipart)`);
  } catch (error) {
    console.error(' Failed to start server:', error);
    process.exit(1);
  }
});

// Set server timeouts
server.timeout = 35000; // 35 seconds
server.keepAliveTimeout = 40000;
server.headersTimeout = 41000;

// Graceful shutdown
const shutdown = () => {
  console.log('  Shutting down gracefully...');
  server.close(() => {
    console.log(' Server closed');
    process.exit(0);
  });
  
  setTimeout(() => {
    console.error(' Forced shutdown after 10s');
    process.exit(1);
  }, 10000);
};

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);