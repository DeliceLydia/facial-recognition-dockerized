# Use Node.js LTS slim for smaller image size
FROM node:18-slim

# Set production environment variables
ENV NODE_ENV=production \
    TF_CPP_MIN_LOG_LEVEL=2 \
    NODE_OPTIONS="--max-old-space-size=2048"

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    libcairo2-dev \
    libpango1.0-dev \
    libjpeg-dev \
    libgif-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy package files first (for better caching)
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production --no-audit --no-fund \
    && npm cache clean --force

# Download face-api models
RUN mkdir -p models && cd models && \
    echo "📥 Downloading tiny face detector..." && \
    wget -q --show-progress \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/tiny_face_detector_model-weights_manifest.json \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/tiny_face_detector_model-shard1 && \
    echo "📥 Downloading tiny landmarks..." && \
    wget -q --show-progress \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_landmark_68_tiny_model-weights_manifest.json \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_landmark_68_tiny_model-shard1 && \
    echo "📥 Downloading face recognition model..." && \
    wget -q --show-progress \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_recognition_model-weights_manifest.json \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_recognition_model-shard1 \
      https://github.com/justadudewhohacks/face-api.js/raw/master/weights/face_recognition_model-shard2 && \
    echo "✅ Models downloaded:" && \
    ls -lh && \
    cd ..

# Copy application code
COPY server.js ./

# Create non-root user for security
RUN groupadd -r nodejs && \
    useradd -r -g nodejs nodejs && \
    chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD node -e "require('http').get('http://localhost:4000/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Start application
CMD ["node", "server.js"]