# Docker Deployment Guide for NVIDIA Jetson

This guide explains how to deploy the LIME image enhancement library on an NVIDIA Jetson device using Docker.

## Prerequisites

1. **NVIDIA Jetson device** (Jetson Nano, Xavier, Orin, etc.)
2. **JetPack installed** (recommended: JetPack 5.x or later)
3. **Docker installed** on the Jetson device
4. **Docker Compose** (optional, for easier deployment)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up -d --build
   ```

2. **Run a single image enhancement:**
   ```bash
   docker-compose run --rm lime-enhancement python3 /app/python/main.py
   ```

3. **Run batch processing:**
   ```bash
   docker-compose run --rm lime-enhancement python3 /app/python/batch_process.py
   ```

4. **View logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Directly

1. **Build the Docker image:**
   ```bash
   docker build -t lime-enhancement:jetson .
   ```

2. **Run a single image enhancement:**
   ```bash
   docker run --rm \
     -v $(pwd)/images:/app/images:ro \
     -v $(pwd)/output:/app/output \
     lime-enhancement:jetson \
     python3 /app/python/main.py
   ```

3. **Run batch processing:**
   ```bash
   docker run --rm \
     -v $(pwd)/images:/app/images:ro \
     -v $(pwd)/output:/app/output \
     lime-enhancement:jetson \
     python3 /app/python/batch_process.py
   ```

4. **Run with custom command:**
   ```bash
   docker run --rm \
     -v $(pwd)/images:/app/images:ro \
     -v $(pwd)/output:/app/output \
     lime-enhancement:jetson \
     python3 -c "from lime import LIME; import numpy as np; from PIL import Image; ..."
   ```

## Configuration

### Modifying Parameters

You can modify the configuration in several ways:

1. **Edit config.py before building:**
   ```bash
   # Edit python/config.py
   vim python/config.py
   # Then rebuild
   docker-compose build
   ```

2. **Mount custom config file:**
   ```bash
   docker run --rm \
     -v $(pwd)/images:/app/images:ro \
     -v $(pwd)/output:/app/output \
     -v $(pwd)/python/config.py:/app/python/config.py:ro \
     lime-enhancement:jetson \
     python3 /app/python/main.py
   ```

3. **Use environment variables** (requires code modification to read from env)

### Input/Output Directories

- **Input images:** Place images in the `./images/` directory (mounted as `/app/images` in container)
- **Output results:** Enhanced images will be saved to `./output/` directory (mounted as `/app/output` in container)

## Base Image Compatibility

The Dockerfile uses `nvcr.io/nvidia/l4t-base:r35.2.1` which corresponds to JetPack 5.1.2.

If you're using a different JetPack version, you may need to adjust the base image tag:

- **JetPack 5.1.2:** `nvcr.io/nvidia/l4t-base:r35.2.1` (default)
- **JetPack 5.1.1:** `nvcr.io/nvidia/l4t-base:r35.1.0`
- **JetPack 5.0.2:** `nvcr.io/nvidia/l4t-base:r35.0.0`
- **JetPack 4.6.x:** `nvcr.io/nvidia/l4t-base:r32.7.1`

To check your JetPack version:
```bash
cat /etc/nv_tegra_release
```

## Performance Optimization

### GPU Acceleration (if applicable)

If you want to use GPU acceleration (requires CUDA-enabled dependencies), you can:

1. **Add GPU support to docker-compose.yml:**
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

2. **Use nvidia-docker runtime:**
   ```bash
   docker run --rm --runtime=nvidia \
     -v $(pwd)/images:/app/images:ro \
     -v $(pwd)/output:/app/output \
     lime-enhancement:jetson \
     python3 /app/python/main.py
   ```

Note: The current implementation uses NumPy which runs on CPU. For GPU acceleration, you would need to modify the code to use CuPy or similar GPU-accelerated libraries.

### Memory Optimization

For Jetson devices with limited RAM:

1. **Limit container memory:**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G
   ```

2. **Process images one at a time** using batch_process.py instead of loading all at once

## Troubleshooting

### Container fails to start

- Check Docker is running: `sudo systemctl status docker`
- Verify base image compatibility with your JetPack version
- Check disk space: `df -h`

### Permission errors

- Ensure output directory is writable: `chmod 777 output/`
- Run with appropriate user permissions or use `--user` flag

### Import errors

- Verify Python path: `docker run --rm lime-enhancement:jetson python3 -c "import sys; print(sys.path)"`
- Check dependencies: `docker run --rm lime-enhancement:jetson pip3 list`

### Performance issues

- Monitor resource usage: `docker stats`
- Consider processing smaller batches
- Check if swap is enabled for memory-intensive operations

## Advanced Usage

### Interactive Shell

Access the container shell for debugging:
```bash
docker-compose run --rm lime-enhancement /bin/bash
# or
docker run -it --rm \
  -v $(pwd)/images:/app/images:ro \
  -v $(pwd)/output:/app/output \
  lime-enhancement:jetson \
  /bin/bash
```

### Custom Entrypoint

Create a custom entrypoint script for more complex workflows:
```bash
docker run --rm \
  -v $(pwd)/images:/app/images:ro \
  -v $(pwd)/output:/app/output \
  --entrypoint /bin/bash \
  lime-enhancement:jetson \
  -c "cd /app/python && python3 batch_process.py"
```

## Building for Different Architectures

This Dockerfile is specifically for ARM64 (Jetson). To build for other architectures, you would need to:

1. Use a different base image (e.g., `python:3.9-slim` for x86_64)
2. Adjust the build context and paths accordingly

## References

- [NVIDIA Container Registry](https://catalog.ngc.nvidia.com/containers)
- [Jetson Docker Containers](https://github.com/dusty-nv/jetson-containers)
- [Docker Documentation](https://docs.docker.com/)

