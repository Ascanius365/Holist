#!/bin/bash

# Help function
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -m, --model MODEL          Model name or path (can use shortcuts like llama3.3-70b)"
  echo "  -t, --tensor-parallel N    Tensor parallel size (default: 4)"
  echo "  -g, --gpu-util FLOAT       GPU memory utilization (default: 0.9)"
  echo "  -d, --devices DEVICES      Comma-separated list of GPU device IDs (default: all available)"
  echo "  -h, --host HOST            Host to serve on (default: 0.0.0.0)"
  echo "  -p, --port PORT            Port to serve on (default: 8000)"
  echo "  --dtype TYPE               Data type (default: bfloat16)"
  echo "  -l, --max-len N            Maximum sequence length (default: 1024)"
  echo "  -s, --swap FLOAT           Swap space in GB (default: 0)"
  echo "  --help                     Show this help message and exit"
  echo ""
  echo "Examples:"
  echo "  $0 llama3.3-70b                     # Use default settings"
  echo "  $0 -m qwen2.5-72b -d 0,1,2,3        # Run on specific GPUs 0-3"
  echo "  $0 -m llama3.3-70b -d 4,5,6,7 -t 4  # Run on GPUs 4-7 with TP=4"
  echo ""
  exit 0
}

# Resolve model shortcut to full path
function resolve_model_name {
  local shortcut=$(echo "$1" | tr '[:upper:]' '[:lower:]')  # Convert to lowercase
  
  case "$shortcut" in
    "llama3.1-8b")
      echo "meta-llama/Llama-3.1-8B-Instruct"
      ;;
    "llama3.1-70b")
      echo "meta-llama/Llama-3.1-70B-Instruct"
      ;;
    "llama3.3-8b")
      echo "meta-llama/Llama-3.3-8B-Instruct"
      ;;
    "llama3.3-70b")
      echo "meta-llama/Llama-3.3-70B-Instruct"
      ;;
    # Qwen models
    "qwen2.5-1.5b")
      echo "Qwen/Qwen2.5-1.5B-Instruct"
      ;;
    "qwen2.5-3b")
      echo "Qwen/Qwen2.5-3B-Instruct"
      ;;
    "qwen2.5-7b")
      echo "Qwen/Qwen2.5-7B-Instruct"
      ;;
    "qwen2.5-14b")
      echo "Qwen/Qwen2.5-14B-Instruct"
      ;;
    "qwen2.5-72b")
      echo "Qwen/Qwen2.5-72B-Instruct"
      ;;
    "qwen3-32b")
      echo "Qwen/Qwen3-32B"
      ;;
    "qwen3-14b")
      echo "Qwen/Qwen3-14B"
      ;;
    "qwen3-4b")
      echo "Qwen/Qwen3-4B"
      ;;
    "qwen3-1.7b")
      echo "Qwen/Qwen3-1.7B"
      ;;
    "gemma3-1b")
      echo "google/gemma-3-1b-it"
      ;;
    "gemma3-4b")
      echo "google/gemma-3-4b-it"
      ;;
    "gemma3-27b")
      echo "google/gemma-3-27b-it"
      ;;
    "gemma3-12b")
      echo "google/gemma-3-12b-it"
      ;;
    # If not a known shortcut, return as is
    *)
      echo "$1"
      ;;
  esac
}

# Default values
MODEL="meta-llama/Llama-3.3-70B-Instruct"
TENSOR_PARALLEL=4
GPU_UTIL=0.9
GPU_DEVICES=""  # 기본값은 빈 문자열 (모든 GPU 사용)
HOST="0.0.0.0"
PORT=8000
DTYPE="bfloat16"
MAX_LEN=1024
SWAP=0
LOG_DIR=".vllm_logs"
# 포트별 CMD 파일 경로 설정
CMD_FILE="$LOG_DIR/vllm_server_port${PORT}.cmd"

# Create log directory with proper permissions
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create log directory $LOG_DIR"
    echo "Using current directory for logs instead."
    LOG_DIR="."
    CMD_FILE="./vllm_server_port${PORT}.cmd"
  fi
fi

# Check if log directory is writable
if [ ! -w "$LOG_DIR" ]; then
  echo "Warning: Log directory $LOG_DIR is not writable"
  echo "Using current directory for logs instead."
  LOG_DIR="."
  CMD_FILE="./vllm_server_port${PORT}.cmd"
fi

LOG_FILE="$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL=$(resolve_model_name "$2")
      shift 2
      ;;
    -t|--tensor-parallel)
      TENSOR_PARALLEL="$2"
      shift 2
      ;;
    -g|--gpu-util)
      GPU_UTIL="$2"
      shift 2
      ;;
    -d|--devices)
      GPU_DEVICES="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      # 포트가 변경되면 CMD 파일 경로도 업데이트
      CMD_FILE="$LOG_DIR/vllm_server_port${PORT}.cmd"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    -l|--max-len)
      MAX_LEN="$2"
      shift 2
      ;;
    -s|--swap)
      SWAP="$2"
      shift 2
      ;;
    --help)
      show_help
      ;;
    *)
      # If first argument without flag, assume it's the model
      if [[ $1 != -* ]] && [[ -z "$CUSTOM_MODEL" ]]; then
        MODEL=$(resolve_model_name "$1")
        CUSTOM_MODEL=1
        shift
      else
        echo "Unknown option: $1"
        show_help
      fi
      ;;
  esac
done

# Print configuration
echo "Starting vLLM API server with the following configuration:"
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "GPU Memory Utilization: $GPU_UTIL"
if [ -n "$GPU_DEVICES" ]; then
  echo "GPU Devices: $GPU_DEVICES"
else
  echo "GPU Devices: all available"
fi
echo "Host: $HOST"
echo "Port: $PORT"
echo "Data Type: $DTYPE"
echo "Max Length: $MAX_LEN"
echo "Swap Space: $SWAP GB"
echo "Log File: $LOG_FILE"
echo "CMD File: $CMD_FILE"
echo "-------------------------------------"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Check if server is already running on the same port
if [ -f "$CMD_FILE" ]; then
  # CMD 파일에서 명령어 추출
  OLD_CMD=$(cat "$CMD_FILE")
  
  # 명령어에서 PID 추출 (마지막 부분에 있다고 가정)
  PID=$(echo "$OLD_CMD" | grep -oE '[0-9]+$')
  
  if [ -n "$PID" ] && ps -p "$PID" > /dev/null; then
    echo "vLLM server is already running on port $PORT with PID $PID"
    echo "To stop it, use: ./stop_llm.sh $PORT"
    exit 1
  else
    echo "Found stale CMD file for port $PORT. Removing it."
    rm -f "$CMD_FILE"
  fi
fi

# Set CUDA_VISIBLE_DEVICES if specified
if [ -n "$GPU_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_DEVICES"
  echo "Using GPUs: $GPU_DEVICES"
else
  echo "Using all available GPUs"
fi

# Construct the command
CMD="python -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size $TENSOR_PARALLEL \
  --gpu-memory-utilization $GPU_UTIL \
  --host $HOST \
  --port $PORT \
  --dtype $DTYPE \
  --max-model-len $MAX_LEN \
  --swap-space $SWAP"

# Start the server
echo "Starting vLLM server with command:"
echo "$CMD"

# Run the command and capture its exit status
nohup $CMD > "$LOG_DIR/vllm_server_port${PORT}.log" 2>&1 &
PID=$!

# Wait a moment to see if the process is still running
sleep 3
if ! ps -p $PID > /dev/null; then
  echo "Error: Server failed to start properly."
  echo "Check the log file for details: $LOG_DIR/vllm_server_port${PORT}.log"
  
  # Clean up the CMD file if it exists
  if [ -f "$CMD_FILE" ]; then
    echo "Removing command file: $CMD_FILE"
    rm -f "$CMD_FILE"
  fi
  
  echo "Server startup failed. Log files have been preserved for debugging."
  exit 1
fi

# Save PID and command for later use
echo "$CMD $PID" > "$CMD_FILE" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Warning: Could not write CMD file. Server is running with PID $PID on port $PORT"
fi

echo "Server started with PID $PID on port $PORT"
echo "To stop the server, run: ./stop_llm.sh $PORT"