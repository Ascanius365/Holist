#!/bin/bash

LOG_DIR=".vllm_logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to format time duration
format_duration() {
  local seconds=$1
  local days=$((seconds/86400))
  local hours=$(((seconds%86400)/3600))
  local minutes=$(((seconds%3600)/60))
  local secs=$((seconds%60))
  
  if [ $days -gt 0 ]; then
    echo "${days}d ${hours}h ${minutes}m ${secs}s"
  elif [ $hours -gt 0 ]; then
    echo "${hours}h ${minutes}m ${secs}s"
  elif [ $minutes -gt 0 ]; then
    echo "${minutes}m ${secs}s"
  else
    echo "${secs}s"
  fi
}

# Function to get GPU usage for a process
get_gpu_info() {
  local pid=$1
  if command -v nvidia-smi &> /dev/null; then
    # Get GPU ID and memory usage for this PID
    local gpu_info=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep " $pid " | tr -d ' ')
    if [ -n "$gpu_info" ]; then
      local gpu_ids=$(nvidia-smi --query-compute-apps=gpu_index,pid --format=csv,noheader | grep " $pid " | cut -d, -f1 | tr '\n' ',' | sed 's/,$//')
      local mem_usage=$(echo "$gpu_info" | cut -d, -f3)
      echo "GPUs: $gpu_ids, Mem: $mem_usage"
    else
      echo "No GPU usage found"
    fi
  else
    echo "nvidia-smi not available"
  fi
}

# Function to get memory usage for a process
get_memory_usage() {
  local pid=$1
  if [ -f /proc/$pid/status ]; then
    local mem=$(grep VmRSS /proc/$pid/status | awk '{print $2}')
    if [ -n "$mem" ]; then
      if [ $mem -gt 1048576 ]; then
        echo "$(echo "scale=2; $mem/1048576" | bc) GB"
      else
        echo "$(echo "scale=2; $mem/1024" | bc) MB"
      fi
    else
      echo "Unknown"
    fi
  else
    echo "Unknown"
  fi
}

# Function to clean up stale command and log files
clean_stale_files() {
  local verbose=$1
  
  if [ "$verbose" = "true" ]; then
    echo "Cleaning up stale vLLM files..."
    echo "==============================="
  fi
  
  # Find all port-specific command files
  cmd_files=$(find "$LOG_DIR" -name "vllm_server_port*.cmd" 2>/dev/null)
  
  if [ -n "$cmd_files" ]; then
    for cmd_file in $cmd_files; do
      # Extract PID from command file (last number in the file)
      pid=$(cat "$cmd_file" | grep -oE '[0-9]+$')
      
      # Extract port from filename
      port=$(echo "$cmd_file" | grep -oE 'port[0-9]+' | grep -oE '[0-9]+')
      
      # Check if process is running
      if [ -z "$pid" ] || ! ps -p "$pid" > /dev/null 2>&1; then
        if [ "$verbose" = "true" ]; then
          echo "Removing stale command file for port $port: $cmd_file"
        fi
        rm "$cmd_file"
        
        # Also remove corresponding log file if it exists
        log_file="$LOG_DIR/vllm_server_port${port}.log"
        if [ -f "$log_file" ]; then
          if [ "$verbose" = "true" ]; then
            echo "Removing stale log file: $log_file"
          fi
          rm "$log_file"
        fi
      fi
    done
  fi
  
  # Find other log files that might be orphaned
  log_files=$(find "$LOG_DIR" -name "vllm_*.log" -not -name "vllm_server_port*.log" 2>/dev/null)
  
  if [ -n "$log_files" ]; then
    for log_file in $log_files; do
      # Try to extract PID from log file content
      pid=$(grep -o "PID: [0-9]\+" "$log_file" 2>/dev/null | head -1 | awk '{print $2}')
      
      # If PID found and not running, or no PID found, remove the log file
      if [ -z "$pid" ] || ! ps -p "$pid" > /dev/null 2>&1; then
        if [ "$verbose" = "true" ]; then
          echo "Removing orphaned log file: $log_file"
        fi
        rm "$log_file"
      fi
    done
  fi
  
  if [ "$verbose" = "true" ]; then
    echo "Cleanup completed."
  fi
  return 0
}

# Function to list all running vLLM processes with detailed info
list_vllm_processes() {
  echo "Running vLLM processes:"
  echo "======================="
  
  # Find all processes with "vllm" in the command
  pids=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}')
  
  if [ -z "$pids" ]; then
    echo "No vLLM processes found."
    return 1
  fi
  
  now=$(date +%s)
  
  i=1
  for pid in $pids; do
    echo "[$i] PID: $pid"
    
    # Get start time
    if [ -e /proc/$pid/stat ]; then
      start_time=$(stat -c %Y /proc/$pid/stat)
      uptime=$((now - start_time))
      echo "    Uptime: $(format_duration $uptime)"
    else
      echo "    Uptime: Unknown"
    fi
    
    # Get port from command
    port=$(ps -p $pid -o cmd= | grep -oE 'port [0-9]+' | awk '{print $2}')
    if [ -n "$port" ]; then
      echo "    Port: $port"
    fi
    
    # Get command
    cmd_file=$(find "$LOG_DIR" -name "vllm_server_port*.cmd" -exec grep -l "$pid" {} \; | head -n 1)
    if [ -n "$cmd_file" ]; then
      cmd=$(cat "$cmd_file" | sed "s/ $pid$//")
      echo "    Command: $cmd"
    else
      cmd=$(ps -p $pid -o cmd= | sed 's/^/    /')
      echo "    Command: $(echo "$cmd" | head -n 1)"
    fi
    
    # Get log file
    if [ -n "$port" ]; then
      log_file="$LOG_DIR/vllm_server_port${port}.log"
      if [ -f "$log_file" ]; then
        echo "    Log: $log_file"
      fi
    else
      log_file=$(find "$LOG_DIR" -type f -name "vllm_*.log" -exec grep -l "$pid" {} \; | head -n 1)
      if [ -n "$log_file" ]; then
        echo "    Log: $log_file"
      fi
    fi
    
    # Get memory usage
    mem_usage=$(get_memory_usage $pid)
    echo "    Memory: $mem_usage"
    
    # Get GPU info
    gpu_info=$(get_gpu_info $pid)
    echo "    $gpu_info"
    
    echo ""
    i=$((i+1))
  done
  
  return 0
}

# Function to show detailed info for a specific process
show_process_details() {
  local pid=$1
  
  if ! ps -p "$pid" > /dev/null; then
    echo "Process with PID $pid is not running."
    return 1
  fi
  
  echo "Details for vLLM process $pid:"
  echo "=============================="
  
  # Get port from command
  port=$(ps -p $pid -o cmd= | grep -oE 'port [0-9]+' | awk '{print $2}')
  if [ -n "$port" ]; then
    echo "Port: $port"
  fi
  
  # Get command
  cmd_file=$(find "$LOG_DIR" -name "vllm_server_port*.cmd" -exec grep -l "$pid" {} \; | head -n 1)
  if [ -n "$cmd_file" ]; then
    cmd=$(cat "$cmd_file" | sed "s/ $pid$//")
    echo "Command:"
    echo "$cmd"
  else
    cmd=$(ps -p $pid -o cmd=)
    echo "Command: $cmd"
  fi
  
  # Get log file
  if [ -n "$port" ]; then
    log_file="$LOG_DIR/vllm_server_port${port}.log"
  else
    log_file=$(find "$LOG_DIR" -type f -name "vllm_*.log" -exec grep -l "$pid" {} \; | head -n 1)
  fi
  
  if [ -n "$log_file" ] && [ -f "$log_file" ]; then
    echo -e "\nLog file: $log_file"
    echo -e "\nLast 10 lines of log:"
    echo "---------------------"
    tail -n 10 "$log_file"
  fi
  
  # Get GPU info
  if command -v nvidia-smi &> /dev/null; then
    echo -e "\nGPU Usage:"
    echo "---------"
    nvidia-smi --query-compute-apps=gpu_index,pid,process_name,used_memory --format=table | grep -E "($pid|GPU)"
  fi
  
  return 0
}

# Main script logic
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  echo "Usage: $0 [OPTIONS] [PID]"
  echo ""
  echo "Options:"
  echo "  -h, --help     Show this help message"
  echo "  -d, --detail   Show detailed information for a specific PID"
  echo "  -c, --clean    Clean up stale command and log files (verbose mode)"
  echo ""
  echo "Examples:"
  echo "  $0             List all running vLLM processes"
  echo "  $0 1234        Show detailed information for PID 1234"
  echo "  $0 -d 1234     Same as above"
  echo "  $0 -c          Clean up stale command and log files (verbose mode)"
  exit 0
fi

# Always clean up stale files silently
clean_stale_files "false"

# Check if clean option was provided (for verbose mode)
if [ "$1" = "-c" ] || [ "$1" = "--clean" ]; then
  clean_stale_files "true"
  exit $?
fi

# Check if a specific PID was provided with -d flag
if [ "$1" = "-d" ] || [ "$1" = "--detail" ]; then
  if [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    show_process_details "$2"
    exit $?
  else
    echo "Error: PID must be provided with -d option"
    exit 1
  fi
fi

# Check if a specific PID was provided
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  show_process_details "$1"
  exit $?
fi

# Default: list all processes
list_vllm_processes
exit $?