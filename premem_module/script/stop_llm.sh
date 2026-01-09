#!/bin/bash

LOG_DIR=".vllm_logs"

# Function to list all running vLLM processes
list_vllm_processes() {
  echo "Running vLLM processes:"
  echo "----------------------"
  
  # Find all processes with "vllm" in the command
  pids=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}')
  
  if [ -z "$pids" ]; then
    echo "No vLLM processes found."
    return 1
  fi
  
  i=1
  for pid in $pids; do
    # Try to find the port from the command file
    port=""
    cmd_file=$(find "$LOG_DIR" -name "vllm_server_port*.cmd" -exec grep -l "$pid" {} \; 2>/dev/null | head -n 1)
    
    if [ -n "$cmd_file" ]; then
      port=$(echo "$cmd_file" | grep -oE 'port[0-9]+' | grep -oE '[0-9]+')
      cmd=$(cat "$cmd_file" | head -n 1)
    else
      cmd=$(ps -p $pid -o cmd= | head -c 80)
      if [ ${#cmd} -eq 80 ]; then
        cmd="$cmd..."
      fi
      # Try to extract port from command
      port=$(echo "$cmd" | grep -oE -- '--port [0-9]+' | awk '{print $2}')
    fi
    
    if [ -n "$port" ]; then
      echo "[$i] PID: $pid - Port: $port - $cmd"
    else
      echo "[$i] PID: $pid - $cmd"
    fi
    i=$((i+1))
  done
  
  return 0
}

# Function to stop a specific process
stop_process() {
  local pid=$1
  
  echo "Stopping vLLM server with PID $pid..."
  
  # Try to find the port from the command file
  cmd_file=$(find "$LOG_DIR" -name "vllm_server_port*.cmd" -exec grep -l "$pid" {} \; 2>/dev/null | head -n 1)
  
  kill "$pid"
  
  # Wait for process to terminate
  for i in {1..30}; do
    if ! ps -p "$pid" > /dev/null; then
      echo "Server stopped successfully."
      # Remove command file if found
      if [ -n "$cmd_file" ] && [ -f "$cmd_file" ]; then
        rm "$cmd_file"
        echo "Removed command file: $cmd_file"
      fi
      return 0
    fi
    echo "Waiting for server to stop... ($i/30)"
    sleep 1
  done
  
  # Force kill if still running
  if ps -p "$pid" > /dev/null; then
    echo "Server did not stop gracefully. Force killing..."
    kill -9 "$pid"
    if ! ps -p "$pid" > /dev/null; then
      echo "Server stopped forcefully."
      # Remove command file if found
      if [ -n "$cmd_file" ] && [ -f "$cmd_file" ]; then
        rm "$cmd_file"
        echo "Removed command file: $cmd_file"
      fi
      return 0
    else
      echo "Failed to stop server. Please check manually."
      return 1
    fi
  fi
}

# Function to stop a server by port
stop_by_port() {
  local port=$1
  local cmd_file="$LOG_DIR/vllm_server_port${port}.cmd"
  
  if [ ! -f "$cmd_file" ]; then
    echo "No command file found for port $port."
    
    # Try to find PID by checking running processes
    pid=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | grep -- "--port $port" | awk '{print $2}')
    
    if [ -z "$pid" ]; then
      echo "No vLLM server found running on port $port."
      return 1
    fi
    
    echo "Found vLLM server running on port $port with PID $pid."
    stop_process "$pid"
    return $?
  fi
  
  # Extract PID from command file
  pid=$(grep -oE '[0-9]+$' "$cmd_file")
  
  if [ -z "$pid" ]; then
    echo "Could not extract PID from command file."
    rm "$cmd_file"
    return 1
  fi
  
  if ! ps -p "$pid" > /dev/null; then
    echo "Process with PID $pid is not running. Removing stale command file."
    rm "$cmd_file"
    return 1
  fi
  
  stop_process "$pid"
  return $?
}

# Main script logic
if [ "$1" = "-l" ] || [ "$1" = "--list" ]; then
  list_vllm_processes
  exit 0
fi

# Check if a specific port was provided
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  # Check if it's a PID or a port
  if ps -p "$1" > /dev/null 2>&1; then
    # It's a valid PID
    stop_process "$1"
    exit $?
  else
    # Assume it's a port
    stop_by_port "$1"
    exit $?
  fi
fi

# No arguments provided, check for running processes
echo "Checking for running vLLM processes..."
if list_vllm_processes; then
  read -p "Enter number to stop or 'q' to quit: " choice
  if [[ "$choice" =~ ^[0-9]+$ ]]; then
    selected_pid=$(ps aux | grep "vllm.entrypoints.openai.api_server" | grep -v grep | awk '{print $2}' | sed -n "${choice}p")
    if [ -n "$selected_pid" ]; then
      stop_process "$selected_pid"
      exit $?
    else
      echo "Invalid selection."
      exit 1
    fi
  else
    echo "Operation cancelled."
    exit 0
  fi
else
  echo "No vLLM servers found running."
  exit 1
fi