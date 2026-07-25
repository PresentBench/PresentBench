#!/usr/bin/env bash

############## Configurations ##############
AGENT_NAMES=(
    "NotebookLM"
)

# When DEBUG=1: run in foreground with debugging enabled; when DEBUG=0: normal mode, log to file
DEBUG=0

# MIN_TIMESTAMP (optional): set the minimum timestamp (format: YYYY-MM-DD_HH-MM-SS)
# Example: MIN_TIMESTAMP="2024-01-01_12-00-00"
MIN_TIMESTAMP="2026-03-07_16-00-00"

# Judge model settings:
#   api_type: gemini | gemini_inline
#   model: model name (e.g. gemini-3-flash-preview or Qwen/Qwen3.5-122B-A10B)
API_TYPE="gemini"
MODEL="gemini-3-flash-preview"
RETRY=5
# TEMPERATURE=


############## Configurations ##############


for agent_name in ${AGENT_NAMES}; do
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    mkdir -p log
    logfile="log/${timestamp}.log"

    # If MAX_WORKERS is not set, default to 16
    MAX_WORKERS="${MAX_WORKERS:-16}"

    # Base command (no debug flag / redirection)
    cmd="python judge_all.py --agent_name ${agent_name} --max_workers ${MAX_WORKERS}"
    cmd="${cmd} --api_type ${API_TYPE} --model ${MODEL}"
    cmd="${cmd} --retry ${RETRY}"
    
    # If MIN_TIMESTAMP is set, append it to the command
    if [ -n "$MIN_TIMESTAMP" ]; then
        cmd="${cmd} --min_timestamp ${MIN_TIMESTAMP}"
    fi

    # If LIMIT is set, only run the first N test cases
    if [ -n "$LIMIT" ]; then
        cmd="${cmd} --limit ${LIMIT}"
    fi

    # If TEMPERATURE is set, fix the judge sampling temperature
    # (otherwise the API server default decoding is used).
    if [ -n "$TEMPERATURE" ]; then
        cmd="${cmd} --temperature ${TEMPERATURE}"
    fi

    # If SEED is set, fix the judge random seed for reproducible verdicts
    # (forwarded to backends that support it; for --repeats>1 repeat k uses seed+(k-1)).
    if [ -n "$SEED" ]; then
        cmd="${cmd} --seed ${SEED}"
    fi

    if [ "$DEBUG" = "1" ]; then
        # Debug mode: add --debug, no redirection (useful for ipdb interaction)
        cmd="${cmd} --debug"
        echo "Running in DEBUG mode:"
        echo "  ${cmd}"
        eval "${cmd}"
    else
        # Normal mode: no --debug, output to both screen and log file
        echo "Running in NORMAL mode, logging to ${logfile}:"
        echo "  ${cmd} 2>&1 | tee ${logfile}"
        eval "${cmd} 2>&1 | tee \"${logfile}\""
    fi
done
