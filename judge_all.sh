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
############## Configurations ##############


for agent_name in ${AGENT_NAMES}; do
    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    mkdir -p log
    logfile="log/${timestamp}.log"

    # Base command (no debug flag / redirection)
    cmd="python judge_all.py --agent_name ${agent_name} --max_workers 16"
    
    # If MIN_TIMESTAMP is set, append it to the command
    if [ -n "$MIN_TIMESTAMP" ]; then
        cmd="${cmd} --min_timestamp ${MIN_TIMESTAMP}"
    fi

    if [ "$DEBUG" = "1" ]; then
        # Debug mode: add --debug, no redirection (useful for ipdb interaction)
        cmd="${cmd} --debug"
        echo "Running in DEBUG mode:"
        echo "  ${cmd}"
        eval "${cmd}"
    else
        # Normal mode: no --debug, redirect output to the log file
        echo "Running in NORMAL mode, logging to ${logfile}:"
        echo "  ${cmd} > ${logfile} 2>&1"
        eval "${cmd} > \"${logfile}\" 2>&1"
    fi
done
