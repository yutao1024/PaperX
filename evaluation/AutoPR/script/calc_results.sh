#!/bin/bash

# --- Configuration ---
# Path to the output file from the evaluation script.
# METRIC_OUTPUT_PATH="metric_results.jsonl"
METRIC_OUTPUT_PATH="PaperX/evaluation/AutoPR/eval/output_dir/refined/llm_metric_result.jsonl"

# Optional: Path to a .txt file with specific IDs to include in the calculation.
# Leave empty to calculate metrics for all available IDs.
SELECTED_IDS_PATH="PaperX/evaluation/AutoPR/eval/output_dir/refined/selected_id.txt"

# --- Execution ---
echo "--- Calculating Final Metrics ---"
CMD_ARGS=("--metric_output_path" "$METRIC_OUTPUT_PATH")

if [ -n "$SELECTED_IDS_PATH" ]; then
    CMD_ARGS+=("--selected_ids_path" "$SELECTED_IDS_PATH")
fi

python eval/calc_metric.py "${CMD_ARGS[@]}"
echo "--- Calculation Complete ---"