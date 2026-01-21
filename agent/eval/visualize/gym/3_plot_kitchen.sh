#!/bin/bash
# ============================================================================
# Step 3: Generate plots for all Kitchen tasks (Linux/Mac shell version)
# Usage: bash 3_plot_kitchen.sh
# ============================================================================
# This script generates plots for Figure 2: Task Completion Rates in Franka Kitchen
# ============================================================================

set -e

# Kitchen tasks to plot
tasks=(kitchen-complete-v0)

# Project root: auto-detect (assume script is in agent/eval/visualize/kitchen/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../../../")"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Set environment variables
export REINFLOW_DIR="$PROJECT_ROOT"
export REINFLOW_DATA_DIR="$PROJECT_ROOT"
export REINFLOW_LOG_DIR="$PROJECT_ROOT/log"

CSV_FILE="all_methods_merged.csv"
EVALUATION_NAME="TaskCompletionRate"

echo "================================================================================"
echo "Figure 2: Task Completion Rates in Franka Kitchen"
echo "================================================================================"
echo ""

for TASK in "${tasks[@]}"; do
    DATA_PATH="visualize/Final_experiments/data/finetune/kitchen/$TASK/$CSV_FILE"
    echo "==============================================="
    echo "Task: $TASK"
    echo "CSV: $DATA_PATH"
    
    if [ ! -f "$DATA_PATH" ]; then
        echo "✗ Error: Merged data file not found: $DATA_PATH"
        echo "Please run the merge script for $TASK first."
        echo ""
        continue
    fi
    
    echo "Generating plot for $TASK..."
    python agent/eval/visualize/success_rate_episode_reward.py \
        evaluation_name=$EVALUATION_NAME \
        environment_name=kitchen \
        task_name=$TASK \
        env.kitchen.$TASK.csv_filename=$CSV_FILE
    
    if [ $? -eq 0 ]; then
        echo "✓ Plots generated for $TASK!"
        echo "  - visualize/Final_experiments/outs/kitchen_${TASK}_${EVALUATION_NAME}.pdf"
        echo "  - visualize/Final_experiments/outs/kitchen_${TASK}_${EVALUATION_NAME}.png"
        echo "  - visualize/Final_experiments/outs/kitchen_${TASK}_${EVALUATION_NAME}_legend.pdf"
    else
        echo "✗ Error: Plotting failed for $TASK"
    fi
    echo ""
done

echo "================================================================================"
echo "All Kitchen plots generated!"
echo "================================================================================"
echo ""
echo "To crop the legend (optional):"
echo "python agent/eval/visualize/crop_pdfs.py \\"
echo "  --input_pdf=visualize/Final_experiments/outs/kitchen_kitchen-complete-v0_TaskCompletionRate_legend.pdf \\"
echo "  --output_pdf=visualize/Final_experiments/outs/kitchen_kitchen-complete-v0_TaskCompletionRate_legend_crop.pdf \\"
echo "  --left_percent=30 --right_percent=30 --top_percent=10 --bottom_percent=10"
echo ""

