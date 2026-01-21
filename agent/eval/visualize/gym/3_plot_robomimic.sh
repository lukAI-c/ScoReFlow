#!/bin/bash
# ============================================================================
# Step 3: Generate plots for all Robomimic tasks (Linux/Mac shell version)
# Usage: bash 3_plot_robomimic.sh
# ============================================================================
# This script generates plots for Figure 3: Success Rates in Robomimic Visual Manipulation Tasks
# ============================================================================

set -e

# Robomimic tasks to plot
tasks=(can-img square-img transport-img)

# Project root: auto-detect (assume script is in agent/eval/visualize/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../../")"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"
echo ""

# Set environment variables
export REINFLOW_DIR="$PROJECT_ROOT"
export REINFLOW_DATA_DIR="$PROJECT_ROOT"
export REINFLOW_LOG_DIR="$PROJECT_ROOT/log"

CSV_FILE="all_methods_merged.csv"
EVALUATION_NAME="SuccessRate"
ENVIRONMENT_NAME="robomimic-img"

echo "================================================================================"
echo "Figure 3: Success Rates in Robomimic Visual Manipulation Tasks"
echo "================================================================================"
echo ""

for TASK in "${tasks[@]}"; do
    DATA_PATH="visualize/Final_experiments/data/finetune/$ENVIRONMENT_NAME/$TASK/$CSV_FILE"
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
        environment_name=$ENVIRONMENT_NAME \
        task_name=$TASK \
        env.$ENVIRONMENT_NAME.$TASK.csv_filename=$CSV_FILE
    
    if [ $? -eq 0 ]; then
        echo "✓ Plots generated for $TASK!"
        echo "  - visualize/Final_experiments/outs/${ENVIRONMENT_NAME}_${TASK}_${EVALUATION_NAME}.pdf"
        echo "  - visualize/Final_experiments/outs/${ENVIRONMENT_NAME}_${TASK}_${EVALUATION_NAME}.png"
        echo "  - visualize/Final_experiments/outs/${ENVIRONMENT_NAME}_${TASK}_${EVALUATION_NAME}_legend.pdf"
    else
        echo "✗ Error: Plotting failed for $TASK"
    fi
    echo ""
done

echo "================================================================================"
echo "All Robomimic plots generated!"
echo "================================================================================"
echo ""
echo "To crop the legend (optional):"
echo "python agent/eval/visualize/crop_pdfs.py \\"
echo "  --input_pdf=visualize/Final_experiments/outs/robomimic-img_can-img_SuccessRate_legend.pdf \\"
echo "  --output_pdf=visualize/Final_experiments/outs/robomimic-img_can-img_SuccessRate_legend_crop.pdf \\"
echo "  --left_percent=20 --right_percent=20 --top_percent=10 --bottom_percent=10"
echo ""

