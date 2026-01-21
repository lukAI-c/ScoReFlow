#!/bin/bash
# ============================================================================
# Step 3: Generate plots for all Gym tasks (Linux/Mac shell version)
# Usage: bash 3_plot_gym.sh
# ============================================================================

set -e

# Gym tasks to plot
tasks=(walker-d4rl)

# Project root: auto-detect (assume script is in agent/eval/visualize/gym/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../../../")"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"

CSV_FILE="all_methods_merged.csv"

for TASK in "${tasks[@]}"; do
    DATA_PATH="visualize/Final_experiments/data/finetune/gym-state/$TASK/$CSV_FILE"
    echo "==============================================="
    echo "Task: $TASK"
    echo "CSV: $DATA_PATH"
    if [ ! -f "$DATA_PATH" ]; then
        echo "✗ Error: Merged data file not found: $DATA_PATH"
        echo "Please run the merge script for $TASK first."
        continue
    fi
    echo "Generating plot for $TASK..."
    python agent/eval/visualize/success_rate_episode_reward.py \
        evaluation_name=AverageEpisodeReward \
        environment_name=gym-state \
        task_name=$TASK \
        env.gym-state.$TASK.csv_filename=$CSV_FILE
    if [ $? -eq 0 ]; then
        echo "✓ Plots generated for $TASK!"
        echo "  - visualize/Final_experiments/outs/gym-state_${TASK}_AverageEpisodeReward.pdf"
        echo "  - visualize/Final_experiments/outs/gym-state_${TASK}_AverageEpisodeReward.png"
        echo "  - visualize/Final_experiments/outs/gym-state_${TASK}_AverageEpisodeReward_legend.pdf"
    else
        echo "✗ Error: Plotting failed for $TASK"
    fi
    echo ""
done
