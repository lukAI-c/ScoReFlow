#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT="${RLINF_ROOT:-/path/to/RLinf}"
SCOREFLOW_ROOT="${SCOREFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${RLINF_ROOT}/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-}"
RL_CHECKPOINT_DIR="${RL_CHECKPOINT_DIR:-}"
CHECKPOINT_PROVENANCE="${CHECKPOINT_PROVENANCE:-}"
TRAINING_ARTIFACT="${TRAINING_ARTIFACT:-}"
SUITE="${SUITE:-libero_spatial}"
METHOD="${METHOD:-flow_noise_baseline}"
EXP_ROOT="${EXP_ROOT:-${RLINF_ROOT}/logs/pirl_official_evaluation}"
PATCH_RLINF="${PATCH_RLINF:-1}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  echo "MODEL_DIR must be an existing directory" >&2
  exit 1
fi
if [[ -z "${RL_CHECKPOINT_DIR}" ]]; then
  echo "RL_CHECKPOINT_DIR is required" >&2
  exit 1
fi
if [[ ! "${CHECKPOINT_PROVENANCE}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "CHECKPOINT_PROVENANCE must be sha256:<64 lowercase hex characters>" >&2
  exit 1
fi
if [[ -z "${TRAINING_ARTIFACT}" || ! -f "${TRAINING_ARTIFACT}" ]]; then
  echo "TRAINING_ARTIFACT must reference an existing compliant training artifact" >&2
  exit 1
fi
checkpoint_weights="${RL_CHECKPOINT_DIR}/actor/model_state_dict/full_weights.pt"
if [[ ! -f "${checkpoint_weights}" ]]; then
  echo "RL checkpoint weights not found: ${checkpoint_weights}" >&2
  exit 1
fi
actual_checkpoint_provenance="$("${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_protocol.py" digest-file --path "${checkpoint_weights}")"
if [[ "${actual_checkpoint_provenance}" != "${CHECKPOINT_PROVENANCE}" ]]; then
  echo "CHECKPOINT_PROVENANCE does not match RL checkpoint weights" >&2
  exit 1
fi
"${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_protocol.py" validate \
  --kind training \
  --artifact "${TRAINING_ARTIFACT}"

config_for_suite() {
  case "$1" in
    libero_spatial) echo "libero_spatial_ppo_openpi_pi05" ;;
    libero_object) echo "libero_object_ppo_openpi_pi05" ;;
    libero_goal) echo "libero_goal_ppo_openpi_pi05" ;;
    libero_10) echo "libero_10_ppo_openpi_pi05" ;;
    *) echo "Unsupported suite: $1" >&2; return 1 ;;
  esac
}

suite_value() {
  case "$1:$2" in
    libero_spatial:interaction_steps) echo "240" ;;
    libero_object:interaction_steps | libero_goal:interaction_steps) echo "320" ;;
    libero_10:interaction_steps) echo "480" ;;
    libero_spatial:action_replan_horizon | libero_object:action_replan_horizon | libero_goal:action_replan_horizon) echo "5" ;;
    libero_10:action_replan_horizon) echo "10" ;;
    libero_spatial:denoise_steps) echo "3" ;;
    libero_object:denoise_steps | libero_goal:denoise_steps | libero_10:denoise_steps) echo "5" ;;
    *) echo "Unsupported suite field: $1:$2" >&2; return 1 ;;
  esac
}

export REPO_PATH="${REPO_PATH:-${RLINF_ROOT}}"
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

if [[ "${PATCH_RLINF}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/patch_rlinf_pirl_evaluator.py" \
    --rlinf-root "${RLINF_ROOT}"
  "${PYTHON_BIN}" -m py_compile \
    "${RLINF_ROOT}/rlinf/envs/libero/libero_env.py" \
    "${RLINF_ROOT}/rlinf/runners/embodied_eval_runner.py" \
    "${RLINF_ROOT}/rlinf/workers/rollout/hf/huggingface_worker.py"
fi

config_name="$(config_for_suite "${SUITE}")"
interaction_steps="$(suite_value "${SUITE}" interaction_steps)"
action_replan_horizon="$(suite_value "${SUITE}" action_replan_horizon)"
denoise_steps="$(suite_value "${SUITE}" denoise_steps)"
run_name="${SUITE}_${METHOD}_official_eval"
run_dir="${EXP_ROOT}/${run_name}"
raw_episodes="${run_dir}/episodes.jsonl"
command_file="${run_dir}/command.txt"
artifact="${run_dir}/evaluation_artifact.json"
checkpoint_receipt="${run_dir}/checkpoint_load_receipt.json"
run_log="${run_dir}/run.log"
terminal_status="${run_dir}/terminal_status.json"
reproducibility_bundle="${run_dir}/reproducibility.json"
mkdir -p "${run_dir}"

cmd=(
  "${PYTHON_BIN}" "${RLINF_ROOT}/examples/embodiment/eval_embodied_agent.py"
  --config-path "${RLINF_ROOT}/examples/embodiment/config/"
  --config-name "${config_name}"
  "runner.logger.log_path=${run_dir}"
  "runner.logger.experiment_name=${run_name}"
  "++runner.official_episode_artifact=${raw_episodes}"
  "++runner.official_checkpoint_receipt=${checkpoint_receipt}"
  "++runner.ckpt_path=${checkpoint_weights}"
  "algorithm.eval_rollout_epoch=1"
  "env.eval.total_num_envs=500"
  "env.eval.auto_reset=true"
  "env.eval.ignore_terminations=true"
  "env.eval.group_size=1"
  "env.eval.use_fixed_reset_state_ids=true"
  "env.eval.use_ordered_reset_state_ids=true"
  "env.eval.max_episode_steps=${interaction_steps}"
  "env.eval.max_steps_per_rollout_epoch=${interaction_steps}"
  "env.eval.video_cfg.save_video=false"
  "actor.model.model_path=${MODEL_DIR}"
  "rollout.model.model_path=${MODEL_DIR}"
  "actor.model.num_action_chunks=${action_replan_horizon}"
  "actor.model.num_steps=${denoise_steps}"
  "actor.model.openpi.noise_method=flow_noise"
  "++actor.model.openpi.joint_logprob=true"
)

printf "%q " "${cmd[@]}" > "${command_file}"
echo >> "${command_file}"
if [[ "${PREPARE_ONLY}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_repro_bundle.py" \
    --output "${reproducibility_bundle}" \
    --command-file "${command_file}" \
    --protocol-artifact "${TRAINING_ARTIFACT}" \
    --local-root "${SCOREFLOW_ROOT}" \
    --rlinf-root "${RLINF_ROOT}" \
    --status "prepared" \
    --exit-code 0
  echo "Prepared ${command_file}"
  exit 0
fi

set +e
"${cmd[@]}" 2>&1 | tee "${run_log}"
exit_code="${PIPESTATUS[0]}"
set -e
if [[ "${exit_code}" != "0" ]]; then
  printf '{"status":"failed","exit_code":%s}\n' "${exit_code}" > "${terminal_status}"
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_repro_bundle.py" \
    --output "${reproducibility_bundle}" \
    --command-file "${command_file}" \
    --protocol-artifact "${TRAINING_ARTIFACT}" \
    --local-root "${SCOREFLOW_ROOT}" \
    --rlinf-root "${RLINF_ROOT}" \
    --status "failed" \
    --exit-code "${exit_code}"
  echo "Official evaluation failed with exit code ${exit_code}" >&2
  exit "${exit_code}"
fi
printf '{"status":"done","exit_code":0}\n' > "${terminal_status}"
"${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_repro_bundle.py" \
  --output "${reproducibility_bundle}" \
  --command-file "${command_file}" \
  --protocol-artifact "${TRAINING_ARTIFACT}" \
  --local-root "${SCOREFLOW_ROOT}" \
  --rlinf-root "${RLINF_ROOT}" \
  --status "done" \
  --exit-code 0

"${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_official_eval.py" \
  --raw-episodes "${raw_episodes}" \
  --command-file "${command_file}" \
  --checkpoint-receipt "${checkpoint_receipt}" \
  --training-artifact "${TRAINING_ARTIFACT}" \
  --terminal-status "${terminal_status}" \
  --reproducibility-bundle "${reproducibility_bundle}" \
  --output "${artifact}" \
  --suite "${SUITE}" \
  --method "${METHOD}" \
  --checkpoint-provenance "${CHECKPOINT_PROVENANCE}"
"${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_protocol.py" validate \
  --kind evaluation \
  --artifact "${artifact}"
echo "Official evaluation artifact: ${artifact}"
