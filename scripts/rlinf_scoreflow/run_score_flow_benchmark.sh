#!/usr/bin/env bash
set -euo pipefail

# RLinf/OpenPI runner for Score-Flow, terminal trust-region, and CR-Reflow.
# Set METHODS to choose arms. Supported methods:
# flow_noise_baseline, scoreflow_original, tr_scalar_l2, tr_pullback,
# tr_pullback_matched, cr_reflow_no_anchor, cr_reflow.

RLINF_ROOT="${RLINF_ROOT:-/path/to/RLinf}"
SCOREFLOW_ROOT="${SCOREFLOW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${RLINF_ROOT}/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-}"
EXP_ROOT="${EXP_ROOT:-${RLINF_ROOT}/logs/scoreflow_original}"

POLICY_VARIANT="${POLICY_VARIANT:-pi0}"
SUITES="${SUITES:-libero_spatial}"
SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-flow_noise_baseline scoreflow_original}"

MAX_EPOCHS="${MAX_EPOCHS:-10}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-1}"
TRAIN_ENVS="${TRAIN_ENVS:-1}"
EVAL_ENVS="${EVAL_ENVS:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export REPO_PATH="${REPO_PATH:-${RLINF_ROOT}}"
export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_ROOT}/examples/embodiment}"
export SRC_FILE="${SRC_FILE:-${EMBODIED_PATH}/train_embodied_agent.py}"
export LIBERO_EXTRA_PYTHONPATH="${LIBERO_EXTRA_PYTHONPATH:-}"
export OSMESA_LIBRARY_DIR="${OSMESA_LIBRARY_DIR:-}"
if [[ -n "${LIBERO_EXTRA_PYTHONPATH}" ]]; then
  export PYTHONPATH="${RLINF_ROOT}:${LIBERO_EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
fi
if [[ -n "${OSMESA_LIBRARY_DIR}" ]]; then
  export LD_LIBRARY_PATH="${OSMESA_LIBRARY_DIR}:${LD_LIBRARY_PATH:-}"
fi

LIBERO_MODEL_DIR="${LIBERO_MODEL_DIR:-/path/to/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT}"
LIBERO_LONG_MODEL_DIR="${LIBERO_LONG_MODEL_DIR:-/path/to/RLinf-Pi0-LIBERO-Long-SFT}"
MANISKILL_MODEL_DIR="${MANISKILL_MODEL_DIR:-/path/to/RLinf-Pi0-ManiSkill-SFT}"
METAWORLD_MODEL_DIR="${METAWORLD_MODEL_DIR:-/path/to/RLinf-Pi0-MetaWorld-SFT}"
CALVIN_MODEL_DIR="${CALVIN_MODEL_DIR:-/path/to/RLinf-Pi0-CALVIN-ABC-D-SFT}"

MANIFEST="${EXP_ROOT}/run_manifest.csv"

mkdir -p "${EXP_ROOT}"

config_for_suite() {
  if [[ "${POLICY_VARIANT}" == "pi05" || "${POLICY_VARIANT}" == "pi0.5" ]]; then
    case "$1" in
      libero_spatial) echo "libero_spatial_ppo_openpi_pi05" ;;
      libero_object) echo "libero_object_ppo_openpi_pi05" ;;
      libero_goal) echo "libero_goal_ppo_openpi_pi05" ;;
      libero_10) echo "libero_10_ppo_openpi_pi05" ;;
      maniskill) echo "maniskill_ppo_openpi_pi05" ;;
      metaworld_mt50) echo "metaworld_50_ppo_openpi_pi05" ;;
      calvin_d_d) echo "calvin_d_d_ppo_openpi_pi05" ;;
      *) echo "Unsupported suite: $1" >&2; return 1 ;;
    esac
  else
    case "$1" in
      libero_spatial) echo "libero_spatial_ppo_openpi" ;;
      libero_object) echo "libero_object_ppo_openpi" ;;
      libero_goal) echo "libero_goal_ppo_openpi" ;;
      libero_10) echo "libero_10_ppo_openpi" ;;
      maniskill) echo "maniskill_ppo_openpi" ;;
      metaworld_mt50) echo "metaworld_50_ppo_openpi" ;;
      calvin_d_d) echo "calvin_d_d_ppo_openpi" ;;
      *) echo "Unsupported suite: $1" >&2; return 1 ;;
    esac
  fi
}

model_dir_for_suite() {
  if [[ -n "${MODEL_DIR}" ]]; then
    echo "${MODEL_DIR}"
    return
  fi
  case "$1" in
    libero_spatial | libero_object | libero_goal) echo "${LIBERO_MODEL_DIR}" ;;
    libero_10) echo "${LIBERO_LONG_MODEL_DIR}" ;;
    maniskill) echo "${MANISKILL_MODEL_DIR}" ;;
    metaworld_mt50) echo "${METAWORLD_MODEL_DIR}" ;;
    calvin_d_d) echo "${CALVIN_MODEL_DIR}" ;;
    *) echo "Unsupported suite: $1" >&2; return 1 ;;
  esac
}

method_overrides() {
  case "$1" in
    flow_noise_baseline)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    scoreflow_original)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=learned_alpha ++actor.model.openpi.score_flow_scale=1.0 ++actor.model.openpi.score_flow_clip_norm=10.0 ++actor.model.openpi.score_flow_alpha_hidden_dim=16 ++actor.model.openpi.score_flow_alpha_init_bias=-2.0 ++actor.model.openpi.score_flow_alpha_max=2.0 ++actor.model.openpi.score_flow_use_time_mask=true ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_scalar_l2)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=scalar_l2 ++actor.model.openpi.tr_penalty_beta=1.0 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_pullback)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=terminal_pullback ++actor.model.openpi.tr_penalty_beta=1.0 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_pullback_matched)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=terminal_pullback ++actor.model.openpi.tr_penalty_beta=0.03 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    cr_reflow_no_anchor)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=cr_reflow_no_anchor ++actor.model.openpi.cr_reflow_kl_epsilon=0.05 ++actor.model.openpi.cr_reflow_eta_min=0.01 ++actor.model.openpi.cr_reflow_eta_max=10.0 ++actor.model.openpi.cr_reflow_weight_clip=10.0 ++actor.model.openpi.cr_reflow_anchor_beta=0.0"
      ;;
    cr_reflow)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true algorithm.entropy_bonus=0.005 ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=cr_reflow ++actor.model.openpi.cr_reflow_kl_epsilon=0.05 ++actor.model.openpi.cr_reflow_eta_min=0.01 ++actor.model.openpi.cr_reflow_eta_max=10.0 ++actor.model.openpi.cr_reflow_weight_clip=10.0 ++actor.model.openpi.cr_reflow_anchor_beta=0.1"
      ;;
    *)
      echo "Unsupported method: $1" >&2
      return 1
      ;;
  esac
}

patch_rlinf() {
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/patch_openpi_score_flow.py" \
    --rlinf-root "${RLINF_ROOT}"
  "${PYTHON_BIN}" -m py_compile \
    "${RLINF_ROOT}/rlinf/models/embodiment/openpi/openpi_action_model.py"
}

append_manifest_header() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "suite,config,model_dir,method,seed,run_name,status,run_dir,start_time,end_time,exit_code" > "${MANIFEST}"
  fi
}

run_one() {
  local suite="$1"
  local method="$2"
  local seed="$3"
  local config_name suite_model_dir run_name run_dir run_log start_time end_time exit_code status
  config_name="$(config_for_suite "${suite}")"
  suite_model_dir="$(model_dir_for_suite "${suite}")"
  run_name="${suite}_${method}_seed${seed}"
  run_dir="${EXP_ROOT}/${suite}/${run_name}"
  run_log="${run_dir}/run.log"
  mkdir -p "${run_dir}"
  start_time="$(date -Iseconds)"

  local cmd=(
    "${PYTHON_BIN}" "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py"
    --config-path "${RLINF_ROOT}/examples/embodiment/config/"
    --config-name "${config_name}"
    "runner.logger.log_path=${run_dir}"
    "runner.logger.experiment_name=${run_name}"
    "runner.max_epochs=${MAX_EPOCHS}"
    "runner.val_check_interval=1"
    "runner.save_interval=${SAVE_INTERVAL}"
    "algorithm.rollout_epoch=${ROLLOUT_EPOCH}"
    "algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
    "env.train.total_num_envs=${TRAIN_ENVS}"
    "env.eval.total_num_envs=${EVAL_ENVS}"
    "actor.seed=${seed}"
    "actor.micro_batch_size=${MICRO_BATCH_SIZE}"
    "actor.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "actor.enable_offload=true"
    "rollout.enable_offload=true"
    "actor.model.model_path=${suite_model_dir}"
    "rollout.model.model_path=${suite_model_dir}"
  )

  for override in ${EXTRA_OVERRIDES}; do
    cmd+=("${override}")
  done
  for override in $(method_overrides "${method}"); do
    cmd+=("${override}")
  done

  printf "%q " "${cmd[@]}" > "${run_dir}/command.txt"
  echo >> "${run_dir}/command.txt"

  echo "[$(date)] START ${run_name}"
  set +e
  "${cmd[@]}" 2>&1 | tee "${run_log}"
  exit_code="${PIPESTATUS[0]}"
  set -e
  end_time="$(date -Iseconds)"
  if [[ "${exit_code}" == "0" ]]; then
    status="done"
  else
    status="failed"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${suite}" "${config_name}" "${suite_model_dir}" "${method}" "${seed}" "${run_name}" "${status}" "${run_dir}" \
    "${start_time}" "${end_time}" "${exit_code}" >> "${MANIFEST}"
}

patch_rlinf
append_manifest_header
for suite in ${SUITES}; do
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      run_one "${suite}" "${method}" "${seed}"
    done
  done
done
