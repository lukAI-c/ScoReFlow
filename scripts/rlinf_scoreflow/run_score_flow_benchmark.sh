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

REAL_CONFIG_PRESET="${REAL_CONFIG_PRESET:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
PATCH_RLINF="${PATCH_RLINF:-1}"
PIRL_OFFICIAL_PROTOCOL="${PIRL_OFFICIAL_PROTOCOL:-0}"
MODEL_PROVENANCE="${MODEL_PROVENANCE:-}"
CR_REFLOW_KL_EPSILON="${CR_REFLOW_KL_EPSILON:-0.05}"
CR_REFLOW_ETA_MIN="${CR_REFLOW_ETA_MIN:-0.01}"
CR_REFLOW_ETA_MAX="${CR_REFLOW_ETA_MAX:-10.0}"
CR_REFLOW_WEIGHT_CLIP="${CR_REFLOW_WEIGHT_CLIP:-10.0}"
CR_REFLOW_ANCHOR_BETA="${CR_REFLOW_ANCHOR_BETA:-0.1}"
if [[ "${REAL_CONFIG_PRESET}" == "1" ]]; then
  POLICY_VARIANT="${POLICY_VARIANT:-pi05}"
  SUITES="${SUITES:-libero_spatial}"
  SEEDS="${SEEDS:-42}"
  METHODS="${METHODS:-cr_reflow_no_anchor cr_reflow}"
  MAX_EPOCHS="${MAX_EPOCHS:-15}"
  ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
  EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-1}"
  TRAIN_ENVS="${TRAIN_ENVS:-4}"
  EVAL_ENVS="${EVAL_ENVS:-8}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
fi
if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
  POLICY_VARIANT="pi05"
  MAX_EPOCHS="500"
  ROLLOUT_EPOCH="8"
  EVAL_ROLLOUT_EPOCH="1"
  TRAIN_ENVS="64"
  EVAL_ENVS="500"
  MICRO_BATCH_SIZE="128"
  GLOBAL_BATCH_SIZE="2048"
  VAL_CHECK_INTERVAL="-1"
  SAVE_INTERVAL="${SAVE_INTERVAL:-40}"
  if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
    echo "MODEL_DIR must be an explicit existing directory for PIRL_OFFICIAL_PROTOCOL=1" >&2
    exit 1
  fi
  if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    echo "EXTRA_OVERRIDES is forbidden for PIRL_OFFICIAL_PROTOCOL=1; use method-specific variables" >&2
    exit 1
  fi
fi

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
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

is_pirl_suite() {
  case "$1" in
    libero_spatial | libero_object | libero_goal | libero_10) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
  for suite in ${SUITES}; do
    if ! is_pirl_suite "${suite}"; then
      echo "PIRL_OFFICIAL_PROTOCOL=1 only supports LIBERO suites, got: ${suite}" >&2
      exit 1
    fi
  done
fi

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
PIRL_PROTOCOL_TOOL="${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_protocol.py"
PIRL_REPRO_TOOL="${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/pirl_repro_bundle.py"

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
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    scoreflow_original)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=learned_alpha ++actor.model.openpi.score_flow_scale=1.0 ++actor.model.openpi.score_flow_clip_norm=10.0 ++actor.model.openpi.score_flow_alpha_hidden_dim=16 ++actor.model.openpi.score_flow_alpha_init_bias=-2.0 ++actor.model.openpi.score_flow_alpha_max=2.0 ++actor.model.openpi.score_flow_use_time_mask=true ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_scalar_l2)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=scalar_l2 ++actor.model.openpi.tr_penalty_beta=1.0 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_pullback)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=terminal_pullback ++actor.model.openpi.tr_penalty_beta=1.0 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    tr_pullback_matched)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=terminal_pullback ++actor.model.openpi.tr_penalty_beta=0.03 ++actor.model.openpi.cr_reflow_mode=none"
      ;;
    cr_reflow_no_anchor)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=cr_reflow_no_anchor ++actor.model.openpi.cr_reflow_kl_epsilon=${CR_REFLOW_KL_EPSILON} ++actor.model.openpi.cr_reflow_eta_min=${CR_REFLOW_ETA_MIN} ++actor.model.openpi.cr_reflow_eta_max=${CR_REFLOW_ETA_MAX} ++actor.model.openpi.cr_reflow_weight_clip=${CR_REFLOW_WEIGHT_CLIP} ++actor.model.openpi.cr_reflow_anchor_beta=0.0"
      ;;
    cr_reflow)
      echo "actor.model.openpi.noise_method=flow_noise ++actor.model.openpi.joint_logprob=true ++actor.model.openpi.score_flow_mode=none ++actor.model.openpi.score_flow_scale=0.0 ++actor.model.openpi.tr_penalty_mode=none ++actor.model.openpi.cr_reflow_mode=cr_reflow ++actor.model.openpi.cr_reflow_kl_epsilon=${CR_REFLOW_KL_EPSILON} ++actor.model.openpi.cr_reflow_eta_min=${CR_REFLOW_ETA_MIN} ++actor.model.openpi.cr_reflow_eta_max=${CR_REFLOW_ETA_MAX} ++actor.model.openpi.cr_reflow_weight_clip=${CR_REFLOW_WEIGHT_CLIP} ++actor.model.openpi.cr_reflow_anchor_beta=${CR_REFLOW_ANCHOR_BETA}"
      ;;
    *)
      echo "Unsupported method: $1" >&2
      return 1
      ;;
  esac
}

protocol_suite_value() {
  local suite="$1"
  local field="$2"
  case "${suite}:${field}" in
    libero_spatial:interaction_steps) echo "240" ;;
    libero_object:interaction_steps | libero_goal:interaction_steps) echo "320" ;;
    libero_10:interaction_steps) echo "480" ;;
    libero_spatial:update_epochs | libero_object:update_epochs) echo "1" ;;
    libero_goal:update_epochs) echo "3" ;;
    libero_10:update_epochs) echo "4" ;;
    libero_spatial:action_replan_horizon | libero_object:action_replan_horizon | libero_goal:action_replan_horizon) echo "5" ;;
    libero_10:action_replan_horizon) echo "10" ;;
    libero_spatial:denoise_steps) echo "3" ;;
    libero_object:denoise_steps | libero_goal:denoise_steps | libero_10:denoise_steps) echo "5" ;;
    libero_10:scheduler) echo "true" ;;
    libero_spatial:scheduler | libero_object:scheduler | libero_goal:scheduler) echo "false" ;;
    *) echo "Unsupported pi_RL protocol field: ${suite}:${field}" >&2; return 1 ;;
  esac
}

patch_rlinf() {
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/patch_openpi_score_flow.py" \
    --rlinf-root "${RLINF_ROOT}"
  "${PYTHON_BIN}" "${SCOREFLOW_ROOT}/scripts/rlinf_scoreflow/patch_rlinf_cr_reflow_actor.py" \
    --rlinf-root "${RLINF_ROOT}"
  "${PYTHON_BIN}" -m py_compile \
    "${RLINF_ROOT}/rlinf/models/embodiment/openpi/openpi_action_model.py" \
    "${RLINF_ROOT}/rlinf/workers/actor/fsdp_actor_worker.py"
}

append_manifest_header() {
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "suite,config,model_dir,model_provenance,protocol_id,protocol_artifact,method,seed,run_name,status,run_dir,start_time,end_time,exit_code" > "${MANIFEST}"
  fi
}

run_one() {
  local suite="$1"
  local method="$2"
  local seed="$3"
  local config_name suite_model_dir run_name run_dir run_log start_time end_time exit_code status
  local interaction_steps update_epochs action_replan_horizon denoise_steps scheduler
  local protocol_id protocol_artifact model_provenance
  local reproducibility_bundle
  config_name="$(config_for_suite "${suite}")"
  suite_model_dir="$(model_dir_for_suite "${suite}")"
  run_name="${suite}_${method}_seed${seed}"
  run_dir="${EXP_ROOT}/${suite}/${run_name}"
  run_log="${run_dir}/run.log"
  protocol_artifact=""
  reproducibility_bundle="${run_dir}/reproducibility.json"
  model_provenance="${MODEL_PROVENANCE:-unverified:${suite_model_dir}}"
  interaction_steps="0"
  update_epochs="0"
  action_replan_horizon="0"
  denoise_steps="0"
  scheduler="false"
  protocol_id="development_unaligned"
  if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
    interaction_steps="$(protocol_suite_value "${suite}" interaction_steps)"
    update_epochs="$(protocol_suite_value "${suite}" update_epochs)"
    action_replan_horizon="$(protocol_suite_value "${suite}" action_replan_horizon)"
    denoise_steps="$(protocol_suite_value "${suite}" denoise_steps)"
    scheduler="$(protocol_suite_value "${suite}" scheduler)"
    protocol_id="pirl_pi05_libero_arxiv_2510.25889v3"
    protocol_artifact="${run_dir}/training_protocol.json"
  fi
  mkdir -p "${run_dir}"
  start_time="$(date -Iseconds)"

  local cmd=(
    "${PYTHON_BIN}" "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py"
    --config-path "${RLINF_ROOT}/examples/embodiment/config/"
    --config-name "${config_name}"
    "runner.logger.log_path=${run_dir}"
    "runner.logger.experiment_name=${run_name}"
    "runner.max_epochs=${MAX_EPOCHS}"
    "runner.val_check_interval=${VAL_CHECK_INTERVAL}"
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

  if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
    cmd+=(
      "algorithm.update_epoch=${update_epochs}"
      "algorithm.gamma=0.99"
      "algorithm.gae_lambda=0.95"
      "algorithm.clip_ratio_high=0.2"
      "algorithm.clip_ratio_low=0.2"
      "env.train.max_episode_steps=${interaction_steps}"
      "env.train.max_steps_per_rollout_epoch=${interaction_steps}"
      "env.eval.max_episode_steps=${interaction_steps}"
      "env.eval.max_steps_per_rollout_epoch=${interaction_steps}"
      "actor.optim.lr=5.0e-6"
      "actor.optim.value_lr=1.0e-4"
      "actor.model.openpi.config_name=pi05_libero"
      "actor.model.num_action_chunks=${action_replan_horizon}"
      "actor.model.num_steps=${denoise_steps}"
      "++actor.model.openpi.noise_logvar_range=[0.04,0.10]"
      "algorithm.entropy_bonus=0.005"
    )
    if [[ "${scheduler}" == "true" ]]; then
      cmd+=("actor.optim.total_training_steps=500" "actor.optim.lr_scheduler=cosine")
    else
      cmd+=("++actor.optim.lr_scheduler=constant")
    fi
  fi

  for override in ${EXTRA_OVERRIDES}; do
    cmd+=("${override}")
  done
  if [[ "${PIRL_OFFICIAL_PROTOCOL}" != "1" ]]; then
    cmd+=("algorithm.entropy_bonus=0.005")
  fi
  for override in $(method_overrides "${method}"); do
    cmd+=("${override}")
  done

  printf "%q " "${cmd[@]}" > "${run_dir}/command.txt"
  echo >> "${run_dir}/command.txt"
  if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
    "${PYTHON_BIN}" "${PIRL_PROTOCOL_TOOL}" emit-training \
      --output "${protocol_artifact}" \
      --command-file "${run_dir}/command.txt" \
      --model-path "${suite_model_dir}" \
      --suite "${suite}" \
      --method "${method}"
    "${PYTHON_BIN}" "${PIRL_PROTOCOL_TOOL}" validate \
      --kind training \
      --artifact "${protocol_artifact}"
    model_provenance="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${protocol_artifact}'))['model_provenance'])")"
    "${PYTHON_BIN}" "${PIRL_REPRO_TOOL}" \
      --output "${reproducibility_bundle}" \
      --command-file "${run_dir}/command.txt" \
      --protocol-artifact "${protocol_artifact}" \
      --local-root "${SCOREFLOW_ROOT}" \
      --rlinf-root "${RLINF_ROOT}" \
      --status "prepared" \
      --exit-code 0
  fi

  if [[ "${PREPARE_ONLY}" == "1" ]]; then
    end_time="$(date -Iseconds)"
    echo "[$(date)] PREPARED ${run_name}"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "${suite}" "${config_name}" "${suite_model_dir}" "${model_provenance}" "${protocol_id}" "${protocol_artifact}" "${method}" "${seed}" "${run_name}" "prepared" "${run_dir}" \
      "${start_time}" "${end_time}" "0" >> "${MANIFEST}"
    return
  fi

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
  if [[ "${PIRL_OFFICIAL_PROTOCOL}" == "1" ]]; then
    "${PYTHON_BIN}" "${PIRL_REPRO_TOOL}" \
      --output "${reproducibility_bundle}" \
      --command-file "${run_dir}/command.txt" \
      --protocol-artifact "${protocol_artifact}" \
      --local-root "${SCOREFLOW_ROOT}" \
      --rlinf-root "${RLINF_ROOT}" \
      --status "${status}" \
      --exit-code "${exit_code}"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${suite}" "${config_name}" "${suite_model_dir}" "${model_provenance}" "${protocol_id}" "${protocol_artifact}" "${method}" "${seed}" "${run_name}" "${status}" "${run_dir}" \
    "${start_time}" "${end_time}" "${exit_code}" >> "${MANIFEST}"
}

if [[ "${PATCH_RLINF}" == "1" ]]; then
  patch_rlinf
fi
append_manifest_header
for suite in ${SUITES}; do
  for seed in ${SEEDS}; do
    for method in ${METHODS}; do
      run_one "${suite}" "${method}" "${seed}"
    done
  done
done
