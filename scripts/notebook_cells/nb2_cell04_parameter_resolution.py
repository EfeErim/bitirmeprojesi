# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 4.
# Resolves Notebook 2 parameters after the user-editable values are set.

# Notebook icinde daha sonra kullanilacak gizli repo varsayimlarini yukle.
BASE_CONFIG = ConfigurationManager(config_dir=str(ROOT / "config"), environment="colab").load_all_configs()
CONTINUAL_DATA_CFG = BASE_CONFIG.get("training", {}).get("continual", {}).get("data", {})

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = bool(TF32_ENABLED)

TARGET_SIZE = int(CONTINUAL_DATA_CFG.get("target_size", 224))
DATA_SAMPLER = str(CONTINUAL_DATA_CFG.get("sampler", "shuffle"))
AUGMENTATION_POLICY = str(CONTINUAL_DATA_CFG.get("augmentation_policy", "randaugment")).strip().lower()
RANDAUGMENT_NUM_OPS = int(CONTINUAL_DATA_CFG.get("randaugment_num_ops", 2))
RANDAUGMENT_MAGNITUDE = int(CONTINUAL_DATA_CFG.get("randaugment_magnitude", 7))
ALLOW_UNDER_MIN_TRAINING = bool(ALLOW_UNDER_MIN_TRAINING)
LOADER_ERROR_POLICY = str(CONTINUAL_DATA_CFG.get("loader_error_policy", "tolerant"))
CACHE_SIZE = int(CONTINUAL_DATA_CFG.get("cache_size", 1000))
VALIDATE_IMAGES_ON_INIT = bool(CONTINUAL_DATA_CFG.get("validate_images_on_init", True))

from scripts.colab_training_recommendations import resolve_effective_notebook_params

def _collect_notebook_base_params():
    return {
        "ALLOW_UNDER_MIN_TRAINING": bool(ALLOW_UNDER_MIN_TRAINING),
        "EPOCHS": int(EPOCHS),
        "BATCH_SIZE": int(BATCH_SIZE),
        "LEARNING_RATE": float(LEARNING_RATE),
        "LORA_R": int(LORA_R),
        "LORA_ALPHA": int(LORA_ALPHA),
        "LORA_DROPOUT": float(LORA_DROPOUT),
        "OOD_FACTOR": float(OOD_FACTOR),
        "SURE_SEMANTIC_PERCENTILE": float(SURE_SEMANTIC_PERCENTILE),
        "SURE_CONFIDENCE_PERCENTILE": float(SURE_CONFIDENCE_PERCENTILE),
        "CONFORMAL_ALPHA": float(CONFORMAL_ALPHA),
        "CONFORMAL_METHOD": str(CONFORMAL_METHOD),
        "CONFORMAL_RAPS_LAMBDA": float(CONFORMAL_RAPS_LAMBDA),
        "CONFORMAL_RAPS_K_REG": int(CONFORMAL_RAPS_K_REG),
        "BER_ENABLED": bool(BER_ENABLED),
        "BER_LAMBDA_OLD": float(BER_LAMBDA_OLD),
        "BER_LAMBDA_NEW": float(BER_LAMBDA_NEW),
        "BER_WARMUP_STEPS": int(BER_WARMUP_STEPS),
        "RANDAUGMENT_NUM_OPS": int(RANDAUGMENT_NUM_OPS),
        "RANDAUGMENT_MAGNITUDE": int(RANDAUGMENT_MAGNITUDE),
        "WEIGHT_DECAY": float(WEIGHT_DECAY),
        "MIXED_PRECISION": str(MIXED_PRECISION),
        "GRAD_ACCUM_STEPS": int(GRAD_ACCUM_STEPS),
        "MAX_GRAD_NORM": float(MAX_GRAD_NORM),
        "LABEL_SMOOTHING": float(LABEL_SMOOTHING),
        "LOSS_NAME": str(LOSS_NAME),
        "LOGITNORM_TAU": float(LOGITNORM_TAU),
        "SCHEDULER_NAME": str(SCHEDULER_NAME),
        "SCHEDULER_WARMUP_RATIO": float(SCHEDULER_WARMUP_RATIO),
        "SCHEDULER_MIN_LR": float(SCHEDULER_MIN_LR),
        "EARLY_STOPPING_PATIENCE": int(EARLY_STOPPING_PATIENCE),
        "EARLY_STOPPING_MIN_DELTA": float(EARLY_STOPPING_MIN_DELTA),
        "DETERMINISTIC": bool(DETERMINISTIC),
        "SEED": int(SEED),
        "NUM_WORKERS": int(NUM_WORKERS),
        "PREFETCH": int(PREFETCH),
        "PIN_MEMORY": bool(PIN_MEMORY),
        "USE_CACHE": bool(USE_CACHE),
        "CACHE_TRAIN_SPLIT": bool(CACHE_TRAIN_SPLIT),
        "VALIDATION_EVERY_N_EPOCHS": int(VALIDATION_EVERY_N_EPOCHS),
        "CHECKPOINT_EVERY_N_STEPS": int(CHECKPOINT_EVERY_N_STEPS),
        "CHECKPOINT_ON_EXCEPTION": bool(CHECKPOINT_ON_EXCEPTION),
        "STDOUT_BATCH_INTERVAL": int(STDOUT_BATCH_INTERVAL),
        "RESUME_MODE": str(RESUME_MODE),
        "AUTO_DISCONNECT_RUNTIME": bool(AUTO_DISCONNECT_RUNTIME),
        "AUTO_DISCONNECT_GRACE_SECONDS": int(AUTO_DISCONNECT_GRACE_SECONDS),
        "AUTO_PUSH_TO_GITHUB": bool(AUTO_PUSH_TO_GITHUB),
        "AUTO_PUSH_REMOTE_NAME": str(AUTO_PUSH_REMOTE_NAME),
        "AUTO_PUSH_BRANCH": AUTO_PUSH_BRANCH,
        "ENABLE_BAYESIAN_OPTIMIZATION": bool(ENABLE_BAYESIAN_OPTIMIZATION),
    }

def _collect_bayesian_notebook_overrides():
    if not bool(ENABLE_BAYESIAN_OPTIMIZATION):
        return {}
    recommendations_path = ROOT / "runs" / "_index" / "bayesian_recommendations.json"
    if not recommendations_path.exists():
        print(f"[BAYES] Toggle acik ama dosya yok: {recommendations_path}")
        return {}
    try:
        payload = json.loads(recommendations_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[BAYES] Oneri dosyasi okunamadi: {exc}")
        return {}

    cohorts = list(payload.get("cohorts", [])) if isinstance(payload, dict) else []
    selected_cohort = None
    for cohort in cohorts:
        comparability = cohort.get("comparability", {}) if isinstance(cohort, dict) else {}
        if str(comparability.get("crop_name", "")).strip().lower() == str(CROP_NAME).strip().lower() and str(comparability.get("part_name", "")).strip().lower() == str(PART_NAME).strip().lower():
            selected_cohort = cohort
            break
    if selected_cohort is None and cohorts:
        selected_cohort = cohorts[0]

    proposals = list((selected_cohort or {}).get("proposals", [])) if isinstance(selected_cohort, dict) else []
    if not proposals:
        print("[BAYES] Uygulanabilir oneriler bulunamadi.")
        return {}

    proposal = proposals[0]
    parameters = proposal.get("parameters", {}) if isinstance(proposal, dict) else {}
    if not isinstance(parameters, dict):
        return {}

    mapping = {
        "training.learning_rate": "LEARNING_RATE",
        "training.weight_decay": "WEIGHT_DECAY",
        "training.num_epochs": "EPOCHS",
        "training.batch_size": "BATCH_SIZE",
        "training.adapter.lora_r": "LORA_R",
        "training.adapter.lora_alpha": "LORA_ALPHA",
        "training.adapter.lora_dropout": "LORA_DROPOUT",
        "training.ood.threshold_factor": "OOD_FACTOR",
        "training.optimization.logitnorm_tau": "LOGITNORM_TAU",
        "training.data.randaugment_num_ops": "RANDAUGMENT_NUM_OPS",
        "training.data.randaugment_magnitude": "RANDAUGMENT_MAGNITUDE",
    }
    overrides = {}
    for source_key, target_key in mapping.items():
        if source_key in parameters:
            overrides[target_key] = parameters[source_key]

    rank = int(proposal.get("rank", 1)) if str(proposal.get("rank", "")).strip() else 1
    print(f"[BAYES] rank={rank} ile {len(overrides)} parametre onerisi yuklendi.")
    return overrides

bayesian_overrides = _collect_bayesian_notebook_overrides()
resolved_manual_overrides = dict(bayesian_overrides)
resolved_manual_overrides.update(dict(MANUAL_PARAM_OVERRIDES or {}))

INITIAL_EFFECTIVE_PARAMS = resolve_effective_notebook_params(
    _collect_notebook_base_params(),
    {"recommended_params": _collect_notebook_base_params()},
    resolved_manual_overrides,
    accepted=False,
)

STATE = {
    "validated": False,
    "class_names": [],
    "runtime_dataset_root": None,
    "runtime_dataset_key": None,
    "selected_dataset_name": None,
    "selected_dataset_root": None,
    "resolved_ood_root": None,
    "resolved_oe_root": None,
    "dataset_inspection": {},
    "hardware_inspection": {},
    "recommendation_report": {},
    "recommendation_decision": "pending",
    "effective_params": dict(INITIAL_EFFECTIVE_PARAMS),
    "adapter": None,
    "loaders": None,
    "history": None,
    "calibration": None,
    "checkpoint_manager": CHECKPOINT_MANAGER,
    "resume_manifest": None,
    "best_val_loss": None,
    "best_metric_state": {},
    "auto_disconnect_report": None,
    "git_push_report": None,
    "telemetry": TELEMETRY,
}

if str(INITIAL_EFFECTIVE_PARAMS.get("RESUME_MODE", RESUME_MODE)) == "resume":
    try:
        STATE["resume_manifest"] = CHECKPOINT_MANAGER.get_latest()
        if STATE["resume_manifest"]:
            manifest = STATE["resume_manifest"]
            print(
                f"[RESUME] Checkpoint bulundu: {manifest.get('name', '?')} "
                f"epoch={manifest.get('epoch', 0)} step={manifest.get('global_step', 0)}"
            )
    except Exception:
        pass

hf_token_ready = False
hf_token = str(resolve_hf_token() or "").strip()
if not hf_token:
    raise RuntimeError(
        "Notebook 2 egitim baslamadan once Hugging Face token ister. "
        "HF_TOKEN, HUGGINGFACE_TOKEN veya HUGGINGFACE_HUB_TOKEN degerini env var ya da Colab secret olarak tanimlayin."
    )
hf_token_ready = bool(login_and_check_hf_token(print_fn=print))
if not hf_token_ready:
    raise RuntimeError(
        "Hugging Face token on kontrolu basarisiz oldu. Egitimden once tokeni duzeltin."
    )

github_token_ready = False
if bool(INITIAL_EFFECTIVE_PARAMS.get("AUTO_PUSH_TO_GITHUB", AUTO_PUSH_TO_GITHUB)):
    github_token = str(resolve_github_token() or "").strip()
    if not github_token:
        raise RuntimeError(
            "AUTO_PUSH_TO_GITHUB is enabled, but no GitHub token was found. "
            "Egitim baslamadan once GH_TOKEN veya GITHUB_TOKEN degerini env var ya da Colab secret olarak tanimlayin."
        )
    github_token_ready = True
    print("[GIT] Auto-push on kontrolu gecti: GitHub token cozuldu.")

print(
    f"[CONFIG] source=notebook_cell crop={CROP_NAME} epochs={EPOCHS} bs={BATCH_SIZE} "
    f"lr={LEARNING_RATE} resume={RESUME_MODE} ber={BER_ENABLED} "
    f"loss={LOSS_NAME} tau={LOGITNORM_TAU} auto_disconnect={AUTO_DISCONNECT_RUNTIME} "
    f"auto_push={AUTO_PUSH_TO_GITHUB} bayes_opt={ENABLE_BAYESIAN_OPTIMIZATION}"
)
print(
    f"[DATASET] runtime_root={RUNTIME_DATASET_ROOT} dataset_name={DATASET_NAME or '<ask>'}"
)
print(
    f"[OOD] ood_root={OOD_ROOT or '<ask>'} ask_for_ood_root={ASK_FOR_OOD_ROOT}"
)
print(
    f"[OE] oe_root={OE_ROOT or '<ask>'} ask_for_oe_root={ASK_FOR_OE_ROOT} "
    f"enabled={OE_ENABLED} loss_weight={OE_LOSS_WEIGHT}"
)
print(
    f"[RUNTIME] defaults=notebook_cell mp={MIXED_PRECISION} workers={NUM_WORKERS} prefetch={PREFETCH} "
    f"sched={SCHEDULER_NAME} wd={WEIGHT_DECAY} accum={GRAD_ACCUM_STEPS} grad_clip={MAX_GRAD_NORM} "
    f"label_smooth={LABEL_SMOOTHING} warmup={SCHEDULER_WARMUP_RATIO} "
    f"early_stop={EARLY_STOPPING_PATIENCE}/{EARLY_STOPPING_MIN_DELTA} "
    f"val_every={VALIDATION_EVERY_N_EPOCHS} cache_train={CACHE_TRAIN_SPLIT} "
    f"aug={AUGMENTATION_POLICY} randaug={RANDAUGMENT_NUM_OPS}/{RANDAUGMENT_MAGNITUDE} "
)
print(
    f"[OOD] factor={OOD_FACTOR} sure={SURE_SEMANTIC_PERCENTILE}/{SURE_CONFIDENCE_PERCENTILE} "
    f"conformal={CONFORMAL_METHOD} alpha={CONFORMAL_ALPHA} "
    f"raps_lambda={CONFORMAL_RAPS_LAMBDA} raps_k={CONFORMAL_RAPS_K_REG}"
)
TELEMETRY.update_latest(
    {
        "phase": "parameters_ready",
        "parameter_source": "notebook_cell",
        "crop": CROP_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_name": DATASET_NAME,
        "ood_root": OOD_ROOT,
        "ask_for_ood_root": ASK_FOR_OOD_ROOT,
        "oe_root": OE_ROOT,
        "ask_for_oe_root": ASK_FOR_OE_ROOT,
        "oe_enabled": OE_ENABLED,
        "oe_loss_weight": OE_LOSS_WEIGHT,
        "loss_name": LOSS_NAME,
        "logitnorm_tau": LOGITNORM_TAU,
        "mixed_precision": MIXED_PRECISION,
        "num_workers": NUM_WORKERS,
        "prefetch": PREFETCH,
        "cache_train_split": CACHE_TRAIN_SPLIT,
        "augmentation_policy": AUGMENTATION_POLICY,
        "randaugment_num_ops": RANDAUGMENT_NUM_OPS,
        "randaugment_magnitude": RANDAUGMENT_MAGNITUDE,
        "allow_under_min_training": ALLOW_UNDER_MIN_TRAINING,
        "ber_enabled": BER_ENABLED,
        "ber_lambda_old": BER_LAMBDA_OLD,
        "ber_lambda_new": BER_LAMBDA_NEW,
        "ber_warmup_steps": BER_WARMUP_STEPS,
        "ood_factor": OOD_FACTOR,
        "sure_semantic_percentile": SURE_SEMANTIC_PERCENTILE,
        "sure_confidence_percentile": SURE_CONFIDENCE_PERCENTILE,
        "conformal_alpha": CONFORMAL_ALPHA,
        "conformal_method": CONFORMAL_METHOD,
        "conformal_raps_lambda": CONFORMAL_RAPS_LAMBDA,
        "conformal_raps_k_reg": CONFORMAL_RAPS_K_REG,
        "label_smoothing": LABEL_SMOOTHING,
        "scheduler_warmup_ratio": SCHEDULER_WARMUP_RATIO,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "resume_mode": RESUME_MODE,
        "validation_every_n_epochs": VALIDATION_EVERY_N_EPOCHS,
        "enable_bayesian_optimization": bool(ENABLE_BAYESIAN_OPTIMIZATION),
        "auto_push_to_github": AUTO_PUSH_TO_GITHUB,
        "hf_token_ready": hf_token_ready,
        "auto_push_token_ready": github_token_ready,
        "auto_push_remote_name": AUTO_PUSH_REMOTE_NAME,
        "auto_push_branch": AUTO_PUSH_BRANCH,
        "auto_disconnect_runtime": AUTO_DISCONNECT_RUNTIME,
        "auto_disconnect_grace_seconds": AUTO_DISCONNECT_GRACE_SECONDS,
    }
)
