import torch
import importlib.util
import os


def _preflight_fairasr():
    required_modules = {
        "wandb": "wandb",
        "nemo": "nemo_toolkit[asr]",
        "pytorch_lightning": "pytorch-lightning",
        "lightning": "lightning",
        "torchaudio": "torchaudio",
    }
    missing = [
        package
        for module, package in required_modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "FairASR dependencies are missing from this environment: "
            + ", ".join(missing)
        )

    if not torch.cuda.is_available():
        raise RuntimeError("FairASR training/validation requires an available CUDA GPU.")

    experiment_root = os.path.join(
        os.path.dirname(__file__), "FairASR", "experiments"
    )
    simclr_checkpoint = os.path.join(
        experiment_root,
        "nemo_fairaudio_pretrain",
        "simclr_supconGRL_samespace_checkpoints",
        "last.ckpt",
    )
    final_model = os.path.join(
        experiment_root,
        "nemo_fairaudio",
        "simclr_supconGRL_1e-1_diffspace",
        "2026-01-07_14-46-16",
        "checkpoints",
        "simclr_supconGRL_1e-1_diffspace.nemo",
    )
    missing_files = [
        path for path in [simclr_checkpoint, final_model] if not os.path.exists(path)
    ]
    if missing_files:
        raise RuntimeError(
            "FairASR checkpoints are missing: " + ", ".join(missing_files)
        )

def FairASR():
    _preflight_fairasr()
    from MAF.algorithms.inprocessing.FairASR.train_simclr import run_train_simclr
    from MAF.algorithms.inprocessing.FairASR.train_hf import run_train_asr
    from MAF.algorithms.inprocessing.FairASR.validation import run_validation 
    
    print("[1/3] SimCLR Pretraining")
    run_train_simclr(
        independent_space=True,
        balance_param=0.1
    )
    torch.cuda.empty_cache()

    print("[2/3] ASR Fine-tuning")
    run_train_asr()
    torch.cuda.empty_cache()

    print("[3/3] Validation")
    metrics = run_validation()
    print('results:', metrics)
    print("FairASR Pipeline Finished")
    return metrics


if __name__ == "__main__":
    metrics = FairASR()
