import importlib

from MAF.algorithms.inprocessing.exponentiated_gradient_reduction import (
    ExponentiatedGradientReduction,
)
from MAF.algorithms.inprocessing.meta_classifier import MetaFairClassifier
from MAF.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
from MAF.algorithms.inprocessing.prejudice_remover import PrejudiceRemover
from MAF.algorithms.inprocessing.slide import SlideFairClassifier
from MAF.algorithms.inprocessing.ftm import FTMFairClassifier
from MAF.algorithms.inprocessing.fair_dimension_filtering import FairDimFilter
from MAF.algorithms.inprocessing.fair_feature_distillation import (
    FairFeatureDistillation,
)
from MAF.algorithms.inprocessing.fairness_vae import FairnessVAE
from MAF.algorithms.inprocessing.kernel_density_estimation import (
    KDEParameters,
    KernelDensityEstimation,
)
from MAF.algorithms.inprocessing.learning_from_fairness import LearningFromFairness
from MAF.algorithms.inprocessing.sipm_lfr import SIPMLFR
from MAF.algorithms.inprocessing.gerry_fair_classifier import GerryFairClassifier
from MAF.algorithms.inprocessing.grid_search_reduction import GridSearchReduction
from MAF.algorithms.inprocessing.dmlbg import *


def __getattr__(name):
    optional_modules = {
        "concse": "MAF.algorithms.inprocessing.concse",
        "fairasr": "MAF.algorithms.inprocessing.fairasr",
    }
    optional_symbols = {
        "mitigate_concse": ("MAF.algorithms.inprocessing.concse", "mitigate_concse"),
        "mitigate_intapt": (
            "MAF.algorithms.inprocessing.INTapt.intapt",
            "mitigate_intapt",
        ),
        "FairASR": ("MAF.algorithms.inprocessing.fairasr", "FairASR"),
    }

    if name in optional_modules:
        module = importlib.import_module(optional_modules[name])
        globals()[name] = module
        return module

    if name in optional_symbols:
        module_name, symbol_name = optional_symbols[name]
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
        globals()[name] = symbol
        return symbol

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
