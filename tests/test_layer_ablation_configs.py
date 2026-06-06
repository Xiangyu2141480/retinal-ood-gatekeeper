from copy import deepcopy
from pathlib import Path

from retinal_ood.utils.io import read_yaml


ABLATION_CONFIGS = {
    "patchcore_l1.yaml": ("patchcore_resnet50_layer1", ["layer1"]),
    "patchcore_l2.yaml": ("patchcore_resnet50_layer2", ["layer2"]),
    "patchcore_l3.yaml": ("patchcore_resnet50_layer3", ["layer3"]),
    "patchcore_l4.yaml": ("patchcore_resnet50_layer4", ["layer4"]),
    "patchcore_l23.yaml": ("patchcore_resnet50_layer2_layer3", ["layer2", "layer3"]),
}


def test_layer_ablation_configs_parse_and_define_expected_layers():
    configs_dir = Path("configs")

    for filename, (run_name, layers) in ABLATION_CONFIGS.items():
        config = read_yaml(configs_dir / filename)

        assert config["project"]["run_name"] == run_name
        assert config["project"]["seed"] == 42
        assert config["model"]["name"] == "patchcore"
        assert config["model"]["backbone"] == "resnet50"
        assert config["model"]["layers"] == layers
        assert config["data"]["train_manifest"] == "data/manifests/train_synthetic_faf.csv"
        assert config["data"]["val_manifest"] == "data/manifests/val_synthetic_faf.csv"
        assert config["data"]["test_id_manifest"] == "data/manifests/test_real_id.csv"
        assert config["data"]["test_ood_manifest"] == "data/manifests/test_ood.csv"
        assert config["data"]["root_dir"] == "data"


def test_layer_ablation_configs_only_differ_by_run_name_and_layers():
    configs_dir = Path("configs")
    normalized_configs = {}
    for filename in ABLATION_CONFIGS:
        config = deepcopy(read_yaml(configs_dir / filename))
        config["project"]["run_name"] = "<run_name>"
        config["model"]["layers"] = ["<layers>"]
        normalized_configs[filename] = config

    baseline = normalized_configs["patchcore_l23.yaml"]
    for filename, config in normalized_configs.items():
        assert config == baseline, f"{filename} differs outside run_name/model.layers"
