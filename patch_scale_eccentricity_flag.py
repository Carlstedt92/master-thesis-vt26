"""
One-off patch: scale_eccentricity was added as a NEW ModelConfig field after
KHOP_EXTFEAT_ECC_SCALED_TEST_60EP had already finished training. Its saved
checkpoints' config dicts predate the field entirely, so ModelConfig.from_dict()
(which fills missing keys with the dataclass default, False) would silently
reconstruct scale_eccentricity=False for this model -- wrong, since it's the
one model that was actually trained on scaled eccentricity. Patches every
checkpoint's stored config dict in place to add scale_eccentricity=True.
"""

import glob
import torch

MODEL_DIR = "models/KHOP_EXTFEAT_ECC_SCALED_TEST_60EP/checkpoints"

paths = sorted(glob.glob(f"{MODEL_DIR}/*.pth"))
print(f"Found {len(paths)} checkpoints to patch.")

for path in paths:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    already_set = ckpt["config"].get("scale_eccentricity")
    ckpt["config"]["scale_eccentricity"] = True
    torch.save(ckpt, path)
    print(f"  {path}: scale_eccentricity {already_set} -> True")

print("Done.")
