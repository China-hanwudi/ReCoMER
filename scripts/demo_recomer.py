"""Exercise ReCoMER with synthetic inputs; this is not a benchmark run."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from recomer import ReCoMERConfig, ReCoMERContextModel, PriorRelativeCorrection, arithmetic_fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = ReCoMERConfig(input_dims=(8, 6, 4), d_model=24, num_heads=3,
                           ff_dim=48, num_layers=1, adapter_rank=4, dropout=0.)
    model = ReCoMERContextModel(config).to(device).eval()
    names = ("text", "audio", "vision")
    current = {name: torch.randn(4, dim, device=device) for name, dim in zip(names, config.input_dims)}
    history = {name: torch.randn(4, 3, dim, device=device) for name, dim in zip(names, config.input_dims)}
    mask = torch.ones(4, 3, dtype=torch.bool, device=device)
    history_mask = torch.ones(4, 3, 3, dtype=torch.bool, device=device)
    history_mask[0] = False
    tav = torch.randn(4, 7, device=device).softmax(-1)
    av = torch.randn(4, 7, device=device).softmax(-1)
    corrector = PriorRelativeCorrection(torch.full((7,), 1 / 7, device=device)).to(device).eval()
    with torch.no_grad():
        output = model(current, history, modality_mask=mask, history_mask=history_mask)
        corrected = corrector(tav, av)
        if isinstance(corrected, dict):
            corrected = corrected["probabilities"]
        elif isinstance(corrected, tuple):
            corrected = corrected[0]
        probabilities = arithmetic_fusion(output["probabilities"], corrected, alpha=.3)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(4, device=device), atol=1e-6)
    print(json.dumps({"example": "synthetic", "device": str(device),
                      "output_shape": list(probabilities.shape),
                      "probability_sums": probabilities.sum(-1).tolist(),
                      "empty_history_gate": output["admission_gate"][0].item()}, indent=2))


if __name__ == "__main__":
    main()
