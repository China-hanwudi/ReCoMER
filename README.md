# ReCoMER

**Reliability and Context Modeling to Mitigate Biases in Multimodal Conversational Emotion Recognition**

[Paper](paper/ReCoMER.pdf) · [中文说明](README.zh-CN.md) · [Method & data interface](paper/IMPLEMENTATION.md) · [Reported results](paper/results.json)

ReCoMER coordinates **which history to admit**, **how strongly it updates current evidence**, and **how modality contributions are estimated after that update**. A complementary audio-visual prediction path supplies prior-relative nonverbal evidence at the decision level.

![ReCoMER architecture](paper/figures/overview.png)

## Paper-reported performance

| Dataset | WF1 (%) | ACC (%) | Summary |
|---|---:|---:|---|
| M3ED | **62.07** | — | +7.02% over the listed MF-EDM result |
| IEMOCAP | **78.53** | — | Three-run mean; sample SD 0.04% |
| EmotionTalk | **72.03** | **72.52** | ACC +1.29% over the listed LMF result |

Performance gains denote differences between percentage-scale scores. The results follow the evaluation protocols in the linked manuscript; the IEMOCAP figure is a mean, while M3ED and EmotionTalk report individual runs. EmotionTalk WF1 is labeled separately from the literature's source F1 metric.

## Method highlights

- **NCHA — conditional history admission.** A preliminary history reading informs the second round of null-candidate competition and a shared admission decision.
- **FCCR — bounded historical feedback.** Modality-specific residual adapters refine current representations within a relative norm budget.
- **SABER — contribution routing after updating.** Fixed-state Shapley supervision connects updated evidence to bounded modality weights. Inference uses a single scorer pass.
- **cRBEF — prior-relative nonverbal correction.** A separately trained AV expert supplies class-wise evidence, an utterance-level gate controls correction strength, and arithmetic fusion combines complete predictions.

The manuscript's selected matched component gains are **NCHA +2.913%**, **FCCR +3.290%**, **SABER +1.902%**, and **cRBEF +7.478%**. Each describes its own matched comparison; these effects are not added together. All 18 mechanism results and their interpretation are available in [the result record](paper/results.json) and the paper.

## Quick start

```bash
git clone https://github.com/China-hanwudi/ReCoMER.git
cd ReCoMER
python -m pip install -e ".[dev]"
python scripts/demo_recomer.py
python -m pytest tests/recomer -q
python scripts/report_paper_results.py
```

The demo uses synthetic feature tensors to exercise the full prediction composition on CPU. The Python package is a readable reference implementation of the manuscript's core mechanisms. The performance table above records the manuscript experiments, rather than a score produced by the synthetic demo.

```python
from recomer import ReCoMERConfig, ReCoMERContextModel

config = ReCoMERConfig(input_dims=(768, 1024, 342), num_classes=7)
model = ReCoMERContextModel(config)
output = model(current, history, modality_mask=present, history_mask=valid_history)
probabilities = output["probabilities"]
```

Feature dimensions should match your own packed features. See [the input schema](paper/IMPLEMENTATION.md) for masks, speaker relations, training objectives, frozen prediction composition, and dataset presets.

## Repository layout

```text
recomer/                    PyTorch modules and prediction composition
tests/recomer/              CPU mechanism tests
scripts/demo_recomer.py     Synthetic forward and correction example
scripts/report_paper_results.py
configs/dataset_presets.json
paper/ReCoMER.pdf           Manuscript supplied for this release
paper/figures/overview.png  Original architecture figure extracted from the paper
paper/results.json         Paper-reported aggregate results
paper/IMPLEMENTATION.md     Interfaces and evaluation protocols
```

The existing TemporalN3 and `ReCoMER_paper_20261006/` directories remain available as earlier research versions. The current ReCoMER entry point is the package and manuscript linked above. Repository history and existing data remain intact.

## Efficiency

The manuscript reports **13.121M parameters** and **1.401 ms per utterance** on an RTX 5090. NCHA, FCCR, and SABER jointly add **0.086M parameters (0.66%)**. Timing uses the median of five post-warmup runs on eight development dialogues (187 utterances), including expert computation and excluding feature extraction, transfers, and model loading.

## Citation and license

The manuscript title and BibTeX entry are in [CITATION.bib](CITATION.bib). Repository-owned code is distributed under the existing [MIT license](LICENSE). Obtain datasets and pretrained feature extractors from their original providers under their own terms.
