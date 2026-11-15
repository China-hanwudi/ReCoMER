# Method implementation and data interface

The PyTorch package exposes HMBR's contextual predictor and prediction correction. The [manuscript](ReCoMER.pdf) describes the model and the [result record](results.json) lists the scores in Tables 1–8. The `recomer` package retains the original public API names.

## Inputs

`ReCoMERContextModel.forward` receives precomputed utterance features:

- `current`: a dictionary with `text`, `audio`, and `vision` tensors shaped `[B, D_m]`.
- `history`: the same keys with tensors `[B, H, D_m]`, or `None` for no history.
- `modality_mask`: a Boolean `[B, 3]` tensor in text/audio/vision order.
- `history_mask`: a Boolean `[B, H, 3]` tensor; invalid or missing history is masked.
- `relative_positions`: optional `[B, H]` positive distances to completed preceding utterances. Zero, future/nonpositive, and out-of-window positions are excluded.
- `speaker_relations`: optional binary `[B, H]` relation indicators.

Keep utterance identifiers and class order aligned across contextual, TAV, and AV prediction banks. Build history separately within each conversation, using only completed earlier turns. The [dataset presets](../configs/dataset_presets.json) specify the contextual path for each benchmark. Set the three input dimensions to match the features you supply.

## Forward path

NCHA computes a preliminary reading, predicts admission, conditions the second null-candidate reading, and produces a shared gate. The null value vector is zero. FCCR applies modality-specific learned feedback within a relative norm limit. SABER scores the updated representations and produces bounded routing weights.

The output dictionary includes `logits`, `probabilities`, `admission_probability`, `admission_gate`, `history_summary`, `updated_tokens`, `routing_weights`, `contribution_scores`, and a `FrozenContext` under `state`. Inference requires no labels, teacher predictions, or subset enumeration.

For M3ED, history enters through current-token feedback and routing acts at token entry and output pooling. The IEMOCAP and EmotionTalk paths additionally admit historical tokens and use masked mean pooling over current outputs.

## Training interfaces

`model.compute_loss(output, labels, admission_targets=...)` exposes task, fixed-state Shapley, proper-subset, and admission losses. `utility_scale` supports an external warmup/ramp schedule. Admission targets are prepared from out-of-fold training teachers: `sigmoid((CE_off - CE_on) / tau_u)`. They are training-only inputs. Select `admission_mode="available"`, `"soft"`, and `"hard"` to implement the stated curriculum.

`model.unimodal_loss(output, labels)` supervises auxiliary modality heads on detached representations. `coalition_logits(state)` and `shapley_targets(state, labels)` reuse the same contextual state and detached routing. They are explicit training/analysis calls and are not invoked by ordinary inference.

## Prediction composition

`calibrate_context` implements the frozen logit composition from a supplied residual, centered class bias, and positive temperature. These tensors/parameters come from the corresponding fitted prediction mapping.

`PriorRelativeCorrection` consumes independent TAV and text-free AV probabilities and a training class prior. Its six-statistic gate scales the prior-relative AV log evidence. `arithmetic_fusion` then combines the corrected and contextual posteriors with the fixed `alpha` for that configuration. The sample gate `rho` and outer mixture `alpha` have different roles.

The reference implementation also provides `emotiontalk_output` for the earlier EmotionTalk configuration. It accepts the fused probabilities and an independent TAV-head posterior, with default temperature 3 and mixing coefficient 0.85. The [prior implementation notes](https://github.com/China-hanwudi/ReCoMER/blob/0d262831d7bb78bdea2307a2c7eefda89470c51c/paper/IMPLEMENTATION.md) document these settings.

Training features, trained contextual/TAV/AV weights, frozen mappings, and fitted gates should be supplied as their own aligned experiment artifacts. The synthetic demo shows the interfaces with generated tensors; it does not substitute for those fitted artifacts.

## Reported experimental setup

All three datasets use up to three utterances that ended before the target began. The contextual predictor uses a 192-dimensional representation, two Transformer layers with six heads each, rank-32 feedback adapters, and a relative-norm budget of 0.1. The optimizer is AdamW with batch size 128 and weight decay 0.01.

| Dataset | Evaluation split | Features and training settings reported in Section 4.1 | HMBR WF1 |
| --- | --- | --- | --- |
| M3ED | Official TV-series split | Qwen text (2,560), HuBERT audio (2,048), visual features (342). Up to 45 epochs, learning rate 0.00015, five warmup epochs, dropout 0.25. | 62.07% (one run) |
| IEMOCAP | Session 5 held out, six classes with happy and excited kept separate | Up to 35 epochs, learning rate 0.0001, two warmup epochs, dropout 0.15. | 78.53% (mean of three runs, SD 0.04 percentage points) |
| EmotionTalk | Seven-class multimodal labels, G01/G12 for validation and G03/G15 for testing | Frozen Qwen3-8B-AWQ text (4,096), Chinese HuBERT-base audio (768), CLIP ViT-L/14 vision (768). | 72.03% (one run) |

WF1 is F1 weighted by class support. Table 4 gives module comparisons on M3ED. Tables 5–8 give mechanism comparisons within each module. Gains are absolute WF1 differences in percentage points. Section 4.3 reports the maximum observed gain for the three contextual modules and for each individual mechanism.

The [prior experiment record](https://github.com/China-hanwudi/ReCoMER/blob/0d262831d7bb78bdea2307a2c7eefda89470c51c/paper/IMPLEMENTATION.md) documents test-informed configuration exploration for the M3ED and EmotionTalk runs that reported the same scores. The [archived measurements](https://github.com/China-hanwudi/ReCoMER/blob/0d262831d7bb78bdea2307a2c7eefda89470c51c/paper/results.json) include the earlier accuracy, timing and full-configuration removal results.

The MIT license applies to repository-owned code. Download datasets, source features, and pretrained extractors from their original providers. Keep their licenses, class ordering, splits, and artifact provenance with each run.
