# Method implementation and data interface

This release pairs the manuscript with an inspectable PyTorch reference implementation of its core mechanisms. `paper/results.json` contains the manuscript's reported measurements. Model defaults are usable reference values; they are not a declaration of every hidden setting of an archived training run.

## Inputs

`ReCoMERContextModel.forward` receives precomputed utterance features:

- `current`: a dictionary with `text`, `audio`, and `vision` tensors shaped `[B, D_m]`.
- `history`: the same keys with tensors `[B, H, D_m]`, or `None` for no history.
- `modality_mask`: a Boolean `[B, 3]` tensor in text/audio/vision order.
- `history_mask`: a Boolean `[B, H, 3]` tensor; invalid or missing history is masked.
- `relative_positions`: optional `[B, H]` positive distances to completed preceding utterances. Zero, future/nonpositive, and out-of-window positions are excluded.
- `speaker_relations`: optional binary `[B, H]` relation indicators.

Keep utterance identifiers and class order aligned across contextual, TAV, and AV prediction banks. Build history separately within each conversation, using only completed earlier turns. The dataset-specific mechanism paths appear in `configs/dataset_presets.json`; set feature dimensions from your own data.

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

`emotiontalk_output` additionally implements the manuscript's final temperature transformation and TAV-head combination. Pass the corresponding independent head posterior, using the reported temperature 3 and mixing coefficient 0.85.

Training features, trained contextual/TAV/AV weights, frozen mappings, and fitted gates should be supplied as their own aligned experiment artifacts. The synthetic demo shows the interfaces with generated tensors; it does not substitute for those fitted artifacts.

## Evaluation protocols

- **M3ED:** official television-source split; reported WF1 62.07.
- **IEMOCAP:** Session 5 held out, happy and excited retained separately; reported WF1 78.53 is a three-run mean with sample SD 0.04.
- **EmotionTalk:** G01/G12 for development and G03/G15 for testing; reported ACC 72.52 and WF1 72.03.

The manuscript records test-informed configuration exploration for M3ED and EmotionTalk. The reported mechanism rows are selected matched effects under row-specific conditions. For new model development, fit on training/development partitions and keep the final evaluation distinct.

The MIT license applies to repository-owned code. Download datasets, source features, and pretrained extractors from their original providers. Keep their licenses, class ordering, splits, and artifact provenance with each run.
