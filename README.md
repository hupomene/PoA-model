# Probability of Approval — Historical ML API Project

An earlier machine-learning inference application built with Python and Flask. It exposes a probability-of-approval prediction through a simple browser form and JSON endpoint.

## Repository contents

- `main.py` — Flask application, feature preparation, and model inference
- `PoA_v5.0.pkl` — serialized model artifact
- `requirements.txt` — original project dependencies

## Model and API overview

The application accepts loan-related numeric and categorical features, constructs a feature vector, and calls the serialized model's `predict_proba` method. The JSON response contains a `PoA` probability.

This repository is a historical portfolio example, **not a currently validated lending decision system**. No current calibration, fairness, model accuracy, or production uptime claim is made here.

## Security and reproducibility

The default branch preserves the original historical implementation. A separate [hardening pull request](https://github.com/hupomene/PoA-model/pull/1) addresses Flask development debugger exposure and basic input validation; those changes are **not merged** at the time of this README update.

- Never expose the Flask development server or debugger to the public internet.
- Serialized Python pickle files should only be loaded from trusted, verified sources.
- Verify package compatibility in an isolated environment before loading the model.
- Do not submit real customer financial or personal information to an unverified demo.

## Suggested future validation

Before using this model in any real lending workflow, establish a documented test dataset, prediction reproducibility, calibration, subgroup evaluation, access controls, audit logging, and an appropriate human decision process.

## Related projects

- [CIVIX — Evidence-grounded AI engineering portfolio](https://github.com/hupomene/civix-portfolio)
- [Plant Seedlings Computer Vision case study](https://github.com/hupomene/Computer-Vision-Project-Plant-Seedlings-Image-Classification)
