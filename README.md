# sae_jailbreak_unlearning
 Investigating how well intervening on Sparse Autoencoder internals prevents adversaries from accessing dangerous knowledge.

This repo studies unlearning and jailbreak-robustness on the [WMDP benchmark](https://www.wmdp.ai/)
using Gemma-2-2B: it applies RMU-based unlearning (`src/scripts/wmdp/`) and Sparse
Autoencoder interventions/steering (`src/scripts/unlearning_with_saes/`,
`src/scripts/llm-auditing/`) to reduce a model's hazardous (bio/cyber) knowledge, evaluates
the result with a forked `lm-evaluation-harness` (`src/scripts/evaluation/`), and probes
how well the unlearning holds up against adversarial attacks (GCG / Dynamic Suffix Search)
that try to recover the "forgotten" knowledge.

Folder structure based on the one described in [this website](https://dev.to/luxacademy/generic-folder-structure-for-your-machine-learning-projects-4coe#:~:text=You%20can%20organize%20it%20with,using%20trained%20models%20for%20predictions.)

For exact environment setup, data acquisition (including the gated WMDP bio-forget
corpus), how to run the main scripts, external dependencies, and hardware requirements,
see **[REPRODUCE.md](./REPRODUCE.md)**.
