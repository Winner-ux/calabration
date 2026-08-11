# Fusion ResNet project guidance

- The project is an in-progress PyTorch image-fusion implementation. Preserve tensor-shape contracts across encoders, attention, fusion blocks, decoder, and loss functions.
- Use the `model` package as the module boundary. When touching imports, prefer package-relative imports and validate with module-style execution; do not maintain competing import conventions.
- The environment is not yet reproducibly declared. Do not invent dependency or CUDA versions. If environment setup is requested, derive requirements from actual imports and record the tested Python, PyTorch, and CUDA combination.
- Treat datasets, checkpoints, and generated images as user data or outputs. Do not delete, overwrite, rename, or regenerate them unless explicitly requested.
- Architecture or loss changes must document expected input/output shapes and preserve checkpoint compatibility when possible. If compatibility breaks, state it explicitly.
- Validate in increasing scope: syntax/import checks, a tiny CPU synthetic forward pass, targeted loss checks, then training only when the task requires it and the environment/data are available.
- Keep `develop_list/` as planning context, not executable truth. Confirm an item against the current code before acting, and update it only when the task materially changes that plan.
