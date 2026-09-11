# Project Agent Guidelines

## Clean-Room Testing Subagent Protocol
When acting as or spawning a test generation subagent:
- **STRICTLY FORBIDDEN FILES**: The entire `src/` directory (including `src/losses/*`, `src/tensor.rs`, `src/modules/*`, `src/lib.rs`, `src/main.rs`) must NEVER be viewed, read, searched, or referenced by the tester agent.
- **INTERFACE SPECIFICATIONS**: The ONLY allowable sources of truth for API signatures, mathematical semantics, and autograd behaviors are:
  - `tests/TENSOR_API.md` for `Tensor` APIs.
  - `tests/LOSSES_API.md` for `Loss` functions (`MSELoss`, `NLLLoss`, `CrossEntropyLoss`).
- **OBJECTIVE**: The tester agent's sole task is writing black-box unit tests in `tests/test_gradients.rs` and `tests/test_losses.rs` using analytical checks, finite-difference numerical gradient checking, edge-case testing, and mathematical invariance verifications.
