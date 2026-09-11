# Project Agent Guidelines

## Clean-Room Testing Subagent Protocol
When acting as or spawning a test generation subagent:
- **FORBIDDEN FILE**: `src/tensor.rs` must NEVER be viewed, read, searched, or referenced by the tester agent.
- **INTERFACE SPECIFICATION**: The ONLY allowable source of truth for `Tensor` API signatures and autograd behavior is `tests/TENSOR_API.rs`.
- **OBJECTIVE**: The tester agent's sole task is writing black-box unit tests in `tests/test_gradients.rs` using analytical checks and finite-difference numerical gradient checking.
