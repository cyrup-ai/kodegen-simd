# kodegen_simd

[![Crates.io](https://img.shields.io/crates/v/kodegen_simd.svg)](https://crates.io/crates/kodegen_simd)
[![Documentation](https://docs.rs/kodegen_simd/badge.svg)](https://docs.rs/kodegen_simd)
[![License](https://img.shields.io/badge/license-Apache%202.0%20OR%20MIT-blue.svg)](LICENSE)

Ultra-high-performance SIMD-accelerated operations for AI/ML workloads in Rust. Part of the [KODEGEN.ᴀɪ](https://kodegen.ai) ecosystem.

## Features

- **🚀 Automatic SIMD Optimization**: Runtime CPU feature detection with zero-overhead dispatch
  - x86_64: AVX-512, AVX2, SSE4.1 support
  - ARM64: NEON support
  - Automatic fallback to optimized scalar implementations

- **🎯 Vector Similarity Operations**: High-performance cosine similarity with intelligent implementation selection

- **🔥 Logits Processing**: Complete pipeline for LLM inference
  - Temperature scaling with SIMD acceleration
  - Top-k and nucleus (top-p) sampling
  - Repetition, frequency, and presence penalties
  - Numerically stable softmax and argmax operations

- **📐 Structured Generation**: Type-safe constrained output
  - JSON syntax validation
  - JSON schema-based constraints
  - Generate from Rust types with `#[derive(JsonSchema)]`
  - Predefined constraint presets

- **⚡ Zero-Allocation Hot Paths**: Stack-based buffers for inference-critical operations

- **🔧 Hardware Acceleration**: Optional backends for neural network operations
  - CUDA / cuDNN support
  - Metal (Apple Silicon) support
  - Intel MKL and Apple Accelerate framework support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kodegen_simd = "0.1"
```

### Optional Hardware Acceleration

For CUDA support:
```toml
[dependencies]
kodegen_simd = { version = "0.1", features = ["cuda"] }
```

For Metal (Apple Silicon):
```toml
[dependencies]
kodegen_simd = { version = "0.1", features = ["metal"] }
```

For Intel MKL:
```toml
[dependencies]
kodegen_simd = { version = "0.1", features = ["mkl"] }
```

For Apple Accelerate:
```toml
[dependencies]
kodegen_simd = { version = "0.1", features = ["accelerate"] }
```

## Quick Start

### Vector Similarity

```rust
use kodegen_simd::similarity::cosine_similarity;

let embedding_a = vec![1.0, 2.0, 3.0, 4.0];
let embedding_b = vec![4.0, 3.0, 2.0, 1.0];

// Automatically uses best available SIMD implementation
let similarity = cosine_similarity(&embedding_a, &embedding_b);
println!("Cosine similarity: {}", similarity);
```

### Logits Processing

```rust
use kodegen_simd::{scale_temperature, softmax, argmax};

let mut logits = vec![2.0, 1.0, 0.1, 3.0];

// Apply temperature scaling
scale_temperature(&mut logits, 0.7)?;

// Compute softmax probabilities
let probabilities = softmax(&logits)?;

// Find most likely token
let best_token = argmax(&probabilities)?;
```

### Processing Context for Generation

```rust
use kodegen_simd::context::ProcessingContext;
use kodegen_simd::logits::process_logits_scalar;
use kodegen_simd::config::ProcessorConfig;

let mut context = ProcessingContext::new()
    .with_temperature(0.8)
    .with_top_k(Some(50))
    .with_top_p(Some(0.95));

let mut logits = vec![2.0, 1.0, 0.1, 3.0, 1.5];
let config = ProcessorConfig::default();

// Apply temperature, top-k, and top-p filtering
process_logits_scalar(&mut logits, &context, &config)?;

// Track token history
context.extend_history(&[best_token]);
```

### Structured Generation with Type Safety

```rust
use kodegen_simd::serde_constraints::constraint_for_type;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;

#[derive(Serialize, Deserialize, JsonSchema)]
struct User {
    name: String,
    age: u32,
    email: Option<String>,
    is_active: bool,
}

// Create constraint from Rust type
let constraint = constraint_for_type::<User>(&tokenizer)?;

// Use in ProcessingContext
let mut context = ProcessingContext::new()
    .with_schema_constraint(constraint);

// During generation, check if tokens are valid
for token in candidate_tokens {
    if context.is_token_valid_schema(token)? {
        // Token produces valid JSON matching User schema
        context.update_schema_constraint_state(token)?;
    }
}
```

### JSON Schema Constraints

```rust
use kodegen_simd::serde_constraints::constraint_for_schema;

let schema = r#"{
    "type": "object",
    "properties": {
        "name": { "type": "string" },
        "age": { "type": "integer", "minimum": 0 },
        "tags": { 
            "type": "array",
            "items": { "type": "string" }
        }
    },
    "required": ["name", "age"]
}"#;

let constraint = constraint_for_schema(schema, &tokenizer)?;
```

### Predefined Constraint Presets

```rust
use kodegen_simd::serde_constraints::presets;

// Array of strings
let constraint = presets::array_of_strings(&tokenizer)?;

// Array of integers
let constraint = presets::array_of_integers(&tokenizer)?;

// Generic object with string keys
let constraint = presets::object_with_string_keys(&tokenizer)?;
```

## Performance

The library uses runtime CPU feature detection to automatically select the fastest implementation:

- **Cosine Similarity**: Up to 8x faster than naive implementations on AVX2
- **Temperature Scaling**: SIMD vectorization across entire logits tensor
- **Softmax**: Numerically stable with minimal allocation overhead
- **Constraint Validation**: Efficient state machine with token lookahead

Run benchmarks:
```bash
cargo bench --features bench
```

## Architecture

### SIMD Abstraction

Operations are implemented multiple times for different CPU features:
- AVX-512 (16-wide vectors)
- AVX2 (8-wide vectors)  
- SSE4.1 (4-wide vectors)
- NEON (4-wide vectors, ARM)
- Scalar fallback

The best implementation is selected once at startup and cached.

### Constraint System

The constrained generation system uses state machines to track valid token transitions:

1. **JSON Constraints**: Validates JSON syntax during generation
2. **Schema Constraints**: Enforces JSON schema structure (types, required fields, ranges)
3. **Type Constraints**: Generated from Rust `serde` types with `JsonSchema` derive

Constraints can force deterministic token sequences when only one valid path exists, improving generation efficiency.

## Requirements

- **Rust**: Nightly toolchain (specified in `rust-toolchain.toml`)
- **Edition**: 2024

## Documentation

- [API Documentation](https://docs.rs/kodegen_simd)
- [CLAUDE.md](CLAUDE.md) - Architecture and development guide
- [Examples](examples/) - Additional usage examples

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

Part of the [KODEGEN.ᴀɪ](https://kodegen.ai) ecosystem for AI-powered database tools and MCP servers.

