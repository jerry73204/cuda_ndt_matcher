# CUDA NDT Matcher - Build System

# Build all packages
build:
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --cargo-args --release

# Clean build artifacts
clean:
    rm -rf build install log target

# Format code
format:
    cargo +nightly fmt

# Lint code
lint:
    cargo +nightly fmt --check
    cargo clippy --all-targets

# Run tests
test:
    cargo test --all-targets

# Run all quality checks (format, lint, test)
quality: lint test
