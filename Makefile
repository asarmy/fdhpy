# Makefile

# Variables
PACKAGE_NAME = fdhpy

# Default target
.PHONY: all
all: help

# Show help message
.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  make help          - Show this help message"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make check         - Run pre-commit hooks on all files"
	@echo "  make build         - Build the distribution packages"
	@echo "  make release       - Full release: clean, check, and build"

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -exec rm -rf {} +

# Run pre-commit hooks on all files
.PHONY: check
check:
	pre-commit run --all-files

# Build the distribution packages using setuptools
.PHONY: build
build:
	python -m build

# Full release process: clean, check, and build
.PHONY: release
release: clean check build
