# Contributing to Wafer Defect Detection Project

Thank you for your interest in contributing to this wafer defect detection project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/wafer-defect-detection.git
   cd wafer-defect-detection
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

### Code Style

- Follow PEP 8 conventions
- Use type hints where possible
- Document functions with docstrings
- Maximum line length: 100 characters

**Format your code:**
```bash
black model_large/ model_small/
flake8 model_large/ model_small/
```

### Testing

Before submitting changes, test locally:

```bash
# Test specific training module
python model_large/train_both.py --epochs 2  # Quick test run

# Test inference
python model_large/predict.py --image test_image.png

# Run unit tests (if available)
pytest tests/
```

### Git Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, focused commits:
   ```bash
   git commit -m "Brief description of changes"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request** on GitHub with:
   - Clear title describing the feature/fix
   - Detailed description of changes
   - Any relevant issue numbers
   - Test results or performance impact

## Types of Contributions

### Bug Reports
- Describe the issue clearly
- Include steps to reproduce
- Provide Python version, OS, and PyTorch version
- Attach error messages/logs

### Feature Requests
- Explain the use case
- Describe expected behavior
- Provide examples if applicable

### Code Improvements
- Performance optimizations
- Code refactoring
- Better error handling
- Documentation improvements

### Documentation
- README enhancements
- Code comment clarification
- Tutorial/example additions

## Project Structure Guidelines

When adding new features:

**For model_large and model_small:**
- Keep both implementations synchronized
- Document differences in respective READMEs
- Maintain consistent interfaces

**For utilities:**
- Place shared code in root-level utilities
- Document dependencies between models

## Pull Request Review Process

- Maintainers will review your PR
- Address feedback in follow-up commits
- Keep discussions professional and constructive
- Once approved, PR will be merged

## Code of Conduct

- Be respectful to all contributors
- Focus discussions on technical merit
- Avoid discrimination or harassment
- Report issues to project maintainers

## Questions or Issues?

- Check existing issues/discussions
- Review README files for detailed documentation
- Comment on relevant issues with questions

Thank you for contributing! 🚀
