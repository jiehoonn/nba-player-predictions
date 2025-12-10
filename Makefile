# Makefile for NBA Player Predictions
# Run the entire pipeline with simple commands

# Virtual environment settings
VENV_NAME := venv
VENV_DIR := $(VENV_NAME)
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest
BLACK := $(VENV_DIR)/bin/black
FLAKE8 := $(VENV_DIR)/bin/flake8
JUPYTER := $(VENV_DIR)/bin/jupyter
STREAMLIT := $(VENV_DIR)/bin/streamlit

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    PYTHON := $(VENV_DIR)/Scripts/python.exe
    PIP := $(VENV_DIR)/Scripts/pip.exe
    PYTEST := $(VENV_DIR)/Scripts/pytest.exe
    BLACK := $(VENV_DIR)/Scripts/black.exe
    FLAKE8 := $(VENV_DIR)/Scripts/flake8.exe
    JUPYTER := $(VENV_DIR)/Scripts/jupyter.exe
    STREAMLIT := $(VENV_DIR)/Scripts/streamlit.exe
endif

.PHONY: help venv install reinstall data features train evaluate visualize test clean all

# Default target
help:
	@echo "NBA Player Predictions - Available Commands:"
	@echo ""
	@echo "SETUP:"
	@echo "  make venv         Create virtual environment"
	@echo "  make install      Create venv + install dependencies (⭐ START HERE)"
	@echo "  make reinstall    Reinstall all dependencies"
	@echo ""
	@echo "PIPELINE (Python modules):"
	@echo "  make data         Collect NBA data from API (2-3 hours) ⚠️"
	@echo "  make features     Engineer features (src/feature_engineering.py)"
	@echo "  make train        Train models (src/train_models.py)"
	@echo "  make evaluate     Evaluate on test set (src/evaluate.py)"
	@echo "  make figures      Generate visualizations (src/generate_figures.py)"
	@echo ""
	@echo "PREDICTIONS:"
	@echo "  make predict                          Interactive prediction tool"
	@echo "  make predict PLAYER=\"Name\" OPP=TEAM   Quick prediction (command-line)"
	@echo "  make fantasy                          Fantasy lineup optimizer"
	@echo ""
	@echo "TESTING:"
	@echo "  make test         Run all tests (unit + integration)"
	@echo "  make test-quick   Run unit tests only (fast)"
	@echo "  make test-integration  Run integration tests (requires data)"
	@echo "  make lint         Run code quality checks"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make notebook     Launch Jupyter notebook (for exploration)"
	@echo "  make clean        Remove generated files"
	@echo "  make clean-all    Remove generated files + venv"
	@echo "  make all          Run full pipeline (features → train → evaluate → figures)"
	@echo "  make full         Run COMPLETE pipeline from scratch (data → all)"
	@echo ""
	@echo "QUICK START (from scratch - takes 3+ hours):"
	@echo "  1. make install      # Install dependencies (5 min)"
	@echo "  2. make data         # Collect NBA data (2-3 hours) ⚠️"
	@echo "  3. make all          # Run full pipeline (3 min)"
	@echo "  4. make test         # Run tests (2 min)"
	@echo ""
	@echo "QUICK START (with existing data - takes 5 min):"
	@echo "  1. make install      # Install dependencies (5 min)"
	@echo "  2. make all          # Run full pipeline (3 min)"
	@echo "  3. make test         # Run tests (2 min)"
	@echo ""
	@echo "To activate venv manually:"
	@echo "  source venv/bin/activate    (macOS/Linux)"
	@echo "  venv\\Scripts\\activate        (Windows)"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	else \
		python3 -m venv $(VENV_DIR); \
		echo "✓ Virtual environment created"; \
	fi

# Install dependencies (creates venv first if needed)
install: venv
	@echo "Installing dependencies in virtual environment..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Verifying SSL certificates..."
	@$(PYTHON) -c "import certifi; print(f'SSL cert path: {certifi.where()}')" || \
		(echo "WARNING: SSL certificate verification failed" && exit 1)
	@echo "✓ SSL certificates OK"
	@echo ""
	@echo "✓ Installation complete!"
	@echo ""
	@echo "To activate the virtual environment:"
	@echo "  source venv/bin/activate    (macOS/Linux)"
	@echo "  venv\\Scripts\\activate        (Windows)"

# Reinstall dependencies (useful after updating requirements.txt)
reinstall: venv
	@echo "Reinstalling dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	@echo ""
	@echo "Verifying SSL certificates..."
	@$(PYTHON) -c "import certifi; print(f'SSL cert path: {certifi.where()}')" || \
		(echo "WARNING: SSL certificate verification failed" && exit 1)
	@echo "✓ SSL certificates OK"
	@echo ""
	@echo "✓ Dependencies reinstalled!"

# Launch Jupyter notebook
notebook: venv
	@echo "Launching Jupyter Notebook..."
	@echo "Navigate to notebooks/01_data_collection.ipynb"
	@echo ""
	$(JUPYTER) notebook

# Data collection (via Python module)
data: venv
	@echo "Collecting NBA data..."
	@echo "⚠️  WARNING: This takes 2-3 HOURS due to NBA API rate limiting"
	@echo "Using Python module: src.data_collection"
	@echo ""
	$(PYTHON) -m src.data_collection
	@echo "✓ Data collection complete"

# Feature engineering (via Python module)
features: venv
	@echo "Engineering features..."
	@echo "Using Python module: src.feature_engineering"
	$(PYTHON) -m src.feature_engineering
	@echo "✓ Features created"

# Model training (baseline + advanced via Python module)
train: venv
	@echo "Training models (Ridge + XGBoost)..."
	@echo "Using Python module: src.train_models"
	$(PYTHON) -m src.train_models
	@echo "✓ Models trained"

# Evaluation (via Python module)
evaluate: venv
	@echo "Evaluating on test set..."
	@echo "Using Python module: src.evaluate"
	$(PYTHON) -m src.evaluate
	@echo "✓ Evaluation complete"

# Visualization (via Python module)
visualize: venv
	@echo "Generating comprehensive figures..."
	@echo "Using Python module: src.generate_figures"
	$(PYTHON) -m src.generate_figures
	@echo "✓ Check results/figures/ for 12 PNG files"

# Alias for visualize
figures: visualize

# Run all tests
test: venv
	@echo "Running test suite..."
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "✓ All tests complete"
	@echo "Coverage report: htmlcov/index.html"

# Run quick tests (skip slow integration tests)
test-quick: venv
	@echo "Running quick tests (unit tests only)..."
	$(PYTEST) tests/test_feature_engineering.py tests/test_training.py tests/test_evaluation.py -v
	@echo "✓ Quick tests complete"

# Run integration tests (require data/models)
test-integration: venv
	@echo "Running integration tests..."
	$(PYTEST) tests/test_pipeline.py tests/test_end_to_end.py -v
	@echo "✓ Integration tests complete"

# Lint code
lint: venv
	@echo "Running linters..."
	$(BLACK) --check src/ tests/
	$(FLAKE8) src/ tests/ --max-line-length=100
	@echo "✓ Linting complete"

# Format code
format: venv
	@echo "Formatting code..."
	$(BLACK) src/ tests/
	@echo "✓ Code formatted"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf data/cache/*
	rm -rf results/models/*
	rm -rf results/figures/*
	rm -rf results/*.json
	rm -rf results/*.parquet
	rm -rf results/*.csv
	rm -rf results/*.png
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f *.log
	@echo "✓ Cleaned all generated files (data, models, figures, results)"

# Clean everything including venv
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "✓ Everything cleaned"

# Run entire pipeline (Python modules)
all: features train evaluate figures
	@echo ""
	@echo "=========================================="
	@echo "✅ PIPELINE COMPLETE! 🎉"
	@echo "=========================================="
	@echo ""
	@echo "📊 Results:"
	@echo "  Models:  results/models/*.pkl (6 files)"
	@echo "  Metrics: results/*.json (3 files)"
	@echo "  Figures: results/figures/*.png (12 files)"
	@echo ""
	@echo "🔍 View results:"
	@echo "  - Open results/figures/*.png"
	@echo "  - Read README.md for full analysis"
	@echo "  - Run 'make predict' for interactive predictions"
	@echo "  - Run 'make test' to validate everything"
	@echo ""
	@echo "📝 Next steps:"
	@echo "  make predict     # Try predictions"
	@echo "  make test        # Run test suite"
	@echo ""

# Run COMPLETE pipeline from scratch (including data collection)
full: data all
	@echo ""
	@echo "=========================================="
	@echo "✅ COMPLETE PIPELINE FINISHED! 🎉"
	@echo "=========================================="
	@echo ""
	@echo "Everything built from scratch:"
	@echo "  ✓ Data collected from NBA API"
	@echo "  ✓ Features engineered"
	@echo "  ✓ Models trained"
	@echo "  ✓ Test evaluation complete"
	@echo "  ✓ Figures generated"
	@echo ""
	@echo "Next: make test"
	@echo ""

# Make predictions (interactive - enhanced version)
predict: venv
	@echo "🏀 Launching NBA Player Prediction Tool..."
	@echo ""
	@echo "Usage examples (command-line mode):"
	@echo "  make predict PLAYER=\"LeBron James\" OPP=BOS"
	@echo "  make predict PLAYER=\"Stephen Curry\" OPP=LAL AWAY=1"
	@echo "  make predict PLAYER=\"Giannis Antetokounmpo\" OPP=MIA REST=0"
	@echo ""
	@if [ -z "$(PLAYER)" ]; then \
		$(PYTHON) -m src.predict_enhanced; \
	else \
		if [ "$(AWAY)" = "1" ]; then \
			$(PYTHON) -m src.predict_enhanced --player "$(PLAYER)" --opponent "$(OPP)" --away --rest $(or $(REST),2); \
		else \
			$(PYTHON) -m src.predict_enhanced --player "$(PLAYER)" --opponent "$(OPP)" --rest $(or $(REST),2); \
		fi \
	fi

# Fantasy basketball lineup optimizer
fantasy: venv
	@echo "🏆 Launching Fantasy Basketball Lineup Optimizer..."
	@echo ""
	$(PYTHON) examples/fantasy_lineup_optimizer.py

# Launch interactive dashboard (optional)
app: venv
	@echo "Launching dashboard at http://localhost:8501"
	@if [ -f "app.py" ]; then \
		$(STREAMLIT) run app.py; \
	else \
		echo "⚠️  app.py not found. Streamlit dashboard not implemented yet."; \
	fi

# CI/CD target (used by GitHub Actions)
ci: install lint test-quick
	@echo "✓ CI checks passed"
