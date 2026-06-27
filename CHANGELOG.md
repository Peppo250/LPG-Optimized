# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-06-27

### Added
- Created **[requirements.txt](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/requirements.txt)** for explicit dependency tracking and environment replication.
- Created robust **[.gitignore](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/.gitignore)** to prevent committing cache directories (`__pycache__`, `.pytest_cache`), virtual environments (`.venv`), secrets (`.env`), IDE settings, and logs.
- Created **[.editorconfig](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/.editorconfig)** to enforce uniform code styling across development editors.
- Created a **[tests/](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/tests/)** directory containing unit tests for Layer 1/2/3 solvers (**[test_optimization.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/tests/test_optimization.py)**) and integration tests for web endpoints (**[test_api.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/tests/test_api.py)**).
- Created **[run_project.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/run_project.py)** CLI runner script to automate pipeline runs, training, unit testing, and server hosting.
- Created a **[Dockerfile](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/Dockerfile)** and **[docker-compose.yml](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/docker-compose.yml)** to support fully containerized FastAPI deployments.
- Created a GitHub Actions workflow **[test.yml](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/.github/workflows/test.yml)** to run automated CI tests on branch pushes.
- Created **[ARCHITECTURE.md](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/docs/ARCHITECTURE.md)** design guide to document optimization formulas and dataset properties.
- Added live dashboard screenshots under **[docs/assets/](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/docs/assets/)**.

### Changed
- Modularized the codebase into a structured package folder **[lpg_catering/](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/lpg_catering/)** containing separate pipeline and optimization solver packages.
- Refactored **[optimization_engine.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/optimization_engine.py)** into a clean, backward-compatible importing wrapper.
- Decoupled styles and scripting from **[dashboard.html](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/dashboard.html)** into **[style.css](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/css/style.css)** and **[app.js](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/js/app.js)**.
- Enhanced REST API **[api.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/api.py)** server to dynamically parse CORS origins using environment variables.

### Fixed
- Fixed dataset generation stockout rate logic in **[data_pipeline.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/data_pipeline.py)** to output the target ~26.5% rate directly without manual patch overrides.
- Fixed console output encoding issues in **[data_pipeline.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/data_pipeline.py)** and **[train_final.py](file:///C:/Users/Porchezhian/Documents/GitHub/LPG-Optimized/train_final.py)** to ensure crash-free execution on Windows terminals.
- Fixed FastAPI boot crash when model binaries are missing by enabling fallback predictions.

## [2.0.0] - 2024-04-12
- Initial development release of the LPG Catering Intelligence system with Scipy optimization solvers and the baseline FastAPI REST API.
