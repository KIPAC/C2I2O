# c2i2o TODO List

**Last Updated:** 2024-12-14
**Current Version:** 0.1.0 (Ready for Release)

---

## Immediate (Pre-Release)

### v0.1.0 Release ✅
- ✅ All core features implemented
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Examples working
- ✅ Release automation configured
- ✅ CHANGELOG.md updated
- ✅ Ready for publication

**Status:** READY TO RELEASE

---

## High Priority (v0.2.0)

### Emulator Improvements
- [ ] **Improve DummyEmulator bounds checking**
  - Implement `_training_bounds` storage
  - Add `_check_bounds()` method with tolerance
  - Provide informative error messages
  - Save/load bounds with emulator
  - Add tests for bounds checking
  - See: Context from v0.1.0 development

- [ ] **Implement Gaussian Process emulator**
  - Use scikit-learn GaussianProcessRegressor
  - Support multiple kernels
  - Uncertainty quantification
  - Hyperparameter optimization

- [ ] **Implement neural network emulator (PyTorch)**
  - Multi-layer perceptron architecture
  - Training with validation
  - GPU support
  - Model checkpointing

- [ ] **Add emulator validation metrics**
  - Cross-validation utilities
  - Accuracy metrics (MSE, MAE, R²)
  - Visualization tools

- [ ] **Benchmark emulator performance**
  - Speed comparisons
  - Accuracy comparisons
  - Memory usage analysis

### Interface Modules
- [ ] **Implement CCL interface**
  - Power spectrum calculations
  - Correlation function calculations
  - Distance calculations
  - Growth factor calculations

- [ ] **Implement PyTorch interface**
  - Neural network emulator base class
  - Training utilities
  - GPU support

- [ ] **Implement Astropy interface**
  - Cosmology objects conversion
  - Units handling
  - Coordinates integration

### Inference Methods
- [ ] **Implement emcee wrapper**
  - Parallel tempering support
  - Convergence diagnostics
  - Auto-correlation analysis

- [ ] **Add convergence diagnostics**
  - Gelman-Rubin statistic
  - Effective sample size
  - Autocorrelation time

- [ ] **Support parallel chains**
  - Multi-chain MCMC
  - Chain comparison
  - Combined posteriors

---

## Medium Priority (v0.3.0)

### Features
- [ ] **Add model comparison tools**
  - Bayesian evidence calculation
  - Information criteria (AIC, BIC)
  - Bayes factors

- [ ] **Implement evidence calculation**
  - Nested sampling integration
  - Bridge sampling
  - Thermodynamic integration

- [ ] **Add cross-validation utilities**
  - K-fold cross-validation
  - Leave-one-out CV
  - Stratified sampling

- [ ] **Create pipeline/workflow system**
  - End-to-end analysis pipelines
  - Configuration files
  - Reproducible workflows

- [ ] **Add catalog/dataset management**
  - Data loading utilities
  - Format conversions
  - Catalog validation

### Interface Modules (Continued)
- [ ] **Implement TensorFlow interface**
  - Neural network emulators
  - Training utilities
  - SavedModel format

- [ ] **Implement SQLAlchemy interface**
  - Database models
  - Query utilities
  - Result storage

### Command-Line Interface
- [ ] **Create CLI tools**
- Parameter validation command
  - Emulator training command
  - Inference running command
  - Result analysis command

---

## Low Priority (Future Versions)

### Examples and Documentation
- [ ] **Create Jupyter notebook versions of examples**
  - Interactive parameter exploration
  - Live visualizations
  - Educational content

- [ ] **Add video tutorials**
  - YouTube series
  - Screen recordings
  - Walkthrough guides

- [ ] **Create quickstart guide**
  - 5-minute introduction
  - Common use cases
  - Troubleshooting tips

- [ ] **Add example with real data**
  - SDSS galaxy clustering
  - DES weak lensing
  - Planck CMB data

- [ ] **Create gallery of visualizations**
  - Corner plots
  - Trace plots
  - Posterior distributions

### Performance
- [ ] **Profile code for bottlenecks**
  - Identify hot spots
  - Optimization opportunities
  - Memory profiling

- [ ] **Optimize hot paths**
  - Vectorization improvements
  - Numba/Cython for critical sections
  - Memory optimization

- [ ] **Add parallel processing support**
  - Multi-core emulator training
  - Parallel MCMC chains
  - Distributed computing

- [ ] **Consider GPU acceleration**
  - CUDA support for emulators
  - GPU-based likelihood calculations
  - Batch processing on GPU

### Community
- [ ] **Create CODE_OF_CONDUCT.md** ✅ (CONTRIBUTING.md exists)
- [ ] **Set up GitHub Discussions**
  - Q&A section
  - Ideas and feedback
  - Show and tell

- [ ] **Add citation information**
  - CITATION.cff file
  - Zenodo DOI
  - Publication (when available)

- [ ] **Create tutorial workshops**
  - Conference tutorials
  - Online workshops
  - Training materials

### Web Application
- [ ] **Design web interface**
  - Parameter input forms
  - Interactive visualizations
  - Result downloads

- [ ] **Implement backend API**
  - FastAPI or Flask
  - RESTful endpoints
  - Authentication

- [ ] **Create frontend**
  - React or Vue.js
  - Interactive plots
  - Responsive design

---

## Completed ✅

### v0.1.0 (2024-12-14)
- ✅ Initial project structure
- ✅ Core module implementation
  - CosmologicalParameters with validation
  - ParameterSpace with priors
  - Intermediate data products
  - Observable definitions
  - Base emulator framework
  - Base inference framework

- ✅ Emulator implementations
  - DummyEmulator with linear interpolation
  - Bounds checking
  - Save/load functionality

- ✅ Inference implementations
  - MetropolisHastings MCMC sampler
  - InferenceResult analysis tools

- ✅ Utilities
  - Validation functions
  - Array helpers

- ✅ Testing infrastructure
  - 100+ unit tests
  - Comprehensive test coverage (~90%)
  - All tests passing
  - Fixtures and test utilities

- ✅ Example scripts
  - 5 complete examples
  - Progressive complexity
  - All running successfully
  - Visualization support

- ✅ Documentation
  - Sphinx setup with ReadTheDocs theme
  - Installation guide
  - Quick start guide
  - Tutorials (basic + advanced)
  - Complete API reference
  - Architecture overview
  - Mathematical theory
  - Performance guide
  - FAQ (30+ questions)
  - Troubleshooting guide
  - Glossary

- ✅ Code quality
  - Pre-commit hooks (Black, Ruff, Mypy)
  - Modern type annotations
  - All quality checks passing
  - PEP 8 compliant

- ✅ Release automation
  - GitHub Actions for PyPI publishing
  - Pre-release verification
  - Release preparation scripts
  - Post-release scripts

- ✅ Project documentation
  - README.md with badges
  - CHANGELOG.md
  - CONTRIBUTING.md
  - RELEASING.md
  - LICENSE (MIT)
  - GitHub issue templates
  - Pull request template

---

## Deferred Items

### DummyEmulator Bounds Checking Enhancement
- **Status:** Quick fixes applied in v0.1.0
- **Full implementation:** Deferred to v0.2.0
- **Reason:** Current solution works well; full enhancement is nice-to-have
- **Design:** Documented in context.md

### Advanced Samplers
- **Status:** MetropolisHastings working well
- **Advanced options:** Deferred to v0.2.0+
- **Reason:** Simple sampler sufficient for initial release
- **Plan:** Add emcee, dynesty as optional integrations

### Interface Modules
- **Status:** Stubs created
- **Implementation:** Deferred to v0.2.0+
- **Reason:** Core functionality prioritized
- **Plan:** CCL first, then PyTorch, then others

---

## Version Planning

### v0.1.0 (Current - Ready for Release)
**Theme:** Core Functionality
- ✅ Core emulation and inference
- ✅ Basic emulator (DummyEmulator)
- ✅ Basic inference (MetropolisHastings)
- ✅ Comprehensive documentation
- ✅ Example workflows

### v0.2.0 (Next - Planned)

**Theme:** Advanced Emulation
- CCL interface for theory calculations
- PyTorch neural network emulators
- Gaussian Process emulators
- Improved bounds checking
- Real data examples
- Advanced MCMC (emcee integration)

**Timeline:** 2-3 months after v0.1.0

### v0.3.0 (Planned)
**Theme:** Advanced Inference
- TensorFlow emulator support
- Nested sampling integration
- Model comparison tools
- Command-line interface
- Performance optimizations
- Parallel processing

**Timeline:** 4-6 months after v0.2.0

### v1.0.0 (Target)
**Theme:** Production Ready
- All interface modules complete
- Web application
- Comprehensive real-data examples
- Published validation studies
- Community adoption
- Performance optimized
- Full documentation suite

**Timeline:** 12+ months

---

## Ideas for Future Consideration

### Research Features
- Integration with popular inference libraries (Cobaya, CosmoSIS)
- Symbolic regression for interpretable models
- Active learning for emulator training
- Multi-fidelity emulation
- Simulation-based inference (SBI)
- Likelihood-free inference

### Infrastructure
- Web interface for interactive analysis
- Cloud deployment support (AWS, GCP, Azure)
- Containerization (Docker, Singularity)
- Automated hyperparameter tuning
- Continuous benchmarking
- Performance regression testing

### Community
- Regular release schedule (quarterly)
- Developer meetings/calls
- User survey for priorities
- Conference presentations
- Tutorial workshops
- Summer school materials

### Integration
- LSST/DESC integration
- Euclid pipeline integration
- CMB analysis tools (Planck, SO, CMB-S4)
- Multi-messenger cosmology
- Cross-correlation analyses

---

## Priority Matrix

### High Impact, High Effort
- CCL interface implementation
- Neural network emulators
- Web application

### High Impact, Low Effort
- emcee integration
- Real data examples
- Performance profiling

### Low Impact, High Effort
- Full web application
- GPU acceleration
- Distributed computing

### Low Impact, Low Effort
- Additional examples
- Minor documentation improvements
- Code cleanup

---

## Dependencies Between Tasks

v0.2.0:
CCL Interface → Real Data Examples
PyTorch Interface → NN Emulators
emcee Integration → Convergence Diagnostics

v0.3.0:
Model Comparison → Evidence Calculation
CLI Tools → Pipeline System
Performance Opts → Parallel Processing

v1.0.0:
All Interfaces → Full Feature Set
Community Adoption → Production Use

---

## Notes

### Lessons from v0.1.0
- Comprehensive documentation is worth the time
- Release automation saves effort
- Edge cases matter (extrapolation, bounds)
- Good examples are crucial
- Pre-commit hooks enforce quality

### For Future Releases
- Maintain documentation quality
- Keep tests comprehensive
- Preserve backward compatibility
- Communicate changes clearly
- Engage with users early

---

## Tracking

### GitHub Milestones
- v0.1.0: Complete ✅
- v0.2.0: Created (https://github.com/KIPAC/c2i2o/milestone/2)
- v0.3.0: Created (https://github.com/KIPAC/c2i2o/milestone/3)

### GitHub Projects
- Development board: https://github.com/KIPAC/c2i2o/projects/1
- Feature requests: Track in issues
- Bug reports: Track in issues

---

**Last Review:** 2024-12-14
**Next Review:** After v0.1.0 release
**Status:** Ready for v0.1.0 🚀
