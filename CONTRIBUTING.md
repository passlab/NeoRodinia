# Contributing to NeoRodinia Benchmark Suite

We welcome contributions from the community! This document provides guidelines for contributing to the NeoRodinia Benchmark Suite. By participating in this project, you agree to abide by its terms.

## Code of Conduct
Our community has a Code of Conduct that all contributors are expected to follow. This code ensures a welcoming and inclusive environment for everyone. https://www.contributor-covenant.org/version/2/1/code_of_conduct/

## How to Contribute
If you are looking to make a contribution to the NeoRodinia Benchmark Suite, you can:

- Submit a pull request with your contributions.
- Report bugs or issues.
- Suggest enhancements or new benchmarks.
- Help improve documentation.

Please, before making a submission, ensure your contribution aligns with our coding and structure standards described below.

## Benchmark Structure Standards
To maintain a high standard of quality and consistency, we follow a standardized structural design for all benchmarks within our framework:

- **Common Driver**: Each benchmark includes a common driver that handles initialization, parameter processing, timing information, verification of results, and other core functions.
- **Separate Kernel Files**: Each kernel is stored in a separate file to improve readability and ease of testing.
- **Naming Convention**: All kernels follow a naming convention that indicates their functionality and parallelization strategy.

## Contribution Format
When adding a new benchmark or modifying an existing one, please adhere to the following format:

- Compile the `main.c` driver, the utility functions, and the corresponding kernel files.
- Each kernel is stored in a separate file. All the kernels use the same kernel name.
- Use the shared utilities file for functions such as timing, thread count retrieval, and more.
- Utilize the `NR_VERIFY` and `NR_FULL_REPORT` environment variables to control output during testing.

This ensures that any additions or changes are compatible with our testing and performance evaluation processes.

## Submission Process
To submit a pull request, please follow these steps:

1. Fork the repository.
2. Create a new branch for your contribution.
3. Add or update benchmarks following our structure standards.
4. Commit your changes with a clear and detailed commit message.
5. Push your branch and open a pull request against our main branch.

## Review Process
Our maintainers will review each pull request. They may provide feedback or request changes to ensure that the contribution meets the project's standards. Once approved, the maintainers will merge the pull request.

## Questions or Help
If you have any questions or need help with your contribution, please open an issue in the repository. Our community is here to help!

Thank you for your interest in contributing to the NeoRodinia Benchmark Suite. We look forward to your valuable input!
