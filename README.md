# NeoRodinia Benchmark Suite

## Description

The NeoRodinia Benchmark Suite is designed for educating and benchmarking parallel programming using OpenMP. It implements various parallelization levels, demonstrating their impact on performance, and serves as a tool for performance evaluation across different computing systems.

## Features

- **Structured Optimization**: Introduces parallelization levels for structured optimization in parallel programming.
- **Automated Testing Framework**: Simplifies the process of performance data collection and plotting for parallel programming benchmarking.
- **Educational Tool**: Acts as an instructional guide for structured application parallelization.

## Getting Started

### Prerequisites

- Linux operating system
- GCC 11, LLVM 17, NVIDIA HPC SDK 22.1

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/passlab/NeoRodinia.git
   ```
2. Navigate to the desired benchmark directory:
   ```bash
   cd NeoRodinia/<benchmark_name>
   ```
3. Compile the benchmark using the provided Makefiles:
   ```bash
   make all CC=<compiler> OPT_LEVEL=<O1/O2/O3>
   ```

### Usage

To run the benchmarks using the automated testing framework:
1. Execute the benchmarks:
   ```bash
   python ../csv_generator.py
   ```
2. Collect and plot the performance data:
   ```bash
   python ../figure_generator.py
   ```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the BSD License.

## Citation

If you use this suite in your research, please cite:
- [To be determined]

## Acknowledgments

- Thanks to all contributors who have helped in developing and refining the NeoRodinia benchmark suite.
