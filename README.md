# Illusion Research

This repository is focused on conducting research in the field of illusions. The project dependencies are managed with [uv](https://pypi.org/project/uv/), streamlining both package management and dependency handling.

## Setup Instructions

1. **Install uv package manager**  
   ```bash
   pip install uv
   ```

2. **Synchronize dependencies**  
   Ensure that dependencies listed in `pyproject.toml` are installed:
   ```bash
   uv sync
   ```

3. **Add Additional Packages**  
   Add new packages to the project:
   ```bash
   uv add <package>
   ```

4. **Requirements File (optional)**  
   While `uv` manages dependencies, a `requirements.txt` file is provided for compatibility. To install via `pip` instead:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Make necessary modifications to `pyproject.toml` for any specific `uv` configurations.

## Training

To start training, run the following bash script:
```bash
bash train.sh

or

python train.py
```

## License

This project is licensed under [Apache2.0 License](LICENSE).
