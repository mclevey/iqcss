import ast
import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from rich.logging import RichHandler


def set_torch_device() -> Any:
    """Set and return the optimal torch device (CUDA, MPS, or CPU).

    Returns:
        Torch device object with logging info about the selected device.
    """
    import torch

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        device_properties: Any = torch.cuda.get_device_properties(device)
        vram = device_properties.total_memory // (1024**2)
        logging.info(
            f"Set device to {device} with {vram}MB (~ {np.round(vram / 1024)}GB) of VRAM"
        )
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
        logging.info(f"Set device to {device}")
    else:
        device = torch.device("cpu")
        logging.info(f"Set device to {device}")
    return device


def find_env_file(start_path: Path = Path.cwd()) -> Optional[Path]:
    """Find .env file by searching up the directory tree.

    Args:
        start_path: Starting directory to search from.

    Returns:
        Path to .env file if found, None otherwise.
    """
    current_path = start_path.resolve()

    # Search up the directory tree
    while current_path != current_path.parent:  # Stop at filesystem root
        env_file = current_path / ".env"
        if env_file.exists():
            return current_path
        current_path = current_path.parent

    # Check filesystem root as well
    env_file = current_path / ".env"
    if env_file.exists():
        return current_path

    return None


def load_api_key(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Load API key from environment file.

    Args:
        key: Name of the environment variable.
        env_path: Path to directory containing .env file. If None, will search
                 up from current working directory.

    Returns:
        API key value or None if not found.

    Note:
        Searches for .env file starting from current directory and moving up
        the directory tree until found. Should be added to .gitignore.
    """
    if env_path is None:
        env_path = find_env_file()
        if env_path is None:
            logging.warning("No .env file found in directory tree")
            return None

    load_dotenv(env_path / ".env")
    api_key = os.getenv(key)
    return api_key


def load_api_key_list(
    key_names: List[str], env_path: Optional[Path] = None
) -> List[Optional[str]]:
    """Load multiple API keys from environment file.

    Args:
        key_names: List of environment variable names.
        env_path: Path to directory containing .env file. If None, will search
                 up from current working directory.

    Returns:
        List of API key values (None for keys not found).

    Note:
        Searches for .env file starting from current directory and moving up
        the directory tree until found. Should be added to .gitignore.
    """
    if env_path is None:
        env_path = find_env_file()
        if env_path is None:
            logging.warning("No .env file found in directory tree")
            return [None] * len(key_names)

    load_dotenv(env_path / ".env")
    keys: List[Optional[str]] = []
    for key in key_names:
        api_key = os.getenv(key)
        keys.append(api_key)
    return keys


def initialize_logger(logging_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, logging_level),
        format="%(asctime)s\n%(message)s",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger("rich")
    return logger


def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file with error handling.

    Args:
        data: Data to save to JSON.
        file_path: Path to save the JSON file.

    Returns:
        None.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"An error occurred while saving data to {file_path}: {e}")


def get_fpaths_and_fnames(dir: str, ftype: str = "json") -> List[Tuple[Path, str]]:
    """Get file paths and names from directory.

    Args:
        dir: Directory path to search.
        ftype: File type extension to search for.

    Returns:
        List of tuples containing (Path, filename_stem).
    """
    directory = Path(dir)
    files = glob.glob(str(directory / f"*.{ftype}"))
    fpaths_fnames = [(Path(file), Path(file).stem) for file in files]
    return fpaths_fnames


def strings_to_lists(series: Any) -> Any:
    """Convert string representations of lists to actual lists.

    Args:
        series: Pandas Series containing string representations of lists.

    Returns:
        Series with strings converted to lists using ast.literal_eval.
    """
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


def lists_to_strings(series: Any, sep: str = ", ") -> Any:
    """Convert lists to string representations.

    Args:
        series: Pandas Series containing lists.
        sep: Separator to use when joining list elements.

    Returns:
        Series with lists converted to strings.

    Note:
        If the lists data you want as a string is stored as a string,
        you need to convert it to lists first, then back to the string.
    """
    series = strings_to_lists(series)
    return series.apply(lambda x: sep.join(x) if isinstance(x, list) else x)


def run_in_conda(script: str, conda_env_name: str = "gt") -> None:
    conda_script = script

    command = (
        "source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {conda_env_name} && "
        f"python {conda_script} && "
        "conda deactivate"
    )

    logging.info(f"Executing command: {command}")

    try:
        process = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(
            f"Successfully executed '{conda_script}' in conda env '{conda_env_name}'\n"
            f"Output:\n{process.stdout.decode()}"
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            f"Failed to execute '{conda_script}' in conda env '{conda_env_name}'\n"
            f"Error:\n{e.stderr.decode()}"
        )


def markdown_table(
    df: pd.DataFrame, filepath: Optional[str] = None, indexed: bool = False
) -> str:
    """Convert a pandas DataFrame to a markdown table.

    Args:
        df: The DataFrame to convert to markdown.
        filepath: The path where the markdown file should be saved.
        indexed: Whether to include the DataFrame index in the markdown table.

    Returns:
        The markdown formatted table as a string.
    """
    pd.set_option("display.float_format", lambda x: "%.0f" % x)

    md = df.to_markdown(
        index=indexed
    )  # Convert the DataFrame to markdown with or without index

    if filepath is not None:
        with open(filepath, "w") as file:
            file.write(md)  # Write the markdown string to file if filepath provided

    return md  # Return the markdown string


def estimate_meters_from_rssi(
    df: pd.DataFrame, rssi_col: str, A: int = -40, n: int = 2
) -> Any:
    """Estimate distance in meters from RSSI values.

    Args:
        df: DataFrame containing RSSI data.
        rssi_col: Column name containing RSSI values.
        A: RSSI value at 1 meter distance.
        n: Path-loss exponent.

    Returns:
        Estimated distances in meters.
    """
    estimated_meters = 10 ** ((A - df[rssi_col]) / (10 * n))
    return estimated_meters


def update_quarto_variables(
    new_key: str, new_value: Any, path: str = "_variables.yml"
) -> None:
    """Update Quarto variables YAML file.

    Args:
        new_key: Key name to add or update.
        new_value: Value to set for the key.
        path: Path to the variables YAML file.

    Returns:
        None.
    """
    with open(path, "r") as file:
        quarto_variables = yaml.safe_load(file)

    # add a new key-value pair or update an existing key
    quarto_variables[new_key] = new_value

    with open(path, "w") as file:
        yaml.dump(quarto_variables, file, default_flow_style=False)
