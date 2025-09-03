"""
Data loading utilities for the iqcss package.

This module provides easy access to datasets included with the iqcss package.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


def _get_data_path() -> Path:
    """Get the path to the data directory within the iqcss package."""
    return Path(__file__).parent / "data"


class DataLoader:
    """Base class for data loaders."""

    def __init__(self, data_dir: str):
        self.data_dir = _get_data_path() / data_dir
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def list_files(self) -> List[str]:
        """List available data files in this directory."""
        return [f.name for f in self.data_dir.iterdir() if f.is_file()]


class WNBADataLoader(DataLoader):
    """Loader for WNBA datasets."""

    def __init__(self):
        super().__init__("wnba")
        self._available_datasets = {
            "wnba": "wnba.csv",
            "wnba_player_text": "wnba_player_text.csv",
            "wnba_team_rosters_2024": "wnba_team_rosters_2024.csv",
        }

    def load(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load a WNBA dataset.

        Args:
            dataset_name: Name of the dataset to load. Options: 'wnba',
                         'wnba_player_text', 'wnba_team_rosters_2024'
            **kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            pandas.DataFrame: The loaded dataset

        Raises:
            ValueError: If dataset_name is not recognized
            FileNotFoundError: If the data file doesn't exist
        """
        if dataset_name not in self._available_datasets:
            available = list(self._available_datasets.keys())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            )

        filename = self._available_datasets[dataset_name]
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        return pd.read_csv(filepath, **kwargs)

    def about(self, dataset_name: Optional[str] = None) -> None:
        """
        Display information about available WNBA datasets.

        Args:
            dataset_name: If provided, show info for specific dataset.
                         Otherwise show all.
        """
        print("WNBA Datasets:")

        if dataset_name:
            if dataset_name not in self._available_datasets:
                available = list(self._available_datasets.keys())
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
                )
            datasets_to_show = [dataset_name]
        else:
            datasets_to_show = list(self._available_datasets.keys())

        for name in datasets_to_show:
            filename = self._available_datasets[name]
            filepath = self.data_dir / filename

            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    print(f"  {name}: {filename} (shape: {df.shape})")
                    if hasattr(df, "columns"):
                        cols = ", ".join(df.columns.tolist()[:5])
                        if len(df.columns) > 5:
                            cols += f", ... (+{len(df.columns) - 5} more)"
                        print(f"    Columns: {cols}")
                except Exception as e:
                    print(f"  {name}: {filename} (could not read: {e})")
            else:
                print(f"  {name}: {filename} (FILE NOT FOUND)")


class YouTubeDataLoader(DataLoader):
    """Loader for YouTube comment datasets."""

    def __init__(self):
        super().__init__("youtube")
        self._available_datasets = {
            "comments": "yearly",
            "comments_sampled": "comments_sampled.csv",
        }

    def load(self, dataset_name: str, **kwargs) -> pd.DataFrame:
        """
        Load a YouTube dataset.

        Args:
            dataset_name: Name of the dataset to load. Options: 'comments', 'comments_sampled'
            **kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            pandas.DataFrame: The loaded dataset

        Raises:
            ValueError: If dataset_name is not recognized
            FileNotFoundError: If the data file doesn't exist
        """
        if dataset_name not in self._available_datasets:
            available = list(self._available_datasets.keys())
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
            )

        filename_or_dir = self._available_datasets[dataset_name]

        # Handle the special case of 'comments' which loads from yearly files
        if dataset_name == "comments":
            return self._load_yearly_comments(**kwargs)

        # Handle regular single file datasets
        filepath = self.data_dir / filename_or_dir
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        return pd.read_csv(filepath, **kwargs)

    def _load_yearly_comments(self, **kwargs) -> pd.DataFrame:
        """
        Load the full comments dataset by concatenating yearly files.

        Args:
            **kwargs: Additional arguments passed to pd.read_csv()

        Returns:
            pandas.DataFrame: The concatenated dataset from all yearly files

        Raises:
            FileNotFoundError: If the yearly directory doesn't exist or has no files
        """
        yearly_dir = self.data_dir / "yearly"

        if not yearly_dir.exists():
            raise FileNotFoundError(f"Yearly data directory not found: {yearly_dir}")

        # Find all yearly CSV files
        yearly_files = sorted(
            [
                f
                for f in yearly_dir.iterdir()
                if f.is_file()
                and f.name.startswith("comments_")
                and f.name.endswith(".csv")
            ]
        )

        if not yearly_files:
            raise FileNotFoundError(f"No yearly comment files found in: {yearly_dir}")

        # Load and concatenate all yearly files
        dataframes = []
        for filepath in yearly_files:
            try:
                # Try standard loading first
                df = pd.read_csv(filepath, **kwargs)
                dataframes.append(df)
            except Exception as e:
                # Try with error handling for malformed data
                try:
                    df = pd.read_csv(
                        filepath,
                        encoding="utf-8",
                        quoting=1,
                        on_bad_lines="skip",
                        engine="python",  # More robust but slower
                        **kwargs,
                    )
                    dataframes.append(df)
                    # print(
                    #     f"Warning: Loaded {filepath.name} with error recovery ({len(df)} rows)"
                    # )
                except Exception as e2:
                    print(f"Warning: Could not load {filepath.name}: {e2}")
                    continue

        if not dataframes:
            raise FileNotFoundError("No yearly files could be loaded successfully")

        # Concatenate all dataframes
        full_df = pd.concat(dataframes, ignore_index=True)

        # Sort by published_at for consistency
        if "published_at" in full_df.columns:
            full_df["published_at"] = pd.to_datetime(full_df["published_at"])
            full_df = full_df.sort_values("published_at").reset_index(drop=True)

        return full_df

    def available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self._available_datasets.keys())

    def about(self, dataset_name: Optional[str] = None) -> None:
        """
        Print information about available datasets.

        Args:
            dataset_name: If provided, show info for specific dataset. Otherwise show all.
        """
        if dataset_name:
            if dataset_name not in self._available_datasets:
                available = list(self._available_datasets.keys())
                print(
                    f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
                )
                return

            # Handle the special case of 'comments' which loads from yearly files
            if dataset_name == "comments":
                try:
                    df = self._load_yearly_comments()
                    yearly_dir = self.data_dir / "yearly"
                    yearly_files = [
                        f.name
                        for f in yearly_dir.iterdir()
                        if f.is_file()
                        and f.name.startswith("comments_")
                        and f.name.endswith(".csv")
                    ]
                    print(f"Dataset: {dataset_name}")
                    print(
                        f"Source: {len(yearly_files)} yearly files ({sorted(yearly_files)[0]} to {sorted(yearly_files)[-1]})"
                    )
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"Error loading {dataset_name}: {e}")
            else:
                filename = self._available_datasets[dataset_name]
                filepath = self.data_dir / filename

                if filepath.exists():
                    df = pd.read_csv(filepath)
                    print(f"Dataset: {dataset_name}")
                    print(f"File: {filename}")
                    print(f"Shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                else:
                    print(f"File not found: {filepath}")
        else:
            print("Available YouTube datasets:")
            for name, filename_or_dir in self._available_datasets.items():
                if name == "comments":
                    try:
                        df = self._load_yearly_comments()
                        yearly_dir = self.data_dir / "yearly"
                        yearly_files = [
                            f.name
                            for f in yearly_dir.iterdir()
                            if f.is_file()
                            and f.name.startswith("comments_")
                            and f.name.endswith(".csv")
                        ]
                        print(
                            f"  {name}: {len(yearly_files)} yearly files (shape: {df.shape})"
                        )
                    except Exception:
                        print(f"  {name}: yearly files (ERROR LOADING)")
                else:
                    filepath = self.data_dir / filename_or_dir
                    if filepath.exists():
                        df = pd.read_csv(filepath)
                        print(f"  {name}: {filename_or_dir} (shape: {df.shape})")
                    else:
                        print(f"  {name}: {filename_or_dir} (FILE NOT FOUND)")


# Create instances that users can import directly
wnba = WNBADataLoader()
youtube = YouTubeDataLoader()
youtube_sampled = (
    YouTubeDataLoader()
)  # For backward compatibility, though both use the same loader


# Convenience functions for direct access
def load_wnba_data(dataset_name: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load WNBA datasets.

    Args:
        dataset_name: Name of the dataset ('wnba', 'wnba_player_text',
                     'wnba_team_rosters_2024')
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        pandas.DataFrame: The loaded dataset
    """
    return wnba.load(dataset_name, **kwargs)


def load_youtube_data(dataset_name: str = "comments", **kwargs) -> pd.DataFrame:
    """
    Convenience function to load YouTube datasets.

    Args:
        dataset_name: Name of the dataset ('comments' or 'comments_sampled')
        **kwargs: Additional arguments passed to pd.read_csv()

    Returns:
        pandas.DataFrame: The loaded dataset
    """
    return youtube.load(dataset_name, **kwargs)
