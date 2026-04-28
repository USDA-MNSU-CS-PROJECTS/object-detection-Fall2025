# Data Downloader Documentation

This downloader automates the process of acquiring and preparing image data from Box for machine learning tasks.

> [!NOTE]
> **One-Time Use & Limitations**: This code was originally written as a one-time utility to download raw data for the initial dataset. It is specifically hardcoded to:
> *   Target the '10xstitch' folder.
> *   Download only images corresponding to timepoints **t0** or **t4**.
>
> **For future use cases**, you will likely need to modify `downloader.py` to adjust these filters and target folders.

> [!IMPORTANT]
> `run_downloader.py` is not cross-platform in its current form. It calls Conda using a hardcoded macOS path:
> `/opt/anaconda3/bin/conda`.
> If you are on Windows/Linux, either:
> - run `downloader.py` directly in your active environment, or
> - update `run_downloader.py` to use your local Conda/Python path.

## How it Works

1.  **Authentication**: Uses Box API credentials (Client ID, Client Secret, and Refresh Token) stored in a `.env` file to authenticate and access the specified Box folder.
2.  **Iterative Download**: Navigates through sub-folders within the main Box folder (specifically targeting folders with 'pg' in their name).
3.  **Selective Processing**: Identifies `.nd2` image files within these sub-folders that contain 't0_' or 't4_' in their filenames.
4.  **Two-Step Conversion**: Each selected `.nd2` file is:
    a.  Downloaded to a temporary location on disk.
    b.  Converted to a `.tiff` format using `nd2_converter.py`.
    c.  Further converted from `.tiff` to `.png` format using `png_converter.py`.
5.  **Cleanup**: Intermediate `.nd2` and `.tiff` files are removed after successful PNG conversion.
6.  **Archiving**: All processed `.png` images within a sub-folder are compressed into a single `.zip` archive.
7.  **Directory Removal**: The temporary sub-folder containing the `.png` images is removed after zipping to save space.

## Usage

To run the downloader:

1. Ensure Box API credentials are present in `src/data_downloader/.env`.
2. Preferred (portable) path: run `downloader.py` directly in the prepared environment.
3. Optional launcher path: run `run_downloader.py` only after adapting its hardcoded Conda path to your machine.
4. If the refresh token expires, use `refresh_token.py` to obtain a new token.

This utility is considered legacy tooling and is not part of the main Gradio runtime path (`src/app/main.py`).
