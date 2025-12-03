# Data Downloader Documentation

This downloader automates the process of acquiring and preparing image data from Box for machine learning tasks.

> [!NOTE]
> **One-Time Use & Limitations**: This code was originally written as a one-time utility to download raw data for the initial dataset. It is specifically hardcoded to:
> *   Target the '10xstitch' folder.
> *   Download only images corresponding to timepoints **t0** or **t4**.
>
> **For future use cases**, you will likely need to modify `downloader.py` to adjust these filters and target folders.

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

To run the downloader, execute `run_downloader.py` from the `src` directory. Ensure your Box API credentials are up-to-date in the `.env` file. If the refresh token expires, use `refresh_token.py` to obtain a new one.
