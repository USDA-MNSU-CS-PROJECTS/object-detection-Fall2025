# Simple ND2 to TIFF Converter

## What You Need

1. Python installed on your computer
2. Two special packages for reading ND2 files and writing TIFF files

## Setup (One Time Only)

### Step 1: Install the packages

Open your command prompt/terminal and type:

```
pip install nd2reader tifffile numpy
```

### Step 2: Download the script

Save the `nd2_converter.py` file to your computer

## How to Use It

1. Open command prompt, navigate to where you saved the file, and type:

   ```
   python nd2_converter.py
   ```

   Or run the file through an IDE

2. The program will ask you what you want to do:

   - Type `1` to convert a single file
   - Type `2` to convert all ND2 files in a folder

3. Enter the path to the file or folder that you want to convert.

   **On Windows**

   - File path:
     ```
     C:/Users/YourName/Documents/ND2Files/File.nd2
     ```
   - Folder path:
     ```
     C:/Users/YourName/Documents/ND2Files
     ```

## What It Does

- Finds your ND2 files
- Opens each one
- Saves it as a TIFF file in the same folder
- If there are multiple layers, each one will be saved separately
- Prints progress messages so you know what's happening

## Example Output

```
ND2 to TIFF Converter
1. Convert a single file
2. Convert all files in a folder
Enter your choice (1 or 2): 2
Enter the path to your folder: /path/to/nd2/files
Looking for ND2 files in: /path/to/nd2/files
Found 2 ND2 files
Converting: /path/to/nd2/files/image1.nd2
Saved: /path/to/nd2/files/image1.tiff
Converting: /path/to/nd2/files/image2.nd2
Saved: /path/to/nd2/files/image2.tiff
Successfully converted 2 files
```

## Troubleshooting

- **"No module named 'nd2reader'"**: You need to install the packages (see Step 1)
- **"File not found"**: Check that your file path is correct
- **Permission errors**: Make sure you can write files to that folder
