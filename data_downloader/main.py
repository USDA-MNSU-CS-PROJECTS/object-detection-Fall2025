import os
import subprocess
from refresh_token import get_new_refresh_token

def update_env_file(new_token):
    """
    Updates the BOX_REFRESH_TOKEN in the .env file.
    """
    env_file = '.env'
    lines = []
    token_updated = False
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()

    with open(env_file, 'w') as f:
        for line in lines:
            if line.strip().startswith('BOX_REFRESH_TOKEN='):
                f.write(f'BOX_REFRESH_TOKEN={new_token}\n')
                token_updated = True
            else:
                f.write(line)
        if not token_updated:
            f.write(f'BOX_REFRESH_TOKEN={new_token}\n')

def main():
    """
    Gets a new refresh token, updates the .env file, and runs the test script.
    """
    print("Attempting to get a new refresh token...")
    new_refresh_token = get_new_refresh_token()

    if new_refresh_token:
        print("Updating .env file with the new refresh token...")
        update_env_file(new_refresh_token)
        print("Running the test download script...")
        
        # Ensure the script is run with the correct python from the venv
        venv_python = os.path.join('.venv', 'bin', 'python')
        if not os.path.exists(venv_python):
             venv_python = 'python3' # Fallback for different venv structures

        result = subprocess.run([venv_python, 'test_download_script.py'], capture_output=True, text=True)
        
        print("--- Test Script Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Test Script Errors ---")
            print(result.stderr)
        print("--------------------------")

    else:
        print("Could not get a new refresh token. Aborting.")

if __name__ == "__main__":
    main()
