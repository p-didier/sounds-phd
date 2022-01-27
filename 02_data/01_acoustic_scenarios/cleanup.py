from pathlib import Path
import os, sys
import shutil

# ----------------------------------

folder = 'validations'  # folder which contents are to be deleted

# ----------------------------------

def main(folder):
    curdir = Path(__file__).parent
    pth = f'{curdir}/{folder}'

    pr = input(f'/!\\ Delete _contents_ of folder "{folder}" (located in current directory)? [Y/N] ')

    if pr == 'y' or pr == 'Y':
        for filename in os.listdir(pth):
            file_path = os.path.join(pth, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        print(f'Folder "{folder}" is now empty.')
    else:
        print(f'Nothing was done.')

# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main(folder))
# ------------------------------------------------------------------------------------