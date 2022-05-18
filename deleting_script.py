# !!!!! careful !!!!!
# this script is deleting files!!!
# only use very carefully

import shutil
import glob

def delete_nonempty_folder(path):
    files = glob.glob(path + '/**/*', recursive=True)

    print(f'You are about to delete {path} and all its files. Containing the files: \n{files}\n To continue plese enter y or n')
    inputName = input().lower()
    if 'y' in inputName:
        print('deleting files')
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s : %s" % (path, e.strerror))
    else:
        print('Abort deleting.')

    
if __name__ == '__main__':
    path = 'submodules/2022_Masterthesis_UniPELT/testdir'
    if path != '' or path is not None:
        delete_nonempty_folder(path)

