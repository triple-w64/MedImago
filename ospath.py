import os
from pathlib import Path
import win32api, win32con


def isfile(path:str):
    return os.path.isfile(path)

def is_hidden(filepath):
    try:
        attrs = win32api.GetFileAttributes(filepath)
        return bool(attrs & win32con.FILE_ATTRIBUTE_HIDDEN)
    except Exception as e:
        # print(f"Error: {filepath} {e}")
        return False

def listdir(path:str, type:int = 0):
    # os.listdir
    if type==0:
        childrenRelpath = os.listdir(path)
        childrenAbspath = [os.path.join(path,childRelpath) for childRelpath in childrenRelpath]
        dirconts = childrenAbspath
    # os.scandir
    elif type==1:
        with os.scandir(path) as entries:
            dirconts = [entry.path for entry in entries]
    # pathlib
    elif type==2:
        p = Path(path)
        dirconts = [path / child for child in p.iterdir()]
        print(dirconts[:2])
        # print(type(dirconts[0]))
    # glob
    elif type==3:
        import glob
        dirconts = glob.glob(os.path.join(path, '*'))
    # print(dirconts[:3])
    
    excludelist = [".dll"]
    for exclusion in excludelist:
        dirconts = [path for path in dirconts if not path.endswith(exclusion)]
    dirconts = [path for path in dirconts if not is_hidden(path)]
    
    return dirconts
    
if __name__=="__main__":
    res = listdir(r"D:",3)
    print("res",res)
    
