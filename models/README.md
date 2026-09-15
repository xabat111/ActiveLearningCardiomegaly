# Model weights

Trained checkpoints belong in this folder. They are around 85 MB each, which is over GitHub's 50 MB warning threshold, so they cannot be uploaded through the GitHub web interface (that one caps at 25 MB per file). Use one of the two options below instead.

The notebooks expect the CheXNet starting checkpoint at `m-30012020-104001.pth`, loaded as `checkpoint['state_dict']`. Checkpoints produced by the Active Learning loop are saved as `best_model<percentage>_strategy_<strategy>_F1_<score>.pth` and contain a bare `state_dict`.

## Option A — push from Colab with Git LFS

The checkpoint already lives in Google Drive, so it can go straight from Colab to the repository without downloading it to your machine. `.gitattributes` already routes `*.pth` and `*.pth.tar` through LFS.

You need a GitHub [personal access token](https://github.com/settings/tokens) with `repo` scope. Never paste it into a cell — read it with `getpass` so it is not saved inside the notebook.

```python
from google.colab import drive
from getpass import getpass
import os, shutil

drive.mount('/content/drive')
!apt-get -qq install git-lfs

TOKEN = getpass('GitHub token: ')
USER  = 'xabat111'
REPO  = 'ActiveLearningCardiomegaly'
SRC   = '/content/drive/MyDrive/MachineLearning/Part3/CHEXNET/archive (Unzipped Files)/m-30012020-104001.pth'

os.chdir('/content')
!git clone https://{TOKEN}@github.com/{USER}/{REPO}.git
os.chdir(f'/content/{REPO}')

!git lfs install
os.makedirs('models', exist_ok=True)
shutil.copy(SRC, 'models/m-30012020-104001.pth')

!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add models/m-30012020-104001.pth
!git commit -m "Add CheXNet checkpoint"
!git lfs ls-files          # should list the file, i.e. it is a pointer and not a raw blob
!git push
```

Bear in mind that LFS storage and bandwidth count against the account quota (1 GB free on GitHub), which is roughly eleven checkpoints of this size.

## Option B — attach it to a GitHub Release

Release assets accept files up to 2 GB, do not consume the LFS quota, and can be uploaded by drag and drop from a browser: go to **Releases → Draft a new release**, create a tag such as `weights-v1`, and drop the `.pth` file in the "Attach binaries" area.

Downloading it afterwards from a notebook needs no token if the repository is public:

```python
!wget -O m-30012020-104001.pth \
  https://github.com/xabat111/ActiveLearningCardiomegaly/releases/download/weights-v1/m-30012020-104001.pth
```

Then point `MODEL_PATH` at the downloaded file.

## Cloning a repository that has LFS files

`git clone` only fetches the pointer files; run `git lfs install` and `git lfs pull` to download the actual weights.
