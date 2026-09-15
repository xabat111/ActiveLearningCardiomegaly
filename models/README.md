# Model weights

The CheXNet starting checkpoint now lives in this folder as `m-30012020-104001.pth` (84 MB), tracked with Git LFS. Checkpoints produced by the Active Learning loop are not committed; follow the instructions below if you want to publish one.

The two kinds of checkpoint are not loaded the same way. The starting CheXNet weights are read as `checkpoint['state_dict']`, while the ones saved by the Active Learning loop are named `best_model<percentage>_strategy_<strategy>_F1_<score>.pth` and contain a bare `state_dict`.

## Getting the weights

`git clone` only brings down the LFS pointer files, not the weights themselves, so a fresh clone leaves a three-line text file where the checkpoint should be. To fetch the real content:

```bash
git lfs install
git lfs pull
```

From Colab, where the repository is usually cloned fresh on every session, `git-lfs` has to be installed first:

```python
!apt-get -qq install git-lfs
!git clone https://github.com/xabat111/ActiveLearningCardiomegaly.git
%cd ActiveLearningCardiomegaly
!git lfs install
!git lfs pull
```

Then point `MODEL_PATH` at `models/m-30012020-104001.pth`.

If more checkpoints get added later and you only want this one, `git lfs pull --include models/m-30012020-104001.pth` skips the rest.

## Adding another checkpoint

Checkpoints are around 85 MB, above GitHub's 50 MB warning threshold and well above the 25 MB cap of the web interface, so they have to go through Git LFS from the command line. `.gitattributes` already routes `*.pth` and `*.pth.tar` through LFS, so there is nothing else to configure.

Because the checkpoints already live in Google Drive, they can go straight from Colab to the repository without passing through your machine. You need a GitHub [personal access token](https://github.com/settings/tokens) with `repo` scope. Never paste it into a cell — read it with `getpass` so it is not saved inside the notebook.

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
DST   = 'models/' + os.path.basename(SRC)

os.chdir('/content')
!git clone https://{TOKEN}@github.com/{USER}/{REPO}.git
os.chdir(f'/content/{REPO}')

!git lfs install
shutil.copy(SRC, DST)

!git config user.email "you@example.com"
!git config user.name "Your Name"
!git add {DST}
!git commit -m "Add checkpoint"
!git lfs ls-files          # should list the file, i.e. it is a pointer and not a raw blob
!git push
```

Bear in mind that LFS storage and bandwidth count against the account quota (1 GB free on GitHub), which is roughly eleven checkpoints of this size.

## Alternative — attach weights to a GitHub Release

Once the LFS quota gets tight, publish further checkpoints as release assets instead. They accept files up to 2 GB, do not consume the LFS quota, and can be uploaded by drag and drop from a browser: go to **Releases → Draft a new release**, create a tag such as `weights-v1`, and drop the `.pth` file in the "Attach binaries" area.

Downloading one afterwards from a notebook needs no token while the repository is public:

```python
!wget -O m-30012020-104001.pth \
  https://github.com/xabat111/ActiveLearningCardiomegaly/releases/download/weights-v1/m-30012020-104001.pth
```

Then point `MODEL_PATH` at the downloaded file.
