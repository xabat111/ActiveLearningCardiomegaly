# Model weights

Trained checkpoints live in this folder and are stored with [Git LFS](https://git-lfs.com/), since they are around 85 MB each and GitHub rejects plain-git files over 100 MB.

The notebooks expect the CheXNet starting checkpoint at `m-30012020-104001.pth`, loaded as `checkpoint['state_dict']`. Checkpoints produced by the Active Learning loop are saved as `best_model<percentage>_strategy_<strategy>_F1_<score>.pth` and contain a bare `state_dict`.

## Adding a checkpoint

Install Git LFS once per machine, then commit the file normally — `.gitattributes` already routes `*.pth` and `*.pth.tar` through LFS:

```bash
git lfs install
cp /path/to/m-30012020-104001.pth models/
git add models/m-30012020-104001.pth
git commit -m "Add CheXNet checkpoint"
git push
```

Verify it was stored as an LFS pointer rather than a raw blob:

```bash
git lfs ls-files
```

## Cloning

`git clone` fetches the pointer files; run `git lfs pull` to download the actual weights. Note that LFS bandwidth and storage count against the account quota (1 GB free on GitHub), so if that becomes a problem, attach the checkpoint to a GitHub Release or keep it in Google Drive and download it from the notebook instead.
