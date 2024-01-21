## 0.2.1 (2024-01-21)

### Fix

- fixed Literal carriage return issue
- fixed issue of too aggresive pre-commit by switching off some checks
- **commitizen-cfg**: the version bumping was not working since its hooked with auto project version check, now it is fixed by recording version info in .cz.toml with commitizen

## 0.2.0 (2024-01-21)

### Feat

- add setup.py and commitizen for changelog automation and move img_concat to image folder, add a skeleton for cli
- add dst_dir input arg in renamer
- add args parser in both typora functions
