<h1 align="center">
<img src="docs/logo.jpg" width="300">
</h1><br>

# NoteMage

This is a collection of useful commands for note management.

# Quick Start

## Install

1. `pip install -r requirements.txt`
2. `python setup.py install`

## Usage

```python
usage: use "nmag --help" for more information

details

positional arguments:
  cmd                   The available cmds are described in format of `cmd | desc | args`:
                            "icat":     | concatenate images                    | -sd, -direction, -dp
                            "cip":      | correct imgpath in md(s)              | -sd|-sp, -bak
                            "rir":      | remove redundant images from md       | -sd, -bak, -ignore
                            "rn":       | rename md                             | -sp, -dp, -auto_cip -bak
                            "p2i":      | pdf to img                            | -sp, -dd
                            "p2m":      | pdf(s) to md                          | -sp|-sd, -dp
                            "pe":       | extract subpages from pdf to pdf      | -sp, -dp, -range
                            "md":       | makedir on primal and mirror paths    | -sd, --mirror_rule, --linking

optional arguments:
  -h, --help            show this help message and exit
  -sd SRC_DIR, --src_dir SRC_DIR
                        the source folder
  -sp SRC_PATH, --src_path SRC_PATH
                        source file
  -dd DST_DIR, --dst_dir DST_DIR
                        the output folder
  -dp DST_PATH, --dst_path DST_PATH
                        the output filepath
  -bak, --backup        backup original data to <output filepath>__bak
  -direction CONCAT_DIRECTION, --concat_direction CONCAT_DIRECTION
                        the direction for concatenation, v or h
  -auto_cip, --auto_imgpath_change
                        change imgpath automatically after renamed file
  -ignore IGNORE_ITEMS [IGNORE_ITEMS ...], --ignore_items IGNORE_ITEMS [IGNORE_ITEMS ...]
                        items to be ignored when collecting redundant files
  -range RANGE          extract pdf range from page a to b: '[a,b]'
  --mirror_rule MIRROR_RULE
                        replace path of src path with part of dst path
  --linking             make shortcut link between two dirs
  -v, --verbose         increase output verbosity
```

# Dive Deeper

## md_red_img_remover

It works with `typora` markdown file by collecting all redundant image files into a folder named 'red_files' for manual check & removal.

Details:

    1. folder path extracting:
        1. recursively get all paths in the target folder
        2. extract md files' path from all paths, all paths include only img paths
        3. check img_paths and warn user if there is any pdf file
    2. extract using imgpaths from md files:
        1. recursively extract all `"![]()" & "<img src=''...>"` patterns in md filelist
        2. filter out all hyper-links and absolute paths e.g.: `![xxx](https://github.com/typora/typora-issues)`
            notice: there maybe some hyper-links rendered as image or contents, it needs to be
            either manually or automatically checked & converted.
    3. redundant path removing:
        1. remove all img_paths used in md_files from all_img_paths in folder, only redundant img_paths left
        2. move red_imgs to redundant folder
    4. manual postprocess:
        1. manual verification of those red_imgs in red folder
        2. check if there is any unmatched relative path in md files (potential broken img link)

It runs with python 3.x, no dependent libs needed.


# Todo List

1. add more documentation
