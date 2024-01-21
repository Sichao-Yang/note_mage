<h1 align="center">
<img src="docs/logo.jpg" width="300">
</h1><br>

# NoteMage

This is a collection of useful commands for note management.


# Quick Start

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
