"""
@Author: Conghao Wong
@Date: 2024-12-04 09:44:45
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-04 14:48:40
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import requests
import shutil

GITHUB_USERNAME = 'cocoon2wong'
GITHUB_REPONAME = 'Re'
GITHUB_READMEFILE = 'README.md'

SOURCE_FILE = './guidelines.md'
TARGET_FILE = './README.md.downloaded'

START_LINE = '## Getting Started'

DOWNLOAD_SOURCE = f'https://github.com/{GITHUB_USERNAME}/{GITHUB_REPONAME}/raw/refs/heads/main/{GITHUB_READMEFILE}'


if __name__ == '__main__':
    # Backup old files
    shutil.copy(SOURCE_FILE, SOURCE_FILE + '.backup')

    # Fetch new file
    with open(TARGET_FILE, 'wb') as f:
        _content = requests.get(DOWNLOAD_SOURCE)
        f.write(_content.content)

    with open(TARGET_FILE, 'r') as f:
        new_lines = f.readlines()

    # Check lines
    for i, line in enumerate(new_lines):
        if line.startswith(START_LINE):
            break

    # Write new file
    with open(SOURCE_FILE, 'a+') as f:
        f.writelines(new_lines[i:])
