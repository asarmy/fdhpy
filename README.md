# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/asarmy/fdhpy/blob/coverage-data-branch/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/fdhpy/\_\_init\_\_.py                           |        9 |        0 |    100% |           |
| src/fdhpy/chiou\_et\_al\_2025.py                    |       94 |       11 |     88% |100, 107-108, 137-140, 204-207, 251, 255 |
| src/fdhpy/cli.py                                    |       50 |       46 |      8% |    29-113 |
| src/fdhpy/fault\_displacement\_model.py             |      103 |       18 |     83% |76, 79-93, 99, 148, 194, 240, 288 |
| src/fdhpy/fault\_displacement\_model\_variables.py  |       65 |        7 |     89% |101-107, 157-158, 182-183 |
| src/fdhpy/kuehn\_et\_al\_2024.py                    |      218 |       48 |     78% |119-120, 137-144, 173-174, 178-191, 212-238, 244, 250, 306-309, 472-478, 508-511, 536, 541-546, 615, 625, 629 |
| src/fdhpy/lavrentiadis\_abrahamson\_2023.py         |      292 |       55 |     81% |112-113, 132, 135-136, 146-153, 165-168, 189-192, 271-274, 312-315, 325-328, 407, 427-430, 480-481, 601, 619-629, 639-640, 677, 686, 690 |
| src/fdhpy/loglinear\_scaling\_model.py              |       67 |        8 |     88% |37, 51-56, 127, 133, 165, 170 |
| src/fdhpy/moss\_et\_al\_2024.py                     |       93 |        8 |     91% |106-107, 125, 132-133, 244, 260, 273 |
| src/fdhpy/moss\_ross\_2011.py                       |       69 |        2 |     97% |  198, 211 |
| src/fdhpy/normalized\_fault\_displacement\_model.py |       71 |        6 |     92% |41, 43-48, 96 |
| src/fdhpy/petersen\_et\_al\_2011.py                 |       81 |       13 |     84% |113-117, 130-134, 188-193, 223-226, 264, 268 |
| src/fdhpy/utils.py                                  |       77 |        5 |     94% |39-41, 62, 139 |
| src/fdhpy/youngs\_et\_al\_2003.py                   |       82 |       21 |     74% |113-117, 123-125, 152-159, 163-170, 202-205, 218, 231 |
|                                           **TOTAL** | **1371** |  **248** | **82%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/asarmy/fdhpy/coverage-data-branch/badge.svg)](https://htmlpreview.github.io/?https://github.com/asarmy/fdhpy/blob/coverage-data-branch/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/asarmy/fdhpy/coverage-data-branch/endpoint.json)](https://htmlpreview.github.io/?https://github.com/asarmy/fdhpy/blob/coverage-data-branch/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fasarmy%2Ffdhpy%2Fcoverage-data-branch%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/asarmy/fdhpy/blob/coverage-data-branch/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.