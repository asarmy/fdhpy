# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/asarmy/fdhpy/blob/coverage-data-branch/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/fdhpy/\_\_init\_\_.py                           |       15 |        2 |     87% |     26-27 |
| src/fdhpy/chiou\_et\_al\_2025.py                    |       94 |       11 |     88% |100, 107-108, 137-140, 204-207, 251, 255 |
| src/fdhpy/cli.py                                    |       51 |       47 |      8% |    29-114 |
| src/fdhpy/fault\_displacement\_model.py             |      103 |       18 |     83% |76, 79-93, 99, 148, 194, 240, 288 |
| src/fdhpy/fault\_displacement\_model\_variables.py  |       65 |        7 |     89% |101-107, 157-158, 182-183 |
| src/fdhpy/kuehn\_et\_al\_2024.py                    |      224 |       48 |     79% |120-121, 138-145, 174-175, 179-192, 213-239, 245, 251, 307-310, 489-495, 525-528, 553, 558-563, 642, 651, 655 |
| src/fdhpy/lavrentiadis\_abrahamson\_2023.py         |      291 |       52 |     82% |118-119, 138, 141-142, 152-159, 172, 227-230, 251-254, 333-336, 374-377, 406, 426-429, 479-480, 600, 618-628, 638-639, 676, 687, 691 |
| src/fdhpy/loglinear\_scaling\_model.py              |       67 |        8 |     88% |37, 51-56, 127, 133, 165, 170 |
| src/fdhpy/moss\_et\_al\_2024.py                     |      106 |        9 |     92% |112-113, 131, 138-139, 271-283, 297, 310 |
| src/fdhpy/moss\_ross\_2011.py                       |       76 |        2 |     97% |  210, 223 |
| src/fdhpy/normalized\_fault\_displacement\_model.py |       71 |        6 |     92% |44, 46-51, 99 |
| src/fdhpy/petersen\_et\_al\_2011.py                 |       81 |       13 |     84% |113-117, 130-134, 188-193, 223-226, 264, 268 |
| src/fdhpy/utils.py                                  |       77 |        6 |     92% |39-41, 62, 83, 139 |
| src/fdhpy/youngs\_et\_al\_2003.py                   |       87 |       21 |     76% |121-125, 131-133, 162-169, 173-180, 212-215, 228, 241 |
| **TOTAL**                                           | **1408** |  **250** | **82%** |           |


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