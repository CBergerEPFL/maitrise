#!/bin/bash

pdm --pep582 >> ~/.bash_profile
eval "$(pdm --pep582)"

pdm install

## Welcome message

echo "d88888b db    db d888888b d88888b d8b   db .d8888. d888888b  .d88b.  d8b   db .d8888. "
echo "88'     '8b  d8' '~~88~~' 88'     888o  88 88'  YP   '88'   .8P  Y8. 888o  88 88'  YP "
echo "88ooooo  '8bd8'     88    88ooooo 88V8o 88 '8bo.      88    88    88 88V8o 88 '8bo.   "
echo "88~~~~~  .dPYb.     88    88~~~~~ 88 V8o88   'Y8b.    88    88    88 88 V8o88   'Y8b. "
echo "88.     .8P  Y8.    88    88.     88  V888 db   8D   .88.   '8b  d8' 88  V888 db   8D "
echo "Y88888P YP    YP    YP    Y88888P VP   V8P '8888Y' Y888888P  'Y88P'  VP   V8P '8888Y' "
echo
echo "01100101 01111000 01110100 01100101 01101110 01110011 01101001 01101111 01101110 01110011"
echo
echo "FOR OPTIMAL EXPERIENCE -> INSTALL RECOMMENDED EXTENSIONS"
