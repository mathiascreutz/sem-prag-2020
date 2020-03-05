#!/bin/bash

cd /home/jovyan/

rm -r work

export GIT_COMMITTER_NAME=anonymous
export GIT_COMMITTER_EMAIL=anon@localhost
git clone https://github.com/mathiascreutz/sem-prag-2020.git

pip install nltk sklearn
python -c "import nltk; nltk.download(['gutenberg', 'genesis', 'treebank', 'nps_chat', 'inaugural', 'webtext', 'wordnet'])"
