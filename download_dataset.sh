#!/bin/bash
kaggle competitions download -c gan-getting-started -p data/raw/
unzip data/raw/gan-getting-started.zip -d data/raw/
rm data/raw/gan-getting-started.zip
