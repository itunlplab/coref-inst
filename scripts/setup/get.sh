#!/bin/bash
#
# CorefUD Dataset Download Script  
# Downloads and sets up the CorefUD 1.2 dataset for coreference resolution
#
# Original script from CorPipe <https://github.com/ufal/crac2023-corpipe>
# Copyright 2023 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Usage: Run from scripts/setup/ directory
#

# Output path (relative to scripts/setup/)
output_path="../../data/raw"

echo "Downloading CorefUD 1.2 dataset..."
echo "Output directory: $output_path"

# Create output directory
mkdir -p "$output_path"

# Download CorefUD dataset
echo "Downloading CorefUD-1.2-public.zip..."
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5478/CorefUD-1.2-public.zip

if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download CorefUD dataset"
    exit 1
fi

# Move and extract dataset
echo "Moving and extracting dataset..."
mv CorefUD-1.2-public.zip "$output_path"
cd "$output_path"
unzip -q CorefUD-1.2-public.zip

if [[ $? -eq 0 ]]; then
    echo "✓ CorefUD dataset successfully downloaded and extracted"
    echo "Location: $(pwd)"
    echo ""
    echo "Available languages:"
    ls -1 CorefUD-1.2-public/ | head -10
    if [[ $(ls CorefUD-1.2-public/ | wc -l) -gt 10 ]]; then
        echo "... and $(( $(ls CorefUD-1.2-public/ | wc -l) - 10 )) more"
    fi
    
    # Clean up zip file
    rm CorefUD-1.2-public.zip
    echo ""
    echo "Next step: Process the dataset using:"
    echo "cd ../../src/data_processing/format_dataset/"
    echo "bash dataset.sh"
else
    echo "✗ Failed to extract dataset"
    exit 1
fi

for f in CorefUD-1.2-public/data/*/*.conllu; do
  lang=$(basename $f)
  lang=${lang%%-*}
  mkdir -p $lang
  mv $f $lang/$(basename $f)
done
rm -r CorefUD-1.2-public/ CorefUD-1.2-public.zip

# # Test data
# mkdir test
# (cd test
#  wget https://ufal.mff.cuni.cz/~mnovak/files/corefud-1.2/test-blind.zip
#  unzip test-blind.zip
#  for f in *.conllu; do
#    lang=${f%%-*}
#    mv $f ../$lang/$f
#  done
# )
# rm -r test/

# # Data cleanup
# sed 's/20.1	trabajó	trabajar/20.1	_	_/' -i es_ancora/es_ancora-corefud-train.conllu
