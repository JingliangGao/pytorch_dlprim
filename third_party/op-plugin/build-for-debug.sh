#!/bin/bash

# set variables
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

# generate backend stubs files
cd $CDIR/
OUTPUT_DIR="$CDIR/op_plugin/config"
python3 -m codegen.gen_backend_stubs  \
  --source_yaml="$OUTPUT_DIR/op_plugin_functions.yaml" \
  --output_dir="$CDIR/op_plugin/generate/"
