#!/bin/bash

# Install label-studio if not already installed
uv tool install label-studio

# Initialize label-studio (first time only)
label-studio init gmsk8_annotation

# Start label-studio and import your data
label-studio start gmsk8_annotation