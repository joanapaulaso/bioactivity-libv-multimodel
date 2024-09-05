#!/bin/bash

# Set up Git remotes (only if you need to change the URL or add a new one)
# git remote set-url origin https://github.com/joanapaulaso/bioatividade_LIBV.git
# git remote add new-remote-name https://github.com/joanapaulaso/bioatividade_LIBV.git

# Verify remotes
git remote -v

# Set Git user (optional if not already set globally)
git config --global user.email "joanapaulasoliveira@gmail.com"
git config --global user.name "Joana Paula Oliveira"

# Check status
git status

# Add all changes
git add .

# Commit with a message provided as an argument
git commit -m "$1"

# Push to the main branch
git push origin main
