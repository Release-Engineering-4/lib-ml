#!/bin/bash

TAG=$(git describe --tags --abbrev=0)

sed -i "s/^version = .*/version = \"$TAG\"/" pyproject.toml

echo "Updated version to: $TAG"
