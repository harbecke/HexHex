#!/bin/bash

npm run build
rm -r ~/code/cleeff.github.io/hex
cp -r build ~/code/cleeff.github.io/hex
