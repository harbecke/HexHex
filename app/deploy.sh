#!/bin/bash

# This will deploy the app to cleeff.github.io from Pascal's machine :-)

npm run build
rm -r ~/code/cleeff.github.io/hex
cp -r build ~/code/cleeff.github.io/hex
cd ~/code/cleeff.github.io 
git add -A && git commit -m "Update hex" && git push



