###
 # @Author: Conghao Wong
 # @Date: 2025-03-24 17:16:55
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2025-03-24 17:21:40
 # @Github: https://cocoon2wong.github.io
 # Copyright 2025 Conghao Wong, All Rights Reserved.
###

rm -r ./_data
rm -r ./_includes
rm -r ./_layouts
rm -r ./assets

cp -r ./Project-Zero-Divided/_data ./
cp -r ./Project-Zero-Divided/_includes ./
cp -r ./Project-Zero-Divided/_layouts ./
cp -r ./Project-Zero-Divided/assets ./

bundle exec jekyll serve