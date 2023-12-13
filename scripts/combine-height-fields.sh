#!/usr/bin/env bash

for i in {1..19}
do
  echo "hi $i"
  convert \
    /media/kennychufk/old-ubuntu/alluvion-data/val-diagonal2/rltruth-0bc4c4ca-1224.06.31.03/truth-hf$i.png -crop 756x580+134+425 \
    /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-diagonal2-0.011/val-uth-0bc4c4c-2v7m4muc-2600-036a3884/recon-hf$i.png -crop 756x580+134+425 \
    -append +repage \
    -fuzz 5% -fill none -draw "matte 0,0 floodfill" \
    -fuzz 8% -fill "#ffffff77" -draw "matte 378,542 floodfill" \
    -fuzz 8% -fill "#ffffff77" -draw "matte 378,1100 floodfill" \
    -fuzz 4% -fill "#ffffff77" -draw "matte 378,573 floodfill" \
    -fuzz 4% -fill "#ffffff77" -draw "matte 378,1152 floodfill" \
    -channel alpha -blur 0x1 -level 20%,100% +channel +repage \
    /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-diagonal2-0.011/val-uth-0bc4c4c-2v7m4muc-2600-036a3884/combined-hf$i.png
done

for i in {1..19}
do
  echo "hi $i"
  convert \
    /media/kennychufk/old-ubuntu/alluvion-data/val-bidir-circles2/rltruth-d45047d7-1223.01.00.24/truth-hf$i.png -crop 756x580+134+425 \
    /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-bidir-circles2-0.011/val-uth-d45047d-2v7m4muc-2600-97eb2dc5/recon-hf$i.png -crop 756x580+134+425 \
    -append +repage \
    -fuzz 5% -fill none -draw "matte 0,0 floodfill" \
    -fuzz 8% -fill "#ffffff77" -draw "matte 378,542 floodfill" \
    -fuzz 8% -fill "#ffffff77" -draw "matte 378,1100 floodfill" \
    -fuzz 4% -fill "#ffffff77" -draw "matte 378,573 floodfill" \
    -fuzz 4% -fill "#ffffff77" -draw "matte 378,1152 floodfill" \
    -channel alpha -blur 0x1 -level 20%,100% +channel +repage \
    /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-bidir-circles2-0.011/val-uth-d45047d-2v7m4muc-2600-97eb2dc5/combined-hf$i.png
done

for i in {1..39}
do
  footage_frame_id=`expr $i \* 15`
  echo "hi $i $footage_frame_id"
  convert \
    \(  /media/kennychufk/vol1bk0/20210416_103739/footage-frames/frame$footage_frame_id.png -crop 1020x620+275+340 \
        /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-piv-0.011/val-0416_103739-2v7m4muc-2600-685aa05c/footage-mask.png -alpha off -compose copy-opacity -composite \) \
    \(  /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-piv-0.011/val-0416_103739-2v7m4muc-2600-685aa05c/recon-hf$i.png -crop 1020x620+275+340\
        +repage \
        -fuzz 5% -fill none -draw "matte 0,0 floodfill" \
        -fuzz 5% -fill none -draw "matte 1015,615 floodfill" \
        -fuzz 8% -fill "#ffffff77" -draw "matte 432,516 floodfill" \
        -channel alpha -blur 0x1 -level 20%,100% +channel +repage \) \
    -append \
    /media/kennychufk/old-ubuntu/evaluation-results/2v7m4mucAug-val-piv-0.011/val-0416_103739-2v7m4muc-2600-685aa05c/combined-hf$i.png
done

