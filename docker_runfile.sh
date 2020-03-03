docker run \
    -it \
    --ipc="host" \
    --rm \
    --env="DISPLAY" \
    --runtime=nvidia \
    -v /media/data/bengaliai-cv19/cropped:/app/data/graphemes \
    -v /media/data/bengaliai-cv19/256_png:/app/data/256_png \
    -v /home/mace/paper_implementations/bai19:/app \
    pytorch python /app/run.py