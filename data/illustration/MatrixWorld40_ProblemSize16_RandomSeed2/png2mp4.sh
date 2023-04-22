ffmpeg -r 2 -i MatrixWorld%04d.png -c:v libx264 -vf fps=24 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" out.mp4
