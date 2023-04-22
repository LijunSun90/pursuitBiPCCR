ffmpeg -f image2 -i MatrixWorld%04d.png video.avi
ffmpeg -i video.avi -pix_fmt rgb24 -loop 0 out.gif
