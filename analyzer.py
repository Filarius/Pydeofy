from ffmpegwrapper import ffmpeg
import numpy as np
path = 'ffmpeg -loglevel panic -hide_banner -y -i bad{0}.mp4 -f image2pipe -pix_fmt gray -c:v rawvideo -'



frame_cnt = 200
videocount = 18


for i in range(videocount-1):
    s = path.format(i)
    ff1 = ffmpeg(s,use_stdout=True,use_stderr=False)
    s = path.format(i+1)
    ff2 = ffmpeg(s, use_stdout=True)
    ff1.start()
    ff2.start()
    mean = 0
    for j in range(frame_cnt):
        ar1 = ff1.readout(1920*1080).astype(np.int16)
        ar2 = ff2.readout(1920*1080).astype(np.int16)
        mean += np.mean(np.abs(ar1-ar2))

    mean = mean / frame_cnt
    print(i,mean)



