import subprocess
import os

# src = './Results/MP_Videos'
# dst = './Results/MP_Videos_MP4'

def convert_format(src, dst):
    for root, dirs, filenames in os.walk(src, topdown=False):
        #print(filenames)
        for filename in filenames:
            print('[INFO] 1',filename)
            try:
                _format = ''
                if ".flv" in filename.lower():
                    _format=".flv"
                if ".mp4" in filename.lower():
                    _format=".mp4"
                if ".avi" in filename.lower():
                    _format=".avi"
                if ".mov" in filename.lower():
                    _format=".mov"

                inputfile = os.path.join(root, filename)
                print('[INFO] 1',inputfile)
                outputfile = os.path.join(dst, filename.replace(_format, ".mp4"))
                subprocess.call(['ffmpeg', '-y', '-i', inputfile, '-c:v', 'copy', '-c:a', 'copy', outputfile])  
            except:
                print("An exception occurred")