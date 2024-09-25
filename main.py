from bp_calc import *
from can2dshare import *
from convert_video import *
from face_hands import *
from dbp import *


input_path = './Input_Videos/'
ppg_path_face = './Results/PPG/Face/'
ppg_path_hand = './Results/PPG/Hand/'
src_face = './Results/MP_Videos/Face'
dst_face = './Results/MP_Videos_MP4/Face'
src_hand = './Results/MP_Videos/Hand'
dst_hand = './Results/MP_Videos_MP4/Hand'
input_image_path = './Input_Photos/'
demographic_path = './Input_Data/Demographic_Data.csv'

def main(input_path, ppg_path_face, ppg_path_hand, src_face, dst_face, src_hand, dst_hand, input_image_path, demographic_path):
    for root,dirs,files in os.walk(input_path):
        videonames=[ _ for _ in files if _.endswith('.mp4') ]
        print(f'Found {len(videonames)} videos in the input folder \n')
    for i in range(0, len(videonames)):
        vidpath = os.path.join(input_path, videonames[i])
        print(vidpath)
        print('Identifying Face')
        face_capture(vidpath)
        print('Identifying Hand')
        hand_capture(vidpath)
    print('Converting videos to .mp4 format')
    convert_format(src_face, dst_face)
    convert_format(src_hand, dst_hand)
    print('Computing PPG signal')
    deep_phys(dst_face, ppg_path_face)
    deep_phys(dst_hand, ppg_path_hand)
    print('Computing Systolic Blood Pressure')
    main_bp(ppg_path_face, ppg_path_hand, input_image_path, demographic_path)
    print('Computing Diastolic Blood Pressure')
    main_dbp(ppg_path_face, ppg_path_hand, input_image_path, demographic_path)





main(input_path, ppg_path_face, ppg_path_hand, src_face, dst_face, src_hand, dst_hand, input_image_path, demographic_path)

