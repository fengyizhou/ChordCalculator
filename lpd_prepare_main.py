import lpd_prepare_utils as utl
import numpy as np
import os
import pypianoroll as pr
import config


RESULT_PRECISION = 16

#root_dir = '/home/miku/UserFolder/Wyz/data2midi-master/testnpz/lpd_5_cleansed'
root_dir = "lpd_5_cleansed"
# root_dir = "NPZ_VAL"

file_list = utl.traverse_dir(root_dir)

all_data = []
all_chord_chroma_Y = np.zeros([1,RESULT_PRECISION,84])
cnt = -1
for file_ in file_list[:]:

    cnt += 1
    if cnt > 1000:
        break

    multitrack = pr.load(file_)
    # if cnt>1000:
    #     break



    sname = str(cnt)+".npz"
    pr.save("list_npz/" + sname, multitrack)
    # pr.save("NPZ_VAL/midi/"+ sname, multitrack)
    data_pr = multitrack.get_stacked_pianoroll()


    ### 对不满1小节的结尾进行padding
    if data_pr.shape[0]%96:
        data_pad = np.zeros((96 - data_pr.shape[0] % 96, 128, 5))
        data_pr = np.concatenate((data_pr, data_pad))

    ### (小节x96, 128, 5) > (小节, 96, 128, 5) > (小节，96，84，5)
    data_pr = np.reshape(data_pr, (-1,96,128,5))
    data_pr = data_pr[:, :, 24:108, :]
    # data_pr = np.transpose(data_pr, [0,3,1,2])
    # print("data_pr's shape is:", data_pr.shape)
    bar_num = data_pr.shape[0]

    ### 弄四个新条件出来，分别是bar_id [0,1,2,...,小节数-1],  bar_pos [0, 0.07, 0.12  ...],
    ### bar_seg[1,0,0,...,0,0,1]，bar_tra_sta,[[1,0,0,0,1],[1,1,0,1,0],...]
    bar_id = np.arange(bar_num)
    bar_pos = np.array([round(id/bar_id[-1],2) for id in bar_id])
    bar_seg = np.zeros(bar_num).astype(np.bool)
    bar_seg[0]=True
    bar_seg[-1]=True
    bar_tra_sta = []
    for bar in data_pr:
        state_this_bar = [1 if np.sum(bar[:,:,i])>0 else 0 for i in range(data_pr.shape[-1])]
        state_this_bar = np.array(state_this_bar)
        #print(state_this_bar.shape)
        bar_tra_sta.append(state_this_bar)

    bar_tra_sta = np.stack(bar_tra_sta)
    bar_cho_li = []

    song_chroma_left, song_chroma_right = utl.get_chroma(data_pr)


    data_tone_left = utl.get_scale_seq(song_chroma_left)
    data_tone_right = utl.get_scale_seq(song_chroma_right)
    bar_cho_li_left = utl.get_bar_chord7_li(song_chroma_left, data_tone_left)
    bar_cho_li_right = utl.get_bar_chord7_li(song_chroma_right, data_tone_right)
    # for i in range(0,len(bar_cho_li)):
    #     print(i+1 , ":", bar_cho_li[i])
    # print(len(bar_cho_li_left))

    #### genarate trainning Y
    # print(bar_cho_li_left, bar_cho_li_right)
    chord_chroma_left = utl.generate_chord7_chroma_32(bar_cho_li_left)
    chord_chroma_right = utl.generate_chord7_chroma_32(bar_cho_li_right)
    chord_chroma_left = np.array(chord_chroma_left).astype(bool)
    chord_chroma_right = np.array(chord_chroma_right).astype(bool)
    chord_chroma_all = np.concatenate((chord_chroma_left,chord_chroma_right), axis=1)
    all_chord_chroma_Y = np.concatenate((all_chord_chroma_Y, chord_chroma_all), axis=0)
    print(all_chord_chroma_Y.shape)

    for i in range(len(bar_cho_li_left)):
        bar_cho_li_one = [bar_cho_li_left[i], bar_cho_li_right[i]]
        bar_cho_li.append(bar_cho_li_one)

    all_data.append(bar_cho_li)
    bar_cho_li = np.array(bar_cho_li)

    np.save("list_chord/"+str(cnt)+".npy", bar_cho_li)
    # for bar in bar_cho_li:
     #   print(bar[0], bar[1])
    print(cnt,'/',len(file_list),'--',file_, "Complete")

all_chord_chroma_Y = np.array(all_chord_chroma_Y)
print(all_chord_chroma_Y.shape)
all_chord_chroma_Y = all_chord_chroma_Y.reshape([-1, 1, RESULT_PRECISION, 84])
# np.save("NPZ_VAL/velo_.npy",all_chord_chroma_Y)


np.save("RES/all_chord_chroma_Y7_"+ str(RESULT_PRECISION)+".npy", all_chord_chroma_Y)
print("RES/all_chord_chroma_Y7_"+ str(RESULT_PRECISION)+".npy saved!!!!!!!!!!!!!")
all_data = np.array(all_data)
np.save("RES/All_Chord.npy", all_data)
    #########
    #未完待续...
