import lpd_prepare_utils as utl
import numpy as np
import os
import pypianoroll as pr
import config
# from lpd_prepare_main import RESULT_PRECISION

RESULT_PRECISION = 16
root_dir = "lpd_5_cleansed"
RESULT_STEP = int(96/RESULT_PRECISION)

file_list = utl.traverse_dir(root_dir)

all_chord_X = np.zeros([1,5,RESULT_PRECISION,84])
cnt = -1
for file_ in file_list[:]:
    cnt+=1
    if cnt>1000:
        break
    # is_final = True
    multitrack = pr.load(file_)
    # for i in range(0, 5):
    #     print(multitrack.tracks[i].name)
    #     lens = len(multitrack.tracks[i].pianoroll)
    #     print(lens)
    #     if lens == 0:
    #         is_final = False
    #         break
    # if is_final == False:
    #     continue
    # print(is_final)
    # cnt+=1
    # if cnt >= 1000:
    #     break

    # sname = str(cnt) + ".npz"
    # pr.save("list_npz/" + sname, multitrack)
    data_pr = multitrack.get_stacked_pianoroll()


    ### 对不满1小节的结尾进行padding
    if data_pr.shape[0]%96:
        data_pad = np.zeros((96 - data_pr.shape[0] % 96, 128, 5))
        data_pr = np.concatenate((data_pr, data_pad))

    ### (小节x96, 128, 5) > (小节, 96, 128, 5) > (小节，96，84，5)
    data_pr = np.reshape(data_pr, [-1,96,128,5])
    data_pr = data_pr[:, :, 24:108, :]
    # print("data_pr's shape is:", data_pr.shape)
    bar_num = data_pr.shape[0]

    chord_X = []
    # bar, 96, 84, 5  ------> 5, bar, 84, 96
    # print(data_pr.shape)
    data_pr = np.transpose(data_pr,[3,0,2,1])
    print(data_pr.shape)
    for j in range(0, 5):
        instru = []
        for i in range(data_pr.shape[1]): #bar
            bar = []
            for k in range(84): #84
                line = []
                for h in range(RESULT_PRECISION): #32
                    sum_res = 0
                    for num in range(RESULT_STEP):
                        sum_res = sum_res + data_pr[j][i][k][h*RESULT_STEP + num]
                    if sum_res == 0:
                        line.append(0)
                    else:
                        line.append(1)
                bar.append(line)
            instru.append(bar)
        chord_X.append(instru)
    chord_X = np.array(chord_X)
    chord_X = np.transpose(chord_X, [1,0,3,2])
    print("chord_X shape is:", chord_X.shape)
    print("all_chord_X shape is:", all_chord_X.shape)
    all_chord_X = np.concatenate((all_chord_X, chord_X), axis=0)
    print(cnt, '/', len(file_list), '--', file_, "Complete")

all_chord_X = np.array(all_chord_X).astype(bool)
print(all_chord_X.shape)
np.save("RES/all_chord_X_"+str(RESULT_PRECISION)+".npy", all_chord_X)