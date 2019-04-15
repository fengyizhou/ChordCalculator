import numpy as np
import os
import pypianoroll as pr
import config
# from lpd_prepare_main import RESULT_PRECISION

RESULT_PRECISION = 16
# root_dir = "lpd_5_cleansed"
RESULT_STEP = int(96/RESULT_PRECISION)
RESULT_BAR_HALF = int(RESULT_PRECISION/2)

######################
#贯穿根目录，拿到所有.npz文件的路径
######################
def traverse_dir(root_dir, extension = '.npz', ):
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        print(root)
        for file in files:
            if file.endswith(extension):
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))

    with open('your_file.txt', 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

    return file_list


######################
#输入一首歌的piano roll，按小节的一半计算chroma，返回一个小节前半小节和后半小节的chroma
######################
def get_chroma(data_pr, d_type='int'):
    '''

    :param data_pr:  (bar_num, in_bar_step, pitch, track_num) e.g. (bar, 96, 84, 5)
                    track--['Drums', 'Piano', 'Guitar', 'Bass', 'Strings']
    :return: data_chroma: (bar_num, in_bar_step, pitch_class) e.g. (bar, 96, 12)
    '''
    pitch_num = data_pr.shape[2]
    track_num = data_pr.shape[3]
    assert pitch_num == 84
    assert track_num == 5

    if d_type == 'bool':

        ### data_pr (bar_num, 96, 84, 5) > (bar_num, 96, 84)
        data_union = np.logical_or.reduce(
            [data_pr[:, :, :, 1], data_pr[:, :, :, 2], data_pr[:, :, :, 3], data_pr[:, :, :, 4]])

        ### data_pr (bar_num, 96, 84) > left:(bar_num, 96/2, 84) right:(bar_num, 96/2, 84)
        data_left = data_union[:, 0:48, 84]
        data_right = data_union[:, 48:96, 84]

        ### (bar_num, 96/2, 84) > (bar_num, 96/2, 12)
        data_chroma_left = np.logical_or.reduce([data_left[:, :, :12], data_left[:, :, 12:24], data_left[:, :, 24:36],
                                                 data_left[:, :, 36:48], data_left[:, :, 48:60], data_left[:, :, 60:72],
                                                 data_left[:, :, 72:84]])
        data_chroma_right = np.logical_or.reduce([data_right[:, :, :12], data_right[:, :, 12:24], data_right[:, :, 24:36],
                                                  data_right[:, :, 36:48], data_right[:, :, 48:60], data_right[:, :, 60:72],
                                                  data_right[:, :, 72:84]])

    elif d_type == 'int':
        data_union = np.add.reduce([data_pr[:, :, :, 1], data_pr[:, :, :, 2], data_pr[:, :, :, 3], data_pr[:, :, :, 4]])

        data_left = data_union[:, 0:48, :]
        data_right = data_union[:, 48:96, :]

        # data_chroma = np.add.reduce([data_union[:, :, :12], data_union[:, :, 12:24], data_union[:, :, 24:36],
        #                              data_union[:, :, 36:48], data_union[:, :, 48:60], data_union[:, :, 60:72],
        #                              data_union[:, :, 72:84]])
        data_chroma_left = np.add.reduce([data_left[:, :, :12], data_left[:, :, 12:24], data_left[:, :, 24:36],
                                          data_left[:, :, 36:48], data_left[:, :, 48:60], data_left[:, :, 60:72],
                                          data_left[:, :, 72:84]])
        data_chroma_right = np.add.reduce([data_right[:, :, :12], data_right[:, :, 12:24], data_right[:, :, 24:36],
                                           data_right[:, :, 36:48], data_right[:, :, 48:60], data_right[:, :, 60:72],
                                           data_right[:, :, 72:84]])

    else:
        assert 1 == 0, 'd_type input wrong'

    # (bar_num, 96, 12)
    return data_chroma_left, data_chroma_right



######################
#scale survival算法,输入一首歌的chroma，输出各小节调号
######################

def get_scale_seq(data_chroma):

    '''

    :param data_chroma: (bar, step, 12),dytpe ='int'
    :return: data_tone: (bar, ), 取值[0~23]，代表24个调式
    '''

    tone_mtx = config.TONE_MTX
    tone_sur_all=[]
    tone_sur_old=[]

    tone_change_bar_id = []
    for bar_id, bar_chroma in enumerate(data_chroma):
        #bar_chroma (96,12) > tone_bar (12,)
        step = bar_chroma.shape[0]
        tone_bar = np.add.reduce([bar_chroma[i] for i in range(step)])

        #### tone mtx (24, 7) 24~调式数，7~调式音阶id
        ### tone bar (12,) 12音名时长-小节
        ### tone_bar (24, ) 24调式包含时长-小节
        tone_bar = np.array([np.sum(tone_bar[tone_mtx[tone]]) for tone in range(len(tone_mtx))])

        tone_this_bar = np.where(tone_bar >= 0.9*np.max(tone_bar))[0]

        if np.sum(tone_sur_old) == 0:
            tone_sur = tone_this_bar
        else:
            # 通过与运算计算出满足更多小节的调号
            tone_sur = np.array(list(set(tone_this_bar)&set(tone_sur_old)))
            if np.sum(tone_sur) ==0 :
                tone_sur = tone_this_bar
                tone_change_bar_id.append(bar_id)

        tone_sur_old = tone_sur
        tone_sur_all.append(tone_sur)

    ########## 分段保存调号
    data_tone = np.zeros(data_chroma.shape[0])
    o_id = 0
    for c_id in tone_change_bar_id:
        data_tone[o_id:c_id] = tone_sur_all[c_id-1][0]   #保留存在可能性的调号中的第一个
        o_id = c_id

    data_tone[o_id:] = tone_sur_all[-1][0]

    return data_tone

############################
#输入一首歌的chroma，调号，输出各小节和弦
############################

def get_bar_chord_li(data_chroma, data_tone):
    '''

    :param data_chroma: (bar,step,12),dtype='int'
    :param data_tone: （bar，）  取值[0~23]，代表24个调式
    :return: (bar, step, 12)
    '''

    # list of dict, [{'C_M_':[0,4,7], 'D_m_':[1,5,8],...,'G_MS':[]},  {}, {}, ..., ]
    tone_chor_li = config.TONE_CHOR_LI

    # 关系调list
    relate_tone_li = [0,1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23]

    #[0,1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23]


    bar_chor_li = []

    for bar_id, bar in enumerate(data_chroma):

        # (step, 12)
        chroma_this_bar = data_chroma[bar_id]

        # (step, 12) > (step, ) > scaler  本小节多少个step有声音 > ratio 本小节多少比例的step有声音
        step_voice_num= np.sum(np.add.reduce([chroma_this_bar[:,pitch] for pitch in range(chroma_this_bar.shape[-1])]).astype(np.bool))
        step_voice_rate = step_voice_num/chroma_this_bar.shape[0]

        # 如果有声的step不到30%，就认为该小节chor为'na'
        if step_voice_rate < 0.3:
            chor_this_bar = 'na'
            bar_chor_li.append(chor_this_bar)
            continue

        # (12,) 各音名该小节的时长
        pitch_add_up = np.add.reduce([chroma_this_bar[step] for step in range(chroma_this_bar.shape[0])])


        # 根据该小节tone，对应的关系和弦，去匹配该小节音名时长，取匹配时长最高的和弦作为本小节和弦。
        # 若该小节tone对应的关系和弦，没有匹配度>0.45的，则从其最邻近的关系tone依次寻找
        tone = data_tone[bar_id]
        tone_id = relate_tone_li.index(tone)
        i_while = 0
        while True:

            tone = relate_tone_li[tone_id]
            # {'C_M_':[0,4,7], 'D_m_':[1,5,8],...,'G_MS':[]}
            chor_dic = tone_chor_li[tone]
            max_chor_time = 0
            for chor, chor_pitch_li in chor_dic.items():

                max_chor_time_new = np.sum(pitch_add_up[chor_pitch_li])

                if max_chor_time_new > max_chor_time:
                    max_chor_time = max_chor_time_new
                    chor_this_bar = chor

            if (max_chor_time/np.sum(pitch_add_up)) > 0.45:
                break
            elif i_while >12:
                break
            tone_id = (tone_id + 1) % 12
            i_while += 1
        bar_chor_li.append(chor_this_bar)




    return bar_chor_li

#### 明显是E的地方变成了Am 因为后一小节有个音提前了一点
def get_bar_chord7_li(data_chroma, data_tone):
    '''

    :param data_chroma: (bar,step,12),dtype='int'
    :param data_tone: （bar，）  取值[0~23]，代表24个调式
    :return: (bar, step, 12)
    '''

    # list of dict, [{'C_M_':[0,4,7], 'D_m_':[1,5,8],...,'G_MS':[]},  {}, {}, ..., ]
    tone_chor_li = config.TONE_CHOR_LI

    # 关系调list
    relate_tone_li = [0,1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23]

    #[0,1,2,3,4,5,6,7,8,9,10,11, 12,13,14,15,16,17,18,19,20,21,22,23]


    bar_chor_li = []

    for bar_id, bar in enumerate(data_chroma):

        # (step, 12)
        chroma_this_bar = data_chroma[bar_id]

        # (step, 12) > (step, ) > scaler  本小节多少个step有声音 > ratio 本小节多少比例的step有声音
        step_voice_num= np.sum(np.add.reduce([chroma_this_bar[:,pitch] for pitch in range(chroma_this_bar.shape[-1])]).astype(np.bool))
        step_voice_rate = step_voice_num/chroma_this_bar.shape[0]

        # 如果有声的step不到30%，就认为该小节chor为'na'
        if step_voice_rate < 0.3:
            chor_this_bar = 'na'
            bar_chor_li.append(chor_this_bar)
            continue

        # (12,) 各音名该小节的时长
        pitch_add_up = np.add.reduce([chroma_this_bar[step] for step in range(chroma_this_bar.shape[0])])


        # 根据该小节tone，对应的关系和弦，去匹配该小节音名时长，取匹配时长最高的和弦作为本小节和弦。
        # 若该小节tone对应的关系和弦，没有匹配度>0.45的，则从其最邻近的关系tone依次寻找
        tone = data_tone[bar_id]
        tone_id = relate_tone_li.index(tone)
        i_while = 0
        while True:

            tone = relate_tone_li[tone_id]
            # {'C_M_':[0,4,7], 'D_m_':[1,5,8],...,'G_MS':[]}
            chor_dic = tone_chor_li[tone]
            max_chor_time = 0
            for chor, chor_pitch_li in chor_dic.items():

                max_chor_time_new = np.sum(pitch_add_up[chor_pitch_li])

                if max_chor_time_new > max_chor_time:
                    max_chor_time = max_chor_time_new
                    chor_this_bar = chor

            if (max_chor_time/np.sum(pitch_add_up)) > 0.45:
                break
            elif i_while >12:
                break
            tone_id = (tone_id + 1) % 12
            i_while += 1

        print(chor_this_bar)


        ########### CALC 七和弦
        root_pitch = chor_this_bar[0:2]
        chord_plus = chor_this_bar[2]
        for i in range(len(config.PITCH_CLASS_LI)):
            if config.PITCH_CLASS_LI[i] == root_pitch:
                index_pit = i

        ###index_pit 为根音在表中的位置 S=11 s=10  pitch_add_up (12, )为该小节各个音高的时长
        s_index = (index_pit + 10) % 12
        S_index = (index_pit + 11) % 12

        chord_steps = config.THIRD_CHOR_PLUS[chord_plus]

        all_pitch_time = 0
        all_pitch_time += pitch_add_up[index_pit]
        all_pitch_time += pitch_add_up[(index_pit + chord_steps[1]) % 12]
        all_pitch_time += pitch_add_up[(index_pit + chord_steps[2]) % 12]

        max = 0
        s_of_pitch = pitch_add_up[s_index] / (all_pitch_time + pitch_add_up[s_index])
        S_of_pitch = pitch_add_up[S_index] / (all_pitch_time + pitch_add_up[S_index])

        tail = '_'
        if s_of_pitch > S_of_pitch:
            max = s_of_pitch
            tail = 's'
        else:
            max = S_of_pitch
            tail = 'S'

        if max >= 0.25:
            chor_this_bar = chor_this_bar + tail
        else:
            chor_this_bar = chor_this_bar + '_'
            

        print(chor_this_bar)

        bar_chor_li.append(chor_this_bar)




    return bar_chor_li

def generate_chord_chroma_32(data_chord):
    # print(data_chroma.shape) # bar_num, 48, 12
    PITCH_CLASS_LI = [ "A_", "A#", "B_", "C_", "C#", "D_", "D#", "E_", "F_", "F#", "G_", "G#"]
    THIRD_CHOR_PLUS = {"M": [0, 4, 7], "m": [0, 3, 7], "a": [0, 4, 8], "d": [0, 3, 6]}
    chord_chroma_32_li = []
    for i in range(len(data_chord)):
        data_chord_chroma = np.zeros([RESULT_BAR_HALF, 84])
        if data_chord[i] == "na":
            chord_chroma_32_li.append(data_chord_chroma)
            continue

        else:
            data_slice = np.zeros([RESULT_BAR_HALF, 24])
            data_slice_trans = np.zeros([RESULT_BAR_HALF, 24])
            pit = data_chord[i][0:2]
            plus = data_chord[i][2]
            index_pit = -1

            chord_steps = THIRD_CHOR_PLUS[plus]
            # print(chord_steps)
            for i in range(len(PITCH_CLASS_LI)):
                if PITCH_CLASS_LI[i] == pit:
                    index_pit = i
            data_slice[:, index_pit] = 1
            data_slice[:, index_pit + chord_steps[1]] = 1
            data_slice[:, index_pit + chord_steps[2]] = 1

            data_chord_chroma[:,21:45] = data_slice
            # print("data_chord_shape is:", data_chord_chroma.shape)
            chord_chroma_32_li.append(data_chord_chroma)

    return chord_chroma_32_li



def generate_chord7_chroma_32(data_chord):
    # print(data_chroma.shape) # bar_num, 48, 12
    PITCH_CLASS_LI = [ "A_", "A#", "B_", "C_", "C#", "D_", "D#", "E_", "F_", "F#", "G_", "G#"]
    THIRD_CHOR_PLUS = {"M": [0, 4, 7], "m": [0, 3, 7], "a": [0, 4, 8], "d": [0, 3, 6]}
    SEVEN_CHOR_PLUS = {"S": [0, 11], "s":[0, 10]}
    chord_chroma_32_li = []
    for i in range(len(data_chord)):
        data_chord_chroma = np.zeros([RESULT_BAR_HALF, 84])
        if data_chord[i] == "na":
            chord_chroma_32_li.append(data_chord_chroma)
            continue

        else:
            data_slice = np.zeros([RESULT_BAR_HALF, 24])
            data_slice_trans = np.zeros([RESULT_BAR_HALF, 24])
            pit = data_chord[i][0:2]
            plus = data_chord[i][2]
            seven = data_chord[i][3]

            index_pit = -1

            chord_steps = THIRD_CHOR_PLUS[plus]
            # print(chord_steps)
            for i in range(len(PITCH_CLASS_LI)):
                if PITCH_CLASS_LI[i] == pit:
                    index_pit = i
            data_slice[:, index_pit] = 1
            data_slice[:, index_pit + chord_steps[1]] = 1
            data_slice[:, index_pit + chord_steps[2]] = 1

            if seven == 's':
                data_slice[:, index_pit + 10] = 1
            elif seven == 'S':
                data_slice[:, index_pit + 11] = 1


            data_chord_chroma[:,21:45] = data_slice
            # print("data_chord_shape is:", data_chord_chroma.shape)
            chord_chroma_32_li.append(data_chord_chroma)

    return chord_chroma_32_li



