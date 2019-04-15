import numpy as np

PITCH_CLASS_LI = ["C_", "C#", "D_", "D#", "E_", "F_", "F#", "G_", "G#", "A_", "A#", "B_"]
THIRD_CHOR_PLUS = {"M": [0, 4, 7], "m": [0, 3, 7], "a": [0, 4, 8], "d": [0, 3, 6]}
SEVEN_CHOR_PLUS = {"S": [0, 11], "s": [0, 10]}

def cal_tone_mtx():
    cal_tone_mtx = np.zeros(shape=(24,7))
    #c major = [0, 2, 4, 5, 7, 9, 11]
    #c harmonic minor = [0, 2, 3, 5, 7, 8, 11]
    for i_1 in range(12):
        cal_tone_mtx[i_1] = [(0+i_1)%12, (2+i_1)%12, (4+i_1)%12, (5+i_1)%12, (7+i_1)%12, (9+i_1)%12, (11+i_1)%12]
    for i_2 in range(12, 24):
        a = i_2 - 12
        cal_tone_mtx[i_2] = [(0+a)%12, (2+a)%12, (3+a)%12, (5+a)%12, (7+a)%12, (8+a)%12, (11+a)%12]
    return cal_tone_mtx.astype(np.int64)

TONE_MTX = cal_tone_mtx()
print(TONE_MTX)

def cal_tone_chor_li(tone_mtx):

    '''
    计算0~23个调式，每个调式对应的关系和弦名（从主和弦开始）及其对应的音名index_list
    :return: list of dict, 例如: list[0]=={'C_M_':[0,4,7], 'D_m_':[1,5,8],...,'G_MS':[]}
    '''
    tone_chor_mtx = []
    for scale_ in tone_mtx:
        major = np.array([0, 2, 4])
        chor_index_li = [scale_[(major+i)%7] for i in (0, 3, 4, 5, 1, 2, 6)]
        tone_chor_mtx.append(chor_index_li)
    tone_chor_li = []
    for tone_chor_ in tone_chor_mtx:
        tone_chor_dict = {}
        for chor_ in tone_chor_:
            base = chor_[0]
            for k,v in THIRD_CHOR_PLUS.items():
                if ((v+base)%12 == chor_).all():
                    tone_chor_dict[PITCH_CLASS_LI[base] + k] = chor_
        tone_chor_li.append(tone_chor_dict)


    return tone_chor_li


TONE_CHOR_LI = cal_tone_chor_li(TONE_MTX)

print(TONE_CHOR_LI)