import os
from aip import AipFace
from playsound import playsound
import time
from aip import AipSpeech

""" 语音合成的APPID AK SK """
APP_ID1 = '19639769'
API_KEY1 = 'tnYotgzVfD6FgrHn00WEdEGX'
SECRET_KEY1 = '5Lgad38gysZY3ynUcpxYkexa0X4ku8DH'

client_voice = AipSpeech(APP_ID1, API_KEY1, SECRET_KEY1)


def voice(info):
    voice_result = client_voice.synthesis(info, 'zh', 1, {
        'vol': 5,
        'per': 0,
    })

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(voice_result, dict):
        with open('auido.mp3', 'wb') as f:
            f.write(voice_result)
            f.close()
    playsound('auido.mp3')
    os.remove('auido.mp3')


# 定义常量
APP_ID = '19640254'
API_KEY = '0Qc1wuVuNq6z7pfgxCLCp8EW '
SECRET_KEY = '6rDdvxlZ4fQKWGZ08tYMeVpW6gvGFm67'
imageType = "BASE64"
groupId = "test"
# 初始化AipFace对象
client = AipFace(APP_ID, API_KEY, SECRET_KEY)


def recognize(image):
    image64 = str(image, 'UTF-8')
    # 调用人脸属性检测接口
    options = {}
    options["max_face_num"] = 2
    """ 调用人脸搜索 """
    result = client.search(image64, imageType, groupId, options);
    if result['result'] is not None:
        score = result['result']['user_list'][0]['score']
        info = result['result']['user_list'][0]['user_info']
    else:
        voice('没有检测到人脸！请重新验证！')
        return

    print('系统识别得分：', score)
    print('用户信息', info)

    if score > 80:
        voice('身份验证成功！{}'.format(info))
    else:
        voice('身份验证失败！请重新验证！')


def register(image, info="姓名：王庆棒； 系统权限等级：最高权限。"):
    image64 = str(image, 'UTF-8')

    userId = 'user' + str(int(time.time()))
    """ 如果有可选参数 """
    options = {}
    options["user_info"] = str(info.encode(), encoding='Latin-1')
    options["quality_control"] = "NORMAL"
    options["liveness_control"] = "LOW"
    options["action_type"] = "REPLACE"
    """ 带参数调用人脸更新 """
    client.addUser(image64, imageType, groupId, userId, options)
    voice('面部信息输入成功！打开程序输入1，快来人脸认证试试吧！')
