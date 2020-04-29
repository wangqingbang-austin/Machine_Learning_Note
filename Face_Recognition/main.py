import base64
from PIL import Image
import cv2
from face_recogination import recognize, voice, register
from io import BytesIO


def frame2base64(frame):
    img = Image.fromarray(frame)  # 将每一帧转为Image
    output_buffer = BytesIO()  # 创建一个BytesIO
    img.save(output_buffer, format='JPEG')  # 写入output_buffer
    byte_data = output_buffer.getvalue()  # 在内存中读取
    base64_data = base64.b64encode(byte_data)  # 转为BASE64
    return base64_data  # 转码成功 返回base64编码

def face_reco():
    voice("开始人脸识别，先用鼠标点一下画面，再按键盘上的回车键进行识别！按ESC键退出！")
    print("先用鼠标点一下画面，之后按键盘上的回车键进行识别！按ESC键退出！")
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        cv2.imshow('Face detection system! Press ESC to exit!', frame)
        flag = cv2.waitKey(1)
        if flag == 13:  # 按下回车键
            image_64 = frame2base64(frame)
            recognize(image_64)
        if flag == 27:  # 按下ESC键
            break

def regist():
    voice("请在输入框内输入你的姓名！")
    name = input("请在输入框内输入你的姓名：")
    info = '姓名：'+ str(name) + '；权限等级：高级用户！'
    cap = cv2.VideoCapture(0)
    voice("请在打开的摄像画面中，先用鼠标点一下画面，再按下回车键，注册人脸信息！")
    while 1:
        ret, frame = cap.read()
        cv2.imshow('Face detection system! Press ESC to exit!', frame)
        flag = cv2.waitKey(1)
        if flag == 13:  # 按下回车键
            image_64 = frame2base64(frame)
            register(image_64, info)
            break

if __name__ == '__main__':
    voice("在命令框中输入数字0，进行人脸注册。在命令框中输入数字1，进行人脸识别。")
    print("体验版本V1.0，制作人：王庆棒")
    choice = input("在命令框中输入数字0，进行人脸注册。在命令框中输入数字1，进行人脸识别。")
    print(choice)
    if choice == '0':
        regist()
    else:
        face_reco()
