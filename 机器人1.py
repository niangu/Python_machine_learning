import itchat
itchat.login()
  # 这里需要你人工手机扫码登录
itchat.send('Hello, 文件助手', toUserName='filehelper')

import itchat
@itchat.msg_register(itchat.content.TEXT)
def text_replay(msg):
    return msg.text  # 人家说啥你回复啥。。。
itchat.auto_login()
itchat.run()

import itchat
import time

itchat.auto_login(hotReload=True)

SINCERE_WISH = u"祝%s端午节快乐！"

friend_list = itchat.get_friends(update=True)  # 第一个是自己
friend_list = friend_list[1:]

for friend in friend_list:
    # 如果是演示，把send改成print就行
    # itchat.send(SINCERE_WISH % (friend['DisplayName'] or friend['NickName']), friend['UserName'])
    print(SINCERE_WISH % (friend['DisplayName'] or friend['NickName']))
    time.sleep(3)

"""有时候我们会想知道某个好友有没有删除自己或者把自己拉入黑名单。

这一操作使用itchat也会变的非常简单。

原理的话，在于将好友拉入群聊时，非好友和黑名单好友不会被拉入群聊。

所以群聊的返回值中就有了好友与你关系的数据。

另外，群聊在第一次产生普通消息时才会被除创建者以外的人发现的（系统消息不算普通消息）。

这样，就可以隐蔽的完成好友检测
"""


#coding=utf8
import itchat

CHATROOM_NAME = 'friend'
CHATROOM = None
HELP_MSG = u'''\
好友状态监测
* 发送名片将会返回好友状态
* 请确有名为%s的未使用的群聊
* 并将该群聊保存到通讯录
* 调用频率存在一定限制\
''' % CHATROOM_NAME
CHATROOM_MSG = u'''\
无法自动创建群聊，请手动创建
确保群聊名称为%s
请不要使用已经使用过的群聊
创建后请将群聊保存到通讯录\
''' % CHATROOM_NAME


def get_chatroom():
    global CHATROOM
    if CHATROOM is None:
        itchat.get_chatrooms(update=True)
        chatrooms = itchat.search_chatrooms(CHATROOM_NAME)
        if chatrooms:
            return chatrooms[0]
        else:
            r = itchat.create_chatroom(itchat.get_friends()[1:4], topic=CHATROOM_NAME)
            if r['BaseResponse']['ErrMsg'] == '':
                CHATROOM = {'UserName': r['ChatRoomName']}
                return CHATROOM
    else:
        return CHATROOM
def get_friend_status(friend):
    ownAccount = itchat.get_friends(update=True)[0]
    if friend['UserName'] == ownAccount['UserName']:
        return u'检测到本人账号。'
    elif itchat.search_friends(userName=friend['UserName']) is None:
        return u'该用户不在你的好友列表中。'
    else:
        chatroom = CHATROOM or get_chatroom()
        if chatroom is None: return CHATROOM_MSG
        r = itchat.add_member_into_chatroom(chatroom['UserName'], [friend])
        if r['BaseResponse']['ErrMsg'] == '':
            status = r['MemberList'][0]['MemberStatus']
            itchat.delete_member_from_chatroom(chatroom['UserName'], [friend])
            return { 3: u'该好友已经将你加入黑名单。',
                4: u'该好友已经将你删除。', }.get(status,
                u'该好友仍旧与你是好友关系。')
        else:
            return u'无法获取好友状态，预计已经达到接口调用限制。'

@itchat.msg_register(itchat.content.CARD)
def get_friend(msg):
    if msg['ToUserName'] != 'filehelper': return
    friendStatus = get_friend_status(msg['RecommendInfo'])
    itchat.send(friendStatus, 'filehelper')

itchat.auto_login(True)
itchat.send(HELP_MSG, 'filehelper')
itchat.run()


# coding=utf-8

"""
这是一个通过微信控制电脑播放音乐的小项目，那么主要就是三个功能： 输入“帮助”，显示帮助 输入“关闭”，关闭音乐播放 * 输入具体歌名，进入歌曲的选择
"""
import os

import itchat
from NetEaseMusicApi import interact_select_song

HELP_MSG = """\
欢迎使用微信网易云音乐
帮助：显示帮助
关闭：关闭歌曲
歌名：按照引导播放音乐
"""

with open('stop.mp3', 'w') as f:
    pass


def close_music():
    os.startfile('stop.mp3')


@itchat.msg_register(itchat.content.TEXT)
def music_player(msg):
    if msg['ToUserName'] != 'filehelper':
        return
    if msg['Text'] == u'关闭':
        close_music()
        itchat.send(u'音乐已关闭', 'filehelper')
    if msg['Text'] == u'帮助':
        itchat.send(HELP_MSG, 'filehelper')
    else:
        itchat.send(interact_select_song(msg['Text']), 'filehelper')


itchat.auto_login(True)
itchat.send(HELP_MSG, 'filehelper')
itchat.run()


import itchat
"""
图片对应itchat.content.PICTURE
语音对应itchat.content.RECORDING
名片对应itchat.content.CARD

TEXT = 'Text'
MAP = 'Map'
CARD = 'Card'
NOTE = 'Note'
SHARING = 'Sharing'
PICTURE = 'Picture'
RECORDING = VOICE = 'Recording'
ATTACHMENT = 'Attachment'
VIDEO = 'Video'
FRIENDS = 'Friends'
SYSTEM = 'System'
"""
@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    print(msg['Text'])
itchat.auto_login()
itchat.run()

import itchat
"""例子将会将文本消息原封不动的返回"""
@itchat.msg_register(itchat.content.TEXT)
def print_content(msg):
    return msg['Text']

itchat.auto_login()
itchat.run()


import requests
import itchat

KEY = '8edce3ce905adbb965e6b35c3834d'


def get_response(msg):
    # 这里我们就像在“3. 实现最简单的与图灵机器人的交互”中做的一样
    # 构造了要发送给服务器的数据
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key': KEY,
        'info': msg,
        'userid': 'wechat-robot',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        # 字典的get方法在字典没有'text'值的时候会返回None而不会抛出异常
        return r.get('text')
    # 为了防止服务器没有正常响应导致程序异常退出，这里用try-except捕获了异常
    # 如果服务器没能正常交互（返回非json或无法连接），那么就会进入下面的return
    except:
        # 将会返回一个None
        return


# 这里是我们在“1. 实现微信消息的获取”中已经用到过的同样的注册方法
@itchat.msg_register(itchat.content.TEXT)
def tuling_reply(msg):
    # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
    defaultReply = 'I received: ' + msg['Text']
    # 如果图灵Key出现问题，那么reply将会是None
    reply = get_response(msg['Text'])
    # a or b的意思是，如果a有内容，那么返回a，否则返回b
    # 有内容一般就是指非空或者非None，你可以用`if a: print('True')`来测试
    return reply or defaultReply


# 为了让实验过程更加方便（修改程序不用多次扫码），我们使用热启动
itchat.auto_login(hotReload=True)
itchat.run()
