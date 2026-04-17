# cython: language_level=3
from . import uiautomation as uia
from .languages import *
from .utils import *
from .color import *
from .errors import *
import datetime
import threading
from abc import ABC, abstractmethod
import traceback
import random
import string
import time
import os
import re

class WxParam:
    SYS_TEXT_HEIGHT = 33
    TIME_TEXT_HEIGHT = 34
    RECALL_TEXT_HEIGHT = 45
    CHAT_TEXT_HEIGHT = 52
    CHAT_IMG_HEIGHT = 117
    DEFALUT_SAVEPATH = os.path.join(os.getcwd(), 'wxauto文件')
    SHORTCUT_SEND = '{Enter}'
    SAVE_PATH_METHOD = 1   # 1: win32api, 2: uiautomation
    MOUSE_MOVE = False
    FILE_DOWNLOAD_TIMEOUT = 100
    SHORTCUT_NEWLINE = '{Ctrl}{Enter}'
    LISTEN_INTERVAL = 1

class WxResponse(dict):
    def __init__(self, status: str, msg: str, data: dict = None):
        super().__init__(status=status, msg=msg, data=data)

    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())

    def to_dict(self):
        return {
            'status': self['status'],
            'message': self['msg'],
            'data': self['data']
        }

    def __bool__(self):
        return self.is_success
    
    @property
    def is_success(self):
        return self['status'] == '成功'

    @classmethod
    def success(cls, msg=None, data: dict = None):
        return cls(status="成功", msg=msg, data=data)

    @classmethod
    def failure(cls, msg: str, data: dict = None):
        return cls(status="失败", msg=msg, data=data)

    @classmethod
    def error(cls, msg: str, data: dict = None):
        return cls(status="错误", msg=msg, data=data)

class Listener(ABC):
    def _listener_start(self):
        wxlog.debug('开始监听')
        self._listener_is_listening = True
        self._listener_messages = {}
        self._lock = threading.Lock()
        self._listener_stop_event = threading.Event()
        self._listener_thread = threading.Thread(target=self._listener_listen, daemon=True)
        self._listener_thread.start()

    def _listener_listen(self):
        if not hasattr(self, 'listen') or not self.listen:
            self.listen = {}
        while not self._listener_stop_event.is_set():
            # wxlog.debug('获取监听消息...')
            try:
                self._get_listen_messages()
            except:
                wxlog.debug(f'监听消息失败：{traceback.format_exc()}')
            time.sleep(WxParam.LISTEN_INTERVAL)

    def _listener_stop(self):
        self._listener_is_listening = False
        self._listener_stop_event.set()
        self._listener_thread.join()

    @abstractmethod
    def _get_listen_messages(self):
        ...

class WeChatBase:
    def _lang(self, text, langtype='MAIN'):
        if langtype == 'MAIN':
            return MAIN_LANGUAGE[text][self.language]
        elif langtype == 'WARNING':
            return WARNING[text][self.language]
        
    def _get_now_msgid(self):
        if not self.C_MsgList.Exists(0):
            return []
        return [''.join([str(i) for i in msgitem.GetRuntimeId()]) for msgitem in self.C_MsgList.GetChildren()]

    def _split(self, MsgItem):
        uia.SetGlobalSearchTimeout(0)
        MsgItemName = MsgItem.Name
        msgrect = MsgItem.BoundingRectangle
        msgheight = msgrect.height()
        if msgheight == WxParam.SYS_TEXT_HEIGHT:
            Msg = ['SYS', MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
        elif msgheight == WxParam.TIME_TEXT_HEIGHT:
            Msg = ['Time', MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
        elif msgheight == WxParam.RECALL_TEXT_HEIGHT:
            if '撤回' in MsgItemName:
                Msg = ['Recall', MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
            else:
                Msg = ['SYS', MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
        else:
            Index = 1
            User = MsgItem.ButtonControl(foundIndex=Index)
            text_control = MsgItem.TextControl()
            try:
                while True:
                    if User.Name == '':
                        Index += 1
                        User = MsgItem.ButtonControl(foundIndex=Index)
                    else:
                        userrect = User.BoundingRectangle
                        username = User.Name
                        break
                mid = (msgrect.left + msgrect.right)/2
                if userrect.left < mid:
                    text_control = MsgItem.TextControl()
                    if text_control.Exists(0.1) and text_control.BoundingRectangle.top < userrect.top:
                        name = (username, text_control.Name)
                    else:
                        name = (username, username)
                else:
                    name = 'Self'
                Msg = [name, MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
            except:
                Msg = ['SYS', text_control.Name if text_control.Exists(0) else MsgItemName, ''.join([str(i) for i in MsgItem.GetRuntimeId()])]
        uia.SetGlobalSearchTimeout(10.0)
        return ParseMessage(Msg, MsgItem, self)
    
    def _getmsgs(self, msgitems, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False):
        msgs = []
        for MsgItem in msgitems:
            if MsgItem.ControlTypeName == 'ListItemControl':
                msgs.append(self._split(MsgItem))

        msgtypes = [
            f"[{self._lang('图片')}]",
            f"[{self._lang('视频')}]",
            f"[{self._lang('文件')}]",
            f"[{self._lang('语音')}]",
            f"[{self._lang('链接')}]",
            f"[{self._lang('音乐')}]",
            f"[{self._lang('位置')}]",
        ]

        if not [i for i in msgs if i.content[:4] in msgtypes]:
            return msgs

        for msg in msgs:
            if msg.type not in ('friend', 'self'):
                continue
            msg.mtype = 'text'
            wxlog.debug(f"消息内容：{msg.content}")
            if savepic and msg.content.startswith(f"[{self._lang('图片')}]"):
                imgpath = self._download_pic(msg.control)
                if imgpath:
                    msg.content = imgpath
                    msg.mtype = 'image'
            elif savevideo and msg.content.startswith(f"[{self._lang('视频')}]"):
                imgpath = self._download_pic(msg.control, video=True)
                if imgpath:
                    msg.content = imgpath
                    msg.mtype = 'video'
            elif savefile and msg.content.startswith(f"[{self._lang('文件')}]"):
                filepath = self._download_file(msg.control)
                if filepath:
                    msg.content = f"{filepath}"
                    msg.mtype = 'file'
            elif savevoice and msg.content.startswith(f"[{self._lang('语音')}]"):
                voice_text = self._get_voice_text(msg.control)
                if voice_text:
                    msg.content = f"[wxauto语音解析]{voice_text}"
                    msg.mtype = 'voice'
            elif parseurl and msg.content in (f"[{self._lang('链接')}]", f"[{self._lang('音乐')}]"):
                card_url = self._get_card_url(msg.control, msg.content)
                if card_url:
                    msg.content = f"[wxauto卡片链接解析]{card_url}"
                    msg.mtype = 'card'
            elif msg.content.startswith(f"[{self._lang('位置')}]"):
                location_control1 = msg.control.GetProgenyControl(7, 1)
                location_control2 = msg.control.GetProgenyControl(7)
                if location_control1 and location_control2:
                    location_text = location_control1.Name + location_control2.Name
                    msg.content = location_text
                    msg.mtype = 'location'
            msg.info[1] = msg.content
        if msgs:
            msgs[-1].roll_into_view()
            self.C_MsgList.WheelDown(wheelTimes=10)
        return msgs
    
    def _download_pic(self, msgitem, video=False):
        
        imgcontrol = msgitem.ButtonControl(Name='')
        if not imgcontrol.Exists(0.5):
            return None
        
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                raise TimeoutError("图片下载超时")
            RollIntoView(self.C_MsgList, imgcontrol)
            imgcontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            imgobj = WeChatImage(video)
            if imgobj._exists(1):
                break
        savepath = imgobj.Save()
        # imgobj.Close()
        return savepath

    def _download_file(self, msgitem):
        filecontrol = msgitem.ButtonControl(Name='')
        wxlog.debug(f"filecontrol: {filecontrol.GetRuntimeId()}")
        if not filecontrol.Exists(0.5):
            return None
        RollIntoView(self.C_MsgList, filecontrol)
        filecontrol.RightClick(simulateMove=False)
        # filename = msgitem.GetProgenyControl(9, control_type='TextControl').Name
        filesize = msgitem.GetProgenyControl(10, control_type='TextControl').Name
        menu = self.UiaAPI.MenuControl(ClassName='CMenuWnd')
        while not menu.Exists(0.1):
            RollIntoView(self.C_MsgList, filecontrol)
            filecontrol.RightClick(simulateMove=False)
        copy_option = menu.ListControl().MenuItemControl(Name='复制')
        if copy_option.Exists(0.5):
            t0 = time.time()
            while time.time() - t0 < WxParam.FILE_DOWNLOAD_TIMEOUT:
                try:
                    copy_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    filepath = ReadClipboardData().get('15')[0]
                    wxlog.debug(f"获取到文件路径1: {filepath}")
                    break
                except Exception as e:
                    while not menu.Exists(0.1):
                        RollIntoView(self.C_MsgList, filecontrol)
                        filecontrol.RightClick(simulateMove=False)
                    copy_option = menu.ListControl().MenuItemControl(Name='复制')
        else:
            RollIntoView(self.C_MsgList, filecontrol)
            filecontrol.RightClick(simulateMove=False)
            if filesize[-1] in ['G', 'M'] and filesize >= '200M':
                print(filesize, '文件过大，点击接收文件')
                filecontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                filewin = self.UiaAPI.WindowControl(ClassName='MsgFileWnd')
                accept_button = filewin.ButtonControl(Name='接收文件')
                if accept_button.Exists(2):
                    accept_button.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            
            t0 = time.time()
            while time.time() - t0 < WxParam.FILE_DOWNLOAD_TIMEOUT:
                try:
                    if msgitem.TextControl(Name='接收中').Exists(0):
                        time.sleep(1)
                        continue
                    filecontrol = msgitem.ButtonControl(Name='')
                    menu = self.UiaAPI.MenuControl(ClassName='CMenuWnd')
                    while not menu.Exists(0.1):
                        RollIntoView(self.C_MsgList, filecontrol)
                        filecontrol.RightClick(simulateMove=False)
                    copy_option = menu.ListControl().MenuItemControl(Name='复制')
                    if copy_option.Exists(0.5):
                        copy_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                        filepath = ReadClipboardData().get('15')[0]
                        wxlog.debug(f"获取到文件路径2: {filepath}")
                        break
                    else:
                        RollIntoView(self.C_MsgList, filecontrol)
                        filecontrol.RightClick(simulateMove=False)
                except Exception as e:
                    pass
                time.sleep(1)
        
        savepath = os.path.join(WxParam.DEFALUT_SAVEPATH, os.path.split(filepath)[1])
        if not os.path.exists(WxParam.DEFALUT_SAVEPATH):
            os.makedirs(WxParam.DEFALUT_SAVEPATH)
        t0 = time.time()
        while time.time() - t0 < 10:
            try:
                shutil.copyfile(filepath, savepath)
                SetClipboardText('')
                return savepath
            except FileNotFoundError:
                time.sleep(0.1)
                continue
        

    def _get_voice_text(self, msgitem):
        if msgitem.GetProgenyControl(8, 4):
            return msgitem.GetProgenyControl(8, 4).Name
        voicecontrol = msgitem.ButtonControl(Name='')
        if not voicecontrol.Exists(0.5):
            return None
        RollIntoView(self.C_MsgList, voicecontrol)
        msgitem.GetProgenyControl(7, 1).RightClick(simulateMove=False)
        menu = self.UiaAPI.MenuControl(ClassName='CMenuWnd')
        option = menu.MenuItemControl(Name="语音转文字")
        if not option.Exists(0.5):
            voicecontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            if not msgitem.GetProgenyControl(8, 4):
                return None
        else:
            option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        text = ''
        while True:
            if not msgitem.Exists(0):
                return '[已撤回]'
            text_control = msgitem.GetProgenyControl(8, 4, refresh=True)
            if text_control is not None:
                if text_control.Name == text:
                    return text
                text = text_control.Name
            time.sleep(0.1)

    def _get_card_url(self, msgitem, content):
        if content not in ('[链接]', '[音乐]'):
            return None
        if content == '[链接]' and (
            msgitem.TextControl(Name="邀请你加入群聊").Exists(0)\
            or msgitem.TextControl(Name="Group Chat Invitation").Exists(0)):
            return '[链接](群聊邀请)'
        if not msgitem.PaneControl().Exists(0):
            return None
        link_control_list = msgitem.PaneControl().GetChildren()
        if len(link_control_list) < 2:
            return None
        link_control = link_control_list[1]
        if not link_control.ButtonControl().Exists(0):
            return None

        RollIntoView(self.C_MsgList, link_control)
        # msgitem.TextControl().Click()
        msgitem.GetAllProgeny()[-1][-1].Click()
        t0 = time.time()
        while not FindWindow('Chrome_WidgetWin_0', '微信'):
            if time.time() - t0 > 10:
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)
        wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)

        while not wxbrowser.DocumentControl().GetChildren() or wxbrowser.DocumentControl().TextControl(Name="mp appmsg sec open").Exists(0):
            if time.time() - t0 > 10:
                if wxbrowser.Exists(0):
                    wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)

        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
                return '[链接]无法获取url'
            wxbrowser.PaneControl(searchDepth=1, ClassName='').MenuItemControl(Name="更多").Click()
            # wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)
            time.sleep(0.5)
            copyurl = wxbrowser.PaneControl(ClassName='Chrome_WidgetWin_0').MenuItemControl(Name='复制链接')
            if copyurl.Exists(0):
                copyurl.Click()
                break
        
        url = ReadClipboardData()['13']
        wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
        return url


class ChatWnd(WeChatBase):
    _clsname = 'ChatWnd'

    def __init__(self, who, wx, language='cn'):
        self.who = who
        self._wx = wx
        self.language = language
        self.usedmsgid = []
        self.UiaAPI = uia.WindowControl(searchDepth=1, ClassName=self._clsname, Name=who)
        self.editbox = self.UiaAPI.EditControl()
        self.C_MsgList = self.UiaAPI.ListControl(Name='消息')
        # self.GetAllMessage()

        self.savepic = False   # 该参数用于在自动监听的情况下是否自动保存聊天图片

    def __repr__(self) -> str:
        return f"<wxauto Chat Window at {hex(id(self))} for {self.who}>"

    def _show(self):
        self.HWND = FindWindow(name=self.who, classname=self._clsname)
        win32gui.ShowWindow(self.HWND, 1)
        win32gui.SetWindowPos(self.HWND, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.HWND, -2, 0, 0, 0, 0, 3)
        self.UiaAPI.SwitchToThisWindow()

    def Close(self):
        try:
            self.UiaAPI.SendKeys('{Esc}')
        except:
            pass

    def ChatInfo(self):

        chat_info = {}

        chat_name_control = self.UiaAPI.GetProgenyControl(11)
        chat_name_control_list = chat_name_control.GetChildren()
        chat_name_control_count = len(chat_name_control_list)
        if chat_name_control_count == 1:
            if self.UiaAPI.ButtonControl(Name='公众号主页').Exists(0):
                chat_info['chat_type'] = 'official'
            else:
                chat_info['chat_type'] = 'friend'
            chat_info['chat_name'] = chat_name_control_list[0].Name
        elif chat_name_control_count == 2:
            chat_info['chat_type'] = 'group'
            chat_info['chat_name'] = chat_name_control_list[0].Name.replace(chat_name_control_list[-1].Name, '')
            chat_info['group_member_count'] = int(chat_name_control_list[-1].Name.replace('(', '').replace(')', '').replace(' ', ''))
            ori_chat_name_control = chat_name_control_list[0].GetParentControl().GetParentControl().TextControl(searchDepth=1)
            if ori_chat_name_control.Exists(0):
                chat_info['chat_remark'] = chat_info['chat_name']
                chat_info['chat_name'] = ori_chat_name_control.Name
        return chat_info

    def AtAll(self, msg=None):
        """@所有人
        
        Args:
            msg (str, optional): 要发送的文本消息
        """
        wxlog.debug(f"@所有人：{self.who} --> {msg}")
        
        if not self.editbox.HasKeyboardFocus:
            self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        self.editbox.Input('@')
        atwnd = self.UiaAPI.PaneControl(ClassName='ChatContactMenu')
        if atwnd.Exists(maxSearchSeconds=0.1):
            atwnd.ListItemControl(Name='所有人').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            if msg:
                if not msg.startswith('\n'):
                    msg = '\n' + msg
                self.SendMsg(msg)
            else:
                self.editbox.SendKeys(WxParam.SHORTCUT_SEND)

    def SendTypingText(self, msg, clear=True):
        """发送文本消息（打字机模式），支持换行及@功能

        Args:
            msg (str): 要发送的文本消息
            who (str): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            clear (bool, optional): 是否清除原本的内容，

        Example:
            >>> wx = WeChat()
            >>> wx.SendTypingText('你好', who='张三')

            换行及@功能：
            >>> wx.SendTypingText('各位下午好\n{@张三}负责xxx\n{@李四}负责xxxx', who='工作群')
        """
        if not msg:
            return None
        
        # msg = msg.replace('\r', '\n').replace('\n', WxParam.SHORTCUT_NEWLINE)

        if clear:
            self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)

        def _at(name):
            self.editbox.Input(name)
            atwnd = self.UiaAPI.PaneControl(ClassName='ChatContactMenu')
            if atwnd.Exists(maxSearchSeconds=0.1):
                atele = atwnd.ListItemControl(Name=name[1:])
                if atele.Exists(0):
                    RollIntoView(atwnd, atele)
                    atele.Click()

        atlist = re.findall(r'{(@.*?)}', msg)
        for name in atlist:
            text, msg = msg.split(f'{{{name}}}')
            self.editbox.Input(text)
            _at(name)
        self.editbox.Input(msg)
        self.editbox.SendKeys(WxParam.SHORTCUT_SEND)

    def SendMsg(self, msg, at=None, clear=True):
        """发送文本消息

        Args:
            msg (str): 要发送的文本消息
            at (str|list, optional): 要@的人，可以是一个人或多个人，格式为str或list，例如："张三"或["张三", "李四"]
        """
        wxlog.debug(f"发送消息：{self.who} --> {msg}")

        if not msg and not at:
            return WxResponse.failure(f"消息为空，发送失败")
        
        while not self.editbox.HasKeyboardFocus:
            self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        if clear:
            self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)

        _msg = msg
        if at:
            if isinstance(at, str):
                at = [at]
            for i in at:
                self.editbox.Input('@'+i)
                atwnd = self.UiaAPI.PaneControl(ClassName='ChatContactMenu')
                if atwnd.Exists(maxSearchSeconds=0.1):
                    ateles = atwnd.ListControl().GetChildren()
                    if len(ateles) == 1:
                        ateles[0].Click()
                    else:
                        atele = atwnd.ListItemControl(Name=i)
                        if atele.Exists(0):
                            RollIntoView(atwnd, atele)
                            atele.Click()
                        else:
                            atwnd.SendKeys('{ESC}')
                            for _ in range(len(i)+1):
                                self.editbox.SendKeys('{BACK}')
                
                    # if msg and not msg.startswith('\n'):
                    #     msg = '\n' + msg
                else:
                    for _ in range(len(i)+1):
                        self.editbox.SendKeys('{BACK}')
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                wxlog.debug(ReadClipboardData())
                raise TimeoutError(f'发送消息超时 --> {self.who} - {_msg}')
            SetClipboardText(msg)
            if not self.editbox.HasKeyboardFocus:
                self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            self.editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE)
            wxlog.debug(ReadClipboardData())
            if self.editbox.GetValuePattern().Value.replace('￼', ''):
                break
            else:
                wxlog.debug('快捷键粘贴失败，尝试右键粘贴')
                self.editbox.RightClick()
                menu = CMenuWnd(self)
                menu.choose('粘贴')
                if self.editbox.GetValuePattern().Value.replace('￼', ''):
                    break
        t0 = time.time()
        while self.editbox.GetValuePattern().Value:
            if time.time() - t0 > 10:
                raise TimeoutError(f'发送消息超时 --> {self.who} - {_msg}')
            if not self.editbox.HasKeyboardFocus:
                self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            if clear:
                self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            self.editbox.SendKeys(WxParam.SHORTCUT_SEND)
            time.sleep(0.2)
        return WxResponse.success(f"发送成功")

    def SendEmotion(self, emotion_index):
        """发送自定义表情

        Args:
            emotion_index (int): 表情序号，从0开始
        """
        lock = threading.Lock()
        with lock:
            self.UiaAPI.ButtonControl(RegexName='表情.*?').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            EmotionWnd = self.UiaAPI.PaneControl(ClassName='EmotionWnd')
            my_emotion_icon = EmotionWnd.CheckBoxControl(Name='添加的单个表情')
            while not my_emotion_icon.Exists(0):
                EmotionWnd.CheckBoxControl().GetParentControl().WheelUp(wheelTimes=10)
            my_emotion_icon.Click(move=False, simulateMove=False, return_pos=False)
            emotion_list = EmotionWnd.ListControl()
            while not emotion_list.TextControl(Name="添加的单个表情").Exists(0):
                emotion_list.WheelUp(wheelTimes=10)

            emotions = emotion_list.GetChildren()[1:]
            amount = len(emotions)
            last_one = emotions[-1]
            top0 = emotions[0].BoundingRectangle.top
            for idx, e in enumerate(emotions):
                if e.BoundingRectangle.top != top0:
                    break

            def next_page(index, emotion_list, emotions, last_one, idx, amount):
                if index < len(emotions):
                    time.sleep(1)
                    emotion = emotions[index]
                    return emotion
                else:
                    while True:
                        position = last_one.BoundingRectangle.top
                        emotions = emotion_list.GetChildren()
                        if last_one.GetRuntimeId() == emotions[idx-1].GetRuntimeId():
                            break
                        emotion_list.WheelDown()
                        time.sleep(0.05)
                        if last_one.BoundingRectangle.top == position:
                            return 
                    fourth = emotions[idx*2- 1]
                    while True:
                        position = fourth.BoundingRectangle.top
                        emotions = emotion_list.GetChildren()
                        if fourth.GetRuntimeId() == emotions[idx-1].GetRuntimeId():
                            new_index = index - amount
                            last_one = emotions[-1]
                            amount = len(emotions)
                            return next_page(new_index, emotion_list, emotions, last_one, idx, amount)
                        emotion_list.WheelDown()
                        time.sleep(0.05)
                        if fourth.BoundingRectangle.top == position:
                            return
            
            emotion = next_page(emotion_index, emotion_list, emotions, last_one, idx, amount)
            if emotion is not None:
                RollIntoView(emotion_list, emotion)
                emotion.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return WxResponse.success(f'发送表情({emotion_index})成功')
            else:
                wxlog.debug(f'未找到表情索引：{emotion_index}')
                EmotionWnd.SendKeys('{Esc}')
                return WxResponse.failure(f'未找到表情索引：{emotion_index}')

    def SendFiles(self, filepath):
        """向当前聊天窗口发送文件
        
        Args:
            filepath (str|list): 要复制文件的绝对路径  
            
        Returns:
            bool: 是否成功发送文件
        """
        wxlog.debug(f"发送文件：{self.who} --> {filepath}")
        filelist = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                return WxResponse.failure(f'未找到文件：{filepath}，无法成功发送')
            else:
                filelist.append(os.path.realpath(filepath))
        elif isinstance(filepath, (list, tuple, set)):
            for i in filepath:
                if os.path.exists(i):
                    filelist.append(i)
                else:
                    wxlog.debug(f'未找到文件：{i}')
        else:
            return WxResponse.failure(f'filepath参数格式错误：{type(filepath)}，应为str、list、tuple、set格式')
        
        if filelist:
            
            self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            t0 = time.time()
            while True:
                if time.time() - t0 > 10:
                    raise TimeoutError(f'发送文件超时 --> {filelist}')
                SetClipboardFiles(filelist)
                time.sleep(0.2)
                self.editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE)
                t1 = time.time()
                while time.time() - t1 < 5:
                    try:
                        edit_value = self.editbox.GetValuePattern().Value
                        break
                    except:
                        time.sleep(0.1)

                if edit_value:
                    break
            
            t0 = time.time()
            while time.time() - t0 < 10:
                t1 = time.time()
                while time.time() - t1 < 5:
                    try:
                        edit_value = self.editbox.GetValuePattern().Value
                        break
                    except:
                        time.sleep(0.1)
                if not edit_value:
                    break
                self.editbox.SendKeys(WxParam.SHORTCUT_SEND)
                time.sleep(0.1)
            return WxResponse.success('发送成功')
        else:
            wxlog.debug('所有文件都无法成功发送')
            return WxResponse.failure('所有文件都无法成功发送')
        
    def GetAllMessage(self, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False):
        '''获取当前窗口中加载的所有聊天记录
        
        Args:
            savepic (bool): 是否自动保存聊天图片
            savefile (bool): 是否自动保存聊天文件
            savevoice (bool): 是否自动保存语音转文字
            parseurl (bool): 是否解析链接
            
        Returns:
            list: 聊天记录信息
        '''
        wxlog.debug(f"获取所有聊天记录：{self.who}")
        MsgItems = self.C_MsgList.GetChildren()
        msgs = self._getmsgs(MsgItems, savepic, savevideo, savefile, savevoice, parseurl)
        return msgs
    
    def GetNewMessage(self, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False):
        '''获取当前窗口中加载的新聊天记录

        Args:
            savepic (bool): 是否自动保存聊天图片
            savefile (bool): 是否自动保存聊天文件
            savevoice (bool): 是否自动保存语音转文字
            parseurl (bool): 是否解析链接
        
        Returns:
            list: 新聊天记录信息
        '''
        # wxlog.debug(f"获取新聊天记录：{self.who}")
        now_msgid = self._get_now_msgid()
        if not self.usedmsgid:
            wxlog.debug(f"`{self.who}`首次监听，缓存消息id")
            self.usedmsgid = self._get_now_msgid()
            return []
        elif now_msgid and now_msgid[-1] == self.usedmsgid[-1]:
            # wxlog.debug(f'`{self.who}`没有新消息')
            return []
    
        MsgItems = self.C_MsgList.GetChildren()

        nowmsgids = [''.join([str(i) for i in i.GetRuntimeId()]) for i in MsgItems]
        new1 = [x for x in nowmsgids if x not in set(self.usedmsgid)]
        a_set = set(self.usedmsgid)
        last_one_msgid = max((x for x in nowmsgids if x in a_set), key=self.usedmsgid.index, default=None)
        new2 = nowmsgids[nowmsgids.index(last_one_msgid) + 1 :] if last_one_msgid is not None else []
        new = [i for i in new1 if i in new2] if new2 else new1

        NewMsgItems = [i for i in MsgItems if ''.join([str(i) for i in i.GetRuntimeId()]) in new]
        if not NewMsgItems:
            wxlog.debug(f'`{self.who}`没有新消息')
            return []

        newmsgs = self._getmsgs(NewMsgItems, savepic, savevideo, savefile, savevoice, parseurl)
        self.usedmsgid = list(self.usedmsgid + [i[-1] for i in newmsgs])[-100:]
        wxlog.debug(f'`{self.who}`返回新消息{newmsgs}')
        return newmsgs
    
    def AddGroupMembers(self, members):
        """添加群成员

        Args:
            group (str): 群名或备注名
            members (list): 成员列表，列表元素可以是好友微信号、昵称、备注名
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        self.UiaAPI.GetProgenyControl(10, control_type='ButtonControl').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        RoomDetailWndControl = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        RoomDetailWndControl.ButtonControl(Name='添加').GetParentControl().GetChildren()[0].Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        addWnd = AddMemberWnd(self)
        for member in members:
            addWnd.Add(member)
            time.sleep(0.3)
        if len(addWnd.UiaAPI.TableControl(Name='已选择联系人', searchDepth=3).GetChildren()) == 0:
            wxlog.debug('未找到任何成员')
            addWnd.Close()
        else:
            wxlog.debug(f'添加 {len(members)} 个成员')
            time.sleep(0.5)
            try:
                addWnd.Submit()
            except:
                pass
        time.sleep(0.2)
        try:
            RoomDetailWndControl.SendKeys('{Esc}')
        except:
            pass

    def RemoveGroupMembers(self, members):
        """移除群成员
        
        Args:
            members (list): 成员列表
        """
        if isinstance(members, str):
            members = [members]
        if WxParam.MOUSE_MOVE:
            self._show()
        self.UiaAPI.GetProgenyControl(10, control_type='ButtonControl').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        RoomDetailWndControl = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        RoomDetailWndControl.ButtonControl(Name='移出').GetParentControl().GetChildren()[0].Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        removeWnd = DeleteMemberWnd(self)
        for member in members:
            removeWnd.Remove(member)
            time.sleep(0.3)
        if len(removeWnd.UiaAPI.TableControl().GetChildren()) == 0:
            wxlog.debug('未找到任何成员')
            removeWnd.Close()
        else:
            wxlog.debug(f'移出 {len(members)} 个群成员')
            time.sleep(0.5)
            try:
                removeWnd.Submit()
            except:
                pass
        time.sleep(0.2)
        try:
            RoomDetailWndControl.SendKeys('{Esc}')
        except:
            pass
    
    def LoadMoreMessage(self, interval=0.3):
        """加载当前聊天页面更多聊天信息

        Args:
            interval (float, optional): 滚动间隔时间，看自己电脑卡顿程度调整，默认0.3秒
        
        Returns:
            bool: 是否成功加载更多聊天信息
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        msg_len = len(self.C_MsgList.GetChildren())
        loadmore = self.C_MsgList.GetChildren()[0]
        loadmore_top = loadmore.BoundingRectangle.top
        while True:
            if len(self.C_MsgList.GetChildren()) > msg_len:
                isload = True
                break
            else:
                msg_len = len(self.C_MsgList.GetChildren())
                self.C_MsgList.WheelUp(wheelTimes=10)
                time.sleep(interval)
                if self.C_MsgList.GetChildren()[0].BoundingRectangle.top == loadmore_top\
                    and len(self.C_MsgList.GetChildren()) == msg_len:
                    isload = False
                    break
                else:
                    loadmore_top = self.C_MsgList.GetChildren()[0].BoundingRectangle.top
                    
        self.C_MsgList.WheelUp(wheelTimes=1, waitTime=0.1)
        return isload

    def GetGroupMembers(self, add_friend_mode=False):
        """获取当前聊天群成员

        Returns:
            list: 当前聊天群成员列表
        """
        wxlog.debug(f"获取当前聊天群成员：{self.who}")
        ele = self.UiaAPI.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
        try:
            uia.SetGlobalSearchTimeout(1)
            rect = ele.BoundingRectangle
            Click(rect)
        except:
            return 
        finally:
            uia.SetGlobalSearchTimeout(10)
        roominfoWnd = self.UiaAPI.WindowControl(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        more = roominfoWnd.ButtonControl(Name='查看更多', searchDepth=8)
        if more.Exists(0.1):
            more.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        
        if add_friend_mode:
            members = [GroupMemberElement(i, self) for i in roominfoWnd.ListControl(Name='聊天成员').GetChildren()]
            while members[-1].nickname in ['添加', '移出']:
                members = members[:-1]
            return members
        else:
            members = [i.Name for i in roominfoWnd.ListControl(Name='聊天成员').GetChildren()]
            while members[-1] in ['添加', '移出']:
                members = members[:-1]
            roominfoWnd.SendKeys('{Esc}')
            return members
        
    def ManageGroup(self, name=None, remark=None, myname=None, notice=None, quit=False):
        """管理当前聊天页面的群聊
        
        Args:
            name (str, optional): 修改群名称
            remark (str, optional): 备注名
            myname (str, optional): 我的群昵称
            notice (str, optional): 群公告
            quit (bool, optional): 是否退出群，当该项为True时，其他参数无效
        
        Returns:
            dict: 修改结果
        """
        edit_result = {}
        chat_info = self.ChatInfo()
        if chat_info['chat_type'] != 'group':
            wxlog.debug('当前聊天对象不是群聊')
            return WxResponse.failure('当前聊天对象不是群聊')
        ele = self.UiaAPI.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
        ele.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=True)
        roomwnd = SessionChatRoomDetailWnd(self)
        if quit:
            quit_result = roomwnd.quit()
            edit_result['quit'] = quit_result
            return edit_result
        if name is not None:
            edit_name_result = roomwnd.edit_group_name(name)
            edit_result['name'] = edit_name_result
        if remark is not None:
            edit_remark_result = roomwnd.edit_remark(remark)
            edit_result['remark'] = edit_remark_result
        if myname is not None:
            edit_myname_result = roomwnd.edit_my_name(myname)
            edit_result['myname'] = edit_myname_result
        if notice is not None:
            edit_notice_result = roomwnd.edit_group_notice(notice)
            edit_result['notice'] = edit_notice_result
        roomwnd.close()
        return edit_result
    
    def CallGroupMsg(self, members):
        """发起群语音通话
        
        Args:
            group (str): 群名或备注名
            members (list): 成员列表，列表元素可以是好友微信号、昵称、备注名
        """
        wxlog.debug(f"发起群语音通话：{members}")
        chat_info = self.ChatInfo()
        if chat_info['chat_type'] != 'group':
            wxlog.debug('当前聊天对象不是群聊')
            return False
        
        self.UiaAPI.ButtonControl(Name='语音聊天').Click()
        addwnd = AddTalkMemberWnd(self)
        if not addwnd.UiaAPI.Exists(5):
            return False
        for member in members:
            addwnd.Add(member)
        addwnd.Submit()

class ChatRecordWnd:
    def __init__(self):
        self.api = uia.WindowControl(ClassName='ChatRecordWnd', searchDepth=1)

    def GetContent(self):
        """获取聊天记录内容"""
        
        msgids = []
        msgs = []
        listcontrol = self.api.ListControl()
        while True:
            listitems = listcontrol.GetChildren()
            listitemids = [item.GetRuntimeId() for item in listitems]
            try:
                msgids = msgids[msgids.index(listitemids[0]):]
            except:
                pass
            for item in listitems:
                msgid = item.GetRuntimeId()
                if msgid not in msgids:
                    msgids.append(msgid)
                    sender = item.GetProgenyControl(4, control_type='TextControl').Name
                    msgtime = ParseWeChatTime(item.GetProgenyControl(4, 1, control_type='TextControl').Name)
                    if '[图片]' in item.Name:
                        imgcontrol = item.GetProgenyControl(6, control_type='ButtonControl')
                        # wait for image loading
                        for _ in range(10):
                            if imgcontrol:
                                RollIntoView(listcontrol, imgcontrol, True)
                                imgcontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                                img = WeChatImage()
                                imgpath = img.Save()
                                img.Close()
                                msgs.append([sender, imgpath, msgtime])
                                break
                            else:
                                time.sleep(1)
                    elif item.Name == '' and item.TextControl(Name='视频').Exists(0.3):
                        videocontrol = item.GetProgenyControl(5, control_type='ButtonControl')
                        if videocontrol:
                            RollIntoView(listcontrol, videocontrol, True)
                            videocontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                            video = WeChatImage(video=True)
                            videopath = video.Save()
                            video.Close()
                            msgs.append([sender, videopath, msgtime])
                    else:
                        textcontrols = [i for i in GetAllControl(item) if i.ControlTypeName == 'TextControl']
                        who = textcontrols[0].Name
                        try:
                            content = textcontrols[2].Name
                        except IndexError:
                            content = ''
                        msgs.append([sender, content, msgtime])
            topcontrol = listitems[-1]
            top = topcontrol.BoundingRectangle.top
            self.api.WheelDown(wheelTimes=3)
            time.sleep(0.1)
            if topcontrol.Exists(0.1) and top == topcontrol.BoundingRectangle.top and listitemids == [item.GetRuntimeId() for item in listcontrol.GetChildren()]:
                self.api.SendKeys('{Esc}')
                return msgs

class WeChatImage:
    _clsname = 'ImagePreviewWnd'

    def __init__(self, video=False, language='cn') -> None:
        self._video_mode = video
        self.language = language
        self.api = uia.WindowControl(ClassName=self._clsname, searchDepth=1)
        if self._exists(1):
            MainControl1 = [i for i in self.api.GetChildren() if not i.ClassName][0]
            self.ToolsBox, self.PhotoBox = MainControl1.GetChildren()
            
            # tools按钮
            self.t_previous = self.ToolsBox.ButtonControl(Name=self._lang('上一张'))
            self.t_next = self.ToolsBox.ButtonControl(Name=self._lang('下一张'))
            self.t_zoom = self.ToolsBox.ButtonControl(Name=self._lang('放大'))
            self.t_translate = self.ToolsBox.ButtonControl(Name=self._lang('翻译'))
            self.t_ocr = self.ToolsBox.ButtonControl(Name=self._lang('提取文字'))
            self.t_save = self.api.ButtonControl(Name=self._lang('另存为...'))
            self.t_qrcode = self.ToolsBox.ButtonControl(Name=self._lang('识别图中二维码'))

    def __repr__(self) -> str:
        return f"<wxauto WeChat Image at {hex(id(self))}>"
    
    def _lang(self, text):
        return IMAGE_LANGUAGE[text][self.language]
    
    def _show(self):
        HWND = FindWindow(classname=self._clsname)
        win32gui.ShowWindow(HWND, 1)
        self.api.SwitchToThisWindow()

    def _exists(self, t=0.1):
        return self.api.Exists(t)
        
    def OCR(self, wait=10):
        result = ''
        ctrls = self.PhotoBox.GetChildren()
        if len(ctrls) == 2:
            self.t_ocr.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        t0 = time.time()
        while time.time() - t0 < wait:
            ctrls = self.PhotoBox.GetChildren()
            if len(ctrls) == 3:
                TranslateControl = ctrls[-1]
                result = TranslateControl.TextControl().Name
                if result:
                    return result
            else:
                self.t_ocr.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            time.sleep(0.1)
        Warnings.lightred('获取文字识别失败', stacklevel=2)
        return result
    
    @tenacity.retry(wait=tenacity.wait_fixed(0.5), stop=tenacity.stop_after_attempt(3))
    def Save(self, savepath='', timeout=10):
        """保存图片/视频

        Args:
            savepath (str): 绝对路径，包括文件名和后缀，例如："D:/Images/微信图片_xxxxxx.png"
            （如果不填，则默认为当前脚本文件夹下，新建一个“微信图片(或视频)”的文件夹，保存在该文件夹内）
            timeout (int, optional): 保存超时时间，默认10秒
        
        Returns:
            str: 文件保存路径，即savepath
        """
        t0 = time.time()
        if WxParam.MOUSE_MOVE:
            self._show()
        if not savepath:
            if self._video_mode:
                savepath = os.path.join(WxParam.DEFALUT_SAVEPATH, f"微信视频_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.mp4")
            else:
                savepath = os.path.join(WxParam.DEFALUT_SAVEPATH, f"微信图片_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.png")
        if not os.path.exists(os.path.split(savepath)[0]):
            os.makedirs(os.path.split(savepath)[0])

        while True:
            if time.time() - t0 > timeout:
                if self.api.Exists(0):
                    self.api.SendKeys('{Esc}')
                raise TimeoutError('下载超时')
            try:
                self.api.ButtonControl(Name='更多').Click()
                menu = self.api.MenuControl(ClassName='CMenuWnd')
                menu.MenuItemControl(Name='复制').Click()
                path = ReadClipboardData()['15'][0]
                wxlog.debug(f"读取到图片/视频路径：{path}")
                break
            except:
                wxlog.debug(traceback.format_exc())
                time.sleep(0.1)
        shutil.copyfile(path, savepath)
        SetClipboardText('')
        if self.api.Exists(0):
            wxlog.debug("关闭图片窗口")
            self.api.SendKeys('{Esc}')
        return savepath
        
    def Previous(self):
        """上一张"""
        if self.t_previous.IsKeyboardFocusable:
            
            self.t_previous.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return WxResponse.success('上一张切换成功')
        else:
            wxlog.debug('上一张按钮不可用')
            return WxResponse.failure('上一张按钮不可用')
        
    def Next(self):
        """下一张"""
        if self.t_next.IsKeyboardFocusable:
            
            self.t_next.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return WxResponse.success('下一张切换成功')
        else:
            wxlog.debug('已经是最新的图片了')
            return WxResponse.failure('已经是最新的图片了')
        
    def Close(self):
        if self.api.Exists(0):
            self.api.SendKeys('{Esc}')
    
class TextElement:
    def __init__(self, ele, wx) -> None:
        self._wx = wx
        chatname = wx.CurrentChat()
        self.ele = ele
        self.sender = ele.ButtonControl(foundIndex=1, searchDepth=2)
        _ = ele.GetFirstChildControl().GetChildren()[1].GetChildren()
        if len(_) == 1:
            self.content = _[0].TextControl().Name
            self.chattype = 'friend'
            self.chatname = chatname
        else:
            self.sender_remark = _[0].TextControl().Name
            self.content = _[1].TextControl().Name
            self.chattype = 'group'
            numtext = re.findall(' \(\d+\)', chatname)[-1]
            self.chatname = chatname[:-len(numtext)]
            
        self.info = {
            'sender': self.sender.Name,
            'content': self.content,
            'chatname': self.chatname,
            'chattype': self.chattype,
            'sender_remark': self.sender_remark if hasattr(self, 'sender_remark') else ''
        }

    def __repr__(self) -> str:
        return f"<wxauto Text Element at {hex(id(self))} ({self.sender.Name}: {self.content})>"

class NewFriendsElement:
    def __init__(self, ele, wx):
        self._wx = wx
        self.ele = ele
        self.name = self.ele.Name
        self.msg = self.ele.GetFirstChildControl().PaneControl(SearchDepth=1).GetChildren()[-1].TextControl().Name
        self.ele.GetChildren()[-1]
        self.NewFriendsBox = self._wx.ChatBox.ListControl(Name='新的朋友').GetParentControl()
        self.Status = self.ele.GetFirstChildControl().GetChildren()[-1]
        self.acceptable = isinstance(self.Status, uia.ButtonControl)
            
    def __repr__(self) -> str:
        return f"<wxauto New Friends Element at {hex(id(self))} ({self.name}: {self.msg})>"
    
    def Delete(self):
        wxlog.info(f'删除好友请求: {self.name}')
        RollIntoView(self.NewFriendsBox, self.ele)
        self.ele.RightClick()
        menu = CMenuWnd(self._wx)
        menu.choose('删除')

    def Reply(self, text):
        wxlog.debug(f'回复好友请求: {self.name}')
        RollIntoView(self.NewFriendsBox, self.ele)
        self.ele.Click()
        self._wx.ChatBox.ButtonControl(Name='回复').Click()
        edit = self._wx.ChatBox.EditControl()
        edit.Click()
        edit.ShortcutSelectAll()
        SetClipboardText(text)
        edit.ShortcutPaste()
        time.sleep(0.1)
        print('点击发送')
        self._wx.ChatBox.ButtonControl(Name='发送').Click()
        dialog = self._wx.UiaAPI.PaneControl(ClassName='WeUIDialog')
        while edit.Exists(0):
            if dialog.Exists(0):
                systext = dialog.TextControl().Name
                wxlog.debug(f'系统提示: {systext}')
                dialog.SendKeys('{Esc}')
                self._wx.ChatBox.ButtonControl(Name='').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return WxResponse.failure(msg=systext)
            time.sleep(0.1)
        self._wx.ChatBox.ButtonControl(Name='').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        return WxResponse.success()


    def Accept(self, remark=None, tags=None, permission='朋友圈'):
        """接受好友请求
        
        Args:
            remark (str, optional): 备注名
            tags (list, optional): 标签列表
            permission (str, optional): 朋友圈权限, 可选值：'朋友圈', '仅聊天'
        """
        if not self.acceptable:
            wxlog.debug(f"当前好友状态无法接受好友请求：{self.name}")
            return 
        wxlog.debug(f"接受好友请求：{self.name}  备注：{remark} 标签：{tags}")
        self._wx._show()
        RollIntoView(self.NewFriendsBox, self.Status)
        self.Status.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        NewFriendsWnd = self._wx.UiaAPI.WindowControl(ClassName='WeUIDialog')
        tipscontrol = NewFriendsWnd.TextControl(Name="你的联系人较多，添加新的朋友时需选择权限")

        permission_sns = NewFriendsWnd.CheckBoxControl(Name='聊天、朋友圈、微信运动等')
        permission_chat = NewFriendsWnd.CheckBoxControl(Name='仅聊天')
        if tipscontrol.Exists(0.5):
            permission_sns = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='朋友圈')
            permission_chat = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='仅聊天')

        if remark:
            remarkedit = NewFriendsWnd.TextControl(Name='备注名').GetParentControl().EditControl()
            remarkedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            remarkedit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            remarkedit.Input(remark)
        
        if tags:
            tagedit = NewFriendsWnd.TextControl(Name='标签').GetParentControl().EditControl()
            for tag in tags:
                tagedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                tagedit.Input(tag)
                NewFriendsWnd.PaneControl(ClassName='DropdownWindow').TextControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        if permission == '朋友圈':
            permission_sns.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        elif permission == '仅聊天':
            permission_chat.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        NewFriendsWnd.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def GetAccount(self, wait=5):
        """获取好友号
        
        Args:
            wait (int, optional): 等待时间
            
        Returns:
            str: 好友号，如果获取失败则返回None
        """
        # if isinstance(self.Status, uia.ButtonControl):
        #     wxlog.debug(f"非好友状态无法获取好友号：{self.name}")
        #     return 
        wxlog.debug(f"获取好友号：{self.name}")
        self.ele.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        account_tag_control = self._wx.ChatBox.TextControl(Name='微信号：')
        if account_tag_control.Exists(wait):
            account = account_tag_control.GetParentControl().GetChildren()[-1].Name
            self._wx.ChatBox.ButtonControl(Name='').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return account
        else:
            self._wx.ChatBox.ButtonControl(Name='').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        

class ContactWnd:
    _clsname = 'ContactManagerWindow'

    def __init__(self):
        self.UiaAPI = uia.WindowControl(ClassName=self._clsname, searchDepth=1)
        self.Sidebar, _, self.ContactBox = self.UiaAPI.PaneControl(ClassName='', searchDepth=3, foundIndex=3).GetChildren()

    def __repr__(self) -> str:
        return f"<wxauto Contact Window at {hex(id(self))}>"

    def _show(self):
        self.HWND = FindWindow(classname=self._clsname)
        win32gui.ShowWindow(self.HWND, 1)
        win32gui.SetWindowPos(self.HWND, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.HWND, -2, 0, 0, 0, 0, 3)
        self.UiaAPI.SwitchToThisWindow()

    def GetFriendNum(self):
        """获取好友人数"""
        wxlog.debug('获取好友人数')
        numText = self.Sidebar.PaneControl(Name='全部').TextControl(foundIndex=2).Name
        return int(re.findall('\d+', numText)[0])
    
    def Search(self, keyword):
        """搜索好友

        Args:
            keyword (str): 搜索关键词
        """
        wxlog.debug(f"搜索好友：{keyword}")
        if WxParam.MOUSE_MOVE:
            self._show()
        self.ContactBox.EditControl(Name="搜索").Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        self.ContactBox.ShortcutSelectAll(click=False)
        self.ContactBox.Input(keyword)

    def GetAllFriends(self, speed: int = 5):
        """获取好友列表
        
        Args:
            speed (int, optional): 滚动速度，数值越大滚动越快，但是太快可能导致遗漏，建议速度1-5之间
            
        Returns:
            list: 好友列表
        """
        wxlog.debug("获取好友列表")
        if WxParam.MOUSE_MOVE:
            self._show()
        
        contacts_list = []

        contact_ele_list = self.ContactBox.ListControl().GetChildren()

        n = 0
        idx = 0
        while n < 5:
            for _, ele in enumerate(contact_ele_list):
                contacts_info = {
                    'nickname': ele.TextControl().Name.replace('</em>', '').replace('<em>', ''),
                    'remark': ele.ButtonControl(foundIndex=2).Name.replace('</em>', '').replace('<em>', ''),
                    'tags': ele.ButtonControl(foundIndex=3).Name.replace('</em>', '').replace('<em>', '').split('，'),
                }
                if contacts_info.get('remark') in ('添加备注', ''):
                    contacts_info['remark'] = None
                if contacts_info.get('tags') in (['添加标签'], ['']):
                    contacts_info['tags'] = None
                # if contacts_info not in contacts_list:
                contacts_list.append(contacts_info)

            lastid = ele.GetRuntimeId()
            top_ele = ele.BoundingRectangle.top

            n = 0
            while n < 5:
                nowlist = [i.GetRuntimeId() for i in self.ContactBox.ListControl().GetChildren()]
                if lastid != nowlist[-1] and lastid in nowlist and top_ele == ele.BoundingRectangle.top:
                    break

                if top_ele == ele.BoundingRectangle.top:
                    self.ContactBox.WheelDown(wheelTimes=speed)
                    time.sleep(0.01)
                    n += 1
                top_ele = ele.BoundingRectangle.top

            while True:
                nowlist = [i.GetRuntimeId() for i in self.ContactBox.ListControl().GetChildren()]
                if lastid in nowlist:
                    break
                time.sleep(0.01)
            idx = nowlist.index(lastid) + 1
            contact_ele_list = self.ContactBox.ListControl().GetChildren()[idx:]
        return contacts_list
    
    def GetAllRecentGroups(self, speed: int = 1, wait=0.05):
        """获取群列表
        
        Args:
            speed (int, optional): 滚动速度，数值越大滚动越快，但是太快可能导致遗漏，建议速度1-3之间
            wait (float, optional): 滚动等待时间，建议和speed一起调整，直至适合你电脑配置和微信群数量达到平衡，不遗漏数据
            
        Returns:
            list: 群列表
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        self.UiaAPI.PaneControl(Name='最近群聊').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        group_list_control = self.UiaAPI.PaneControl(Name='最近群聊').GetParentControl().ListControl()
        groups = []

        n = 0
        idx = 0
        group_list_items = group_list_control.GetChildren()
        while n < 5:
            for _, item in enumerate(group_list_items):
                text_control1, text_control2 = item.TextControl().GetParentControl().GetChildren()
                group_name = text_control1.Name
                group_members = text_control2.Name.strip('(').strip(')')
                groups.append((group_name, group_members))

            lastid = item.GetRuntimeId()
            top_ele = item.BoundingRectangle.top

            n = 0
            while n < 5:
                nowlist = [i.GetRuntimeId() for i in group_list_control.GetChildren()]
                if lastid != nowlist[-1] and lastid in nowlist and top_ele == item.BoundingRectangle.top:
                    break

                if top_ele == item.BoundingRectangle.top:
                    group_list_control.WheelDown(wheelTimes=speed)
                    time.sleep(wait)
                    n += 1
                top_ele = item.BoundingRectangle.top

            while True:
                nowlist = [i.GetRuntimeId() for i in group_list_control.GetChildren()]
                if lastid in nowlist:
                    break
                time.sleep(0.01)
            idx = nowlist.index(lastid) + 1
            group_list_items = group_list_control.GetChildren()[idx:]
        return groups
    
    def Close(self):
        """关闭联系人窗口"""
        wxlog.debug('关闭联系人窗口')
        
        self.UiaAPI.SendKeys('{Esc}')


class ContactElement:
    def __init__(self, ele):
        self.element = ele
        self.nickname = ele.TextControl().Name
        self.remark = ele.ButtonControl(foundIndex=2).Name
        self.tags = ele.ButtonControl(foundIndex=3).Name.split('，')

    def __repr__(self) -> str:
        return f"<wxauto Contact Element at {hex(id(self))} ({self.nickname}: {self.remark})>"
    
    def EditRemark(self, remark: str):
        """修改好友备注名
        
        Args:
            remark (str): 新备注名
        """
        wxlog.debug(f"修改好友备注名：{self.nickname} --> {remark}")
        self.element.ButtonControl(foundIndex=2).Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        self.element.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
        self.element.Input(remark)
        self.element.SendKeys('{Enter}')

class AddTalkMemberWnd:
    def __init__(self, wx) -> None:
        self._wx = wx
        self.UiaAPI = self._wx.UiaAPI.WindowControl(ClassName='AddTalkMemberWnd', searchDepth=3)
        self.searchbox = self.UiaAPI.EditControl(Name='搜索')

    def __repr__(self) -> str:
        return f"<wxauto Add Member Window at {hex(id(self))}>"
    
    def Search(self, keyword):
        """搜索好友
        
        Args:
            keyword (str): 搜索关键词
        """
        wxlog.debug(f"搜索好友：{keyword}")
        self.searchbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        self.searchbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
        self.searchbox.Input(keyword)
        time.sleep(0.5)
        result = self.UiaAPI.ListControl().GetChildren()
        return result
    
    def Add(self, keyword):
        """搜索并添加好友
        
        Args:
            keyword (str): 搜索关键词
        """
        result = self.Search(keyword)
        if len(result) == 1:
            result[0].ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            wxlog.debug(f"添加好友：{keyword}")
        elif len(result) > 1:
            wxlog.warning(f"搜索到多个好友：{keyword}")
        else:
            wxlog.error(f"未找到好友：{keyword}")

    def Submit(self):
        self.UiaAPI.ButtonControl(Name='完成').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        wxlog.debug("发起多人语音聊天")
        # confirmdlg = self.UiaAPI.WindowControl(ClassName='ConfirmDialog')

class AddMemberWnd:
    def __init__(self, wx) -> None:
        self._wx = wx
        self.UiaAPI = self._wx.UiaAPI.WindowControl(ClassName='AddMemberWnd', searchDepth=3)
        self.searchbox = self.UiaAPI.EditControl(Name='搜索')

    def __repr__(self) -> str:
        return f"<wxauto Add Member Window at {hex(id(self))}>"
    
    def Search(self, keyword):
        """搜索好友
        
        Args:
            keyword (str): 搜索关键词
        """
        wxlog.debug(f"搜索好友：{keyword}")
        self.searchbox.Input(keyword)
        time.sleep(0.5)
        result = self.UiaAPI.ListControl(Name="请勾选需要添加的联系人").GetChildren()
        return result
    
    def Add(self, keyword):
        """搜索并添加好友
        
        Args:
            keyword (str): 搜索关键词
        """
        result = self.Search(keyword)
        if len(result) == 1:
            result[0].ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            wxlog.debug(f"添加好友：{keyword}")
        elif len(result) > 1:
            wxlog.warning(f"搜索到多个好友：{keyword}")
        else:
            wxlog.error(f"未找到好友：{keyword}")

    def Submit(self):
        self.UiaAPI.ButtonControl(Name='完成').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        wxlog.debug("提交添加好友请求")
        confirmdlg = self.UiaAPI.WindowControl(ClassName='ConfirmDialog')
        t0 = time.time()
        while True:
            if time.time() - t0 > 5:
                raise TimeoutError("新增群好友等待超时")
            if not self.UiaAPI.Exists(0.1):
                wxlog.debug("新增群好友成功，无须再次确认")
                return
            if confirmdlg.Exists(0.1):
                wxlog.debug("新增群好友成功，确认添加")
                time.sleep(1)
                # confirmdlg.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                confirmdlg.SendKeys('{ENTER}')
                return

    def Close(self):
        self.UiaAPI.SendKeys('{ESC}')

class DeleteMemberWnd:
    def __init__(self, wx) -> None:
        self._wx = wx
        self.UiaAPI = self._wx.UiaAPI.WindowControl(ClassName='DeleteMemberWnd', searchDepth=3)
        self.searchbox = self.UiaAPI.EditControl(Name='搜索')

    def __repr__(self) -> str:
        return f"<wxauto Add Member Window at {hex(id(self))}>"
    
    def Search(self, keyword):
        """搜索群成员
        
        Args:
            keyword (str): 搜索关键词
        """
        wxlog.debug(f"搜索群成员：{keyword}")
        self.searchbox.Click()
        self.searchbox.ShortcutSelectAll()
        self.searchbox.Input(keyword)
        time.sleep(0.5)
        result = self.UiaAPI.ListControl(Name="请勾选需要添加的联系人").GetChildren()
        return result
    
    def Remove(self, keyword):
        """搜索并移出群成员
        
        Args:
            keyword (str): 搜索关键词
        """
        result = self.Search(keyword)
        if len(result) == 1:
            result[0].ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            target = result[0].ButtonControl().Name
            wxlog.debug(f"移出群成员：{target}")
            return target
        elif len(result) > 1:
            wxlog.warning(f"搜索到多个群成员：{keyword}")
            for item in result:
                if item.TextControl().Exists(0) and (item.TextControl().Name == keyword or item.ButtonControl().Name == keyword):
                    item.ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    wxlog.debug(f"移出完全匹配项群成员：{keyword}")
                    return item.ButtonControl().Name
        else:
            wxlog.error(f"未找到群成员：{keyword}")

    def Submit(self):
        """提交移出群成员请求"""
        submit_btn = self.UiaAPI.ButtonControl(Name='完成')
        threading.Thread(target=submit_btn.Click).start()
        time.sleep(0.5)
        wxlog.debug("提交移出群成员请求")
        confirmdlg = self.UiaAPI.WindowControl(ClassName='ConfirmDialog')
        t0 = time.time()
        while True:
            wxlog.debug("等待移出群成员确认对话框")
            if time.time() - t0 > 5:
                raise TimeoutError("移出群成员等待超时")
            if not self.UiaAPI.Exists(0.1):
                wxlog.debug("移出群成员成功，无须再次确认")
                return
            if confirmdlg.Exists(0.1):
                wxlog.debug("移出群成员成功")
                time.sleep(1)
                # confirmdlg.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                confirmdlg.SendKeys('{ENTER}')
                return

    def Close(self):
        self.UiaAPI.SendKeys('{ESC}')
        
class GroupMemberElement:
    def __init__(self, ele, wx) -> None:
        self.UiaAPI = ele
        self._wx = wx

    def __repr__(self) -> str:
        return f"<wxauto Group Member Element at {hex(id(self))}>"
    
    @property
    def nickname(self):
        return self.UiaAPI.Name
    
    def add_friend(self, addmsg=None, remark=None, tags=None, permission='朋友圈', **kwargs):
        """添加新的好友

        Args:
            addmsg (str, optional): 添加好友的消息
            remark (str, optional): 备注名
            tags (list, optional): 标签列表
            permission (str, optional): 朋友圈权限, 可选值：'朋友圈', '仅聊天'

        Returns:
            int
            0 - 添加失败
            1 - 发送请求成功
            2 - 已经是好友
            3 - 对方不允许通过群聊添加好友
                
        Example:
            >>> addmsg = '你好，我是xxxx'      # 添加好友的消息
            >>> remark = '备注名字'            # 备注名
            >>> tags = ['朋友', '同事']        # 标签列表
            >>> msg.add_friend(keywords, addmsg=addmsg, remark=remark, tags=tags)
        """
        returns = {
            '添加失败': 0,
            '发送请求成功': 1,
            '已经是好友': 2,
            '对方不允许通过群聊添加好友': 3
        }
        # self._wx._show()
        roominfoWnd = self._wx.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        bias = 150
        if 'bias' in kwargs:
            bias = kwargs['bias']
        RollIntoView(roominfoWnd, self.UiaAPI, equal=True, bias=bias)
        self.UiaAPI.Click(simulateMove=False, move=True)
        contactwnd = self._wx.UiaAPI.PaneControl(ClassName='ContactProfileWnd')
        if not contactwnd.Exists(1):
            return returns['添加失败']
        addbtn = contactwnd.ButtonControl(Name='添加到通讯录')
        isfriend = not addbtn.Exists(0.2)
        if isfriend:
            contactwnd.SendKeys('{Esc}')
            return returns['已经是好友']
        else:
            addbtn.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            NewFriendsWnd = self._wx.UiaAPI.WindowControl(ClassName='WeUIDialog')
            AlertWnd = self._wx.UiaAPI.WindowControl(ClassName='AlertDialog')

            t0 = time.time()
            status = 0
            while time.time() - t0 < 5:
                if NewFriendsWnd.Exists(0.1):
                    status = 1
                    break
                elif AlertWnd.Exists(0.1):
                    status = 2
                    break
            
            if status == 0:
                return returns['添加失败']
            elif status == 2:
                AlertWnd.ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return returns['对方不允许通过群聊添加好友']
            elif status == 1:
                tipscontrol = NewFriendsWnd.TextControl(Name="你的联系人较多，添加新的朋友时需选择权限")

                permission_sns = NewFriendsWnd.CheckBoxControl(Name='聊天、朋友圈、微信运动等')
                permission_chat = NewFriendsWnd.CheckBoxControl(Name='仅聊天')
                if tipscontrol.Exists(0.5):
                    permission_sns = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='朋友圈')
                    permission_chat = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='仅聊天')

                if addmsg:
                    msgedit = NewFriendsWnd.TextControl(Name="发送添加朋友申请").GetParentControl().EditControl()
                    msgedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    msgedit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                    msgedit.Input(addmsg)

                if remark:
                    remarkedit = NewFriendsWnd.TextControl(Name='备注名').GetParentControl().EditControl()
                    remarkedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    remarkedit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                    remarkedit.Input(remark)

                if tags:
                    tagedit = NewFriendsWnd.TextControl(Name='标签').GetParentControl().EditControl()
                    for tag in tags:
                        tagedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                        tagedit.Input(tag)
                        NewFriendsWnd.PaneControl(ClassName='DropdownWindow').TextControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                
                if permission == '朋友圈':
                    permission_sns.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                elif permission == '仅聊天':
                    permission_chat.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

                # while NewFriendsWnd.Exists(0.3):
                NewFriendsWnd.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return returns['发送请求成功']
            return returns['添加失败']

class SessionElement:
    def __init__(self, item, debut_output=True):
        self.control = item
        self.name = temp_control.Name if (temp_control := item.GetProgenyControl(4, control_type='TextControl')) else None
        self.time = temp_control.Name if (temp_control := item.GetProgenyControl(4, 1, control_type='TextControl')) else None
        self.content = temp_control.Name if (temp_control := item.GetProgenyControl(4, 2, control_type='TextControl')) else None
        self.isnew = (new_tag_control := item.GetProgenyControl(2, 2)) is not None
        if self.isnew and new_tag_control.Name:
            self.new_count = int(new_tag_control.Name)
        else:
            new_text = re.findall(r'\[(\d+)条\]', str(self.content))
            self.new_count = new_text[0] if new_text else 0
            if not self.new_count:
                self.new_count = 0
        if debut_output:
            wxlog.debug(f"============== 【{self.name}】 ==============")
            wxlog.debug(f"最后一条消息时间: {self.time}")
            wxlog.debug(f"最后一条消息内容: {self.content}")
            wxlog.debug(f"是否有新消息: {self.isnew}")
            wxlog.debug(f"新消息数量: {self.new_count}")


class Message:
    type = 'message'
    mtype = ''

    def __getitem__(self, index):
        return self.info[index]
    
    def __str__(self):
        return self.content
    
    def __repr__(self):
        return str(self.info[:2])
    
    def roll_into_view(self):
        if RollIntoView(self.chatbox.ListControl(), self.control, equal=True) == 'not exist':
            wxlog.warning('消息目标控件不存在，无法滚动至显示窗口')
            return WxResponse.failure('消息目标控件不存在，无法滚动至显示窗口')
        return WxResponse.success('成功')
    
    @property
    def details(self):
        if hasattr(self, '_details'):
            return self._details
        chat_info = {
            'id': self.id,
            'type': self.type,
            'sender': self.sender,
            'content': self.content,
        }
        if self.type == 'time':
            chat_info['time'] = self.time
        elif self.type == 'friend':
            chat_info['sender_remark'] = self.sender_remark
        if self.chatbox.ControlTypeName == 'WindowControl':
            chat_name_control = self.chatbox.GetProgenyControl(12)
        else:
            chat_name_control = self.chatbox.GetProgenyControl(11)
        chat_name_control_list = chat_name_control.GetParentControl().GetChildren()
        chat_name_control_count = len(chat_name_control_list)
        if chat_name_control_count == 1:
            if self.chatbox.ButtonControl(Name='公众号主页').Exists(0):
                chat_info['chat_type'] = 'official'
            else:
                chat_info['chat_type'] = 'friend'
            chat_info['chat_name'] = chat_name_control.Name
        elif chat_name_control_count >= 2:
            try:
                chat_info['group_member_count'] = int(chat_name_control_list[1].Name.replace('(', '').replace(')', ''))
                chat_info['chat_type'] = 'group'
                chat_info['chat_name'] = chat_name_control.Name.replace(chat_name_control_list[1].Name, '')
            except:
                chat_info['chat_type'] = 'friend'
                chat_info['chat_name'] = chat_name_control.Name

            ori_chat_name_control = chat_name_control.GetParentControl().GetParentControl().TextControl(searchDepth=1)
            if ori_chat_name_control.Exists(0):
                chat_info['chat_remark'] = chat_info['chat_name']
                chat_info['chat_name'] = ori_chat_name_control.Name
    
        money_control = self.control.TextControl(RegexName='￥.*?')
        if self.content == '微信转账' and money_control.Exists(0):
            chat_info['money'] = float(money_control.Name.replace('￥',''))
        self._details = chat_info
        return self._details
    

class SysMessage(Message):
    type = 'sys'
    
    def __init__(self, info, control, wx):
        self.info = info
        self.control = control
        self.wx = wx
        self.sender = info[0]
        self.content = info[1]
        self.id = info[-1]
        _is_main_window = hasattr(wx, 'ChatBox')
        self.chatbox = wx.ChatBox if _is_main_window else wx.UiaAPI
        
        wxlog.debug(f"【系统消息】{self.content}")

    
    # def __repr__(self):
    #     return f'<wxauto SysMessage at {hex(id(self))}>'
    

class TimeMessage(Message):
    type = 'time'
    
    def __init__(self, info, control, wx):
        self.info = info
        self.control = control
        self.wx = wx
        self.time = ParseWeChatTime(info[1])
        self.sender = info[0]
        self.content = info[1]
        self.id = info[-1]
        _is_main_window = hasattr(wx, 'ChatBox')
        self.chatbox = wx.ChatBox if _is_main_window else wx.UiaAPI
        
        wxlog.debug(f"【时间消息】{self.time}")
    
    # def __repr__(self):
    #     return f'<wxauto TimeMessage at {hex(id(self))}>'
    

class RecallMessage(Message):
    type = 'recall'
    
    def __init__(self, info, control, wx):
        self.info = info
        self.control = control
        self.wx = wx
        self.sender = info[0]
        self.content = info[1]
        self.id = info[-1]
        _is_main_window = hasattr(wx, 'ChatBox')
        self.chatbox = wx.ChatBox if _is_main_window else wx.UiaAPI
        
        wxlog.debug(f"【撤回消息】{self.content}")
    
    # def __repr__(self):
    #     return f'<wxauto RecallMessage at {hex(id(self))}>'
    

class SelfMessage(Message):
    type = 'self'
    
    def __init__(self, info, control, obj):
        self.info = info
        self.control = control
        self._winobj = obj
        self.sender = info[0]
        self.content = info[1]
        self.id = info[-1]
        _is_main_window = hasattr(obj, 'ChatBox')
        self.chatbox = obj.ChatBox if _is_main_window else obj.UiaAPI
        
        wxlog.debug(f"【自己消息】{self.content}")
    
    # def __repr__(self):
    #     return f'<wxauto SelfMessage at {hex(id(self))}>'

    def click(self):
        """点击该消息"""
        wxlog.debug(f'点击自己消息：{self.sender} | {self.content}')
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=-xbias, move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def quote(self, msg, at=None):
        """引用该消息

        Args:
            msg (str): 引用的消息内容

        Returns:
            bool: 是否成功引用
        """
        wxlog.debug(f'发送引用消息：{msg}  --> {self.sender} | {self.content}')
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        filecontrol = self.chatbox.ButtonControl(Name='发送文件')
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        headcontrol.RightClick(x=-xbias, simulateMove=False)
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        quote_option = menu.MenuItemControl(Name="引用")
        if not quote_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法引用')
            return WxResponse.failure('该消息当前状态无法引用')
        quote_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        editbox = self.chatbox.EditControl(searchDepth=15)
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                raise TimeoutError(f'发送消息超时 --> {msg}')
            SetClipboardText(msg)
            filecontrol.Click(y=int(xbias/1.5), move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE, click=False)
            if editbox.GetValuePattern().Value.replace('\r￼', ''):
                break
        
        if at:
            if isinstance(at, str):
                at = [at]
            for i in at:
                editbox.Input('@'+i)
                atwnd = self._winobj.UiaAPI.PaneControl(ClassName='ChatContactMenu')
                if atwnd.Exists(maxSearchSeconds=0.1):
                    self._winobj.UiaAPI.SendKeys('{ENTER}')

        time.sleep(0.1)
        editbox.SendKeys(WxParam.SHORTCUT_SEND)
        # headcontrol.RightClick()
        return WxResponse.success()
    
    def parse_url(self, timeout=10):
        """解析消息中的链接

        Args:
            timeout (int, optional): 超时时间

        Returns:
            str: 链接地址
        """
        if self.content not in ('[链接]', '[音乐]'):
            return None
        if self.content == '[链接]' and (
            self.control.TextControl(Name="邀请你加入群聊").Exists(0)\
            or self.control.TextControl(Name="Group Chat Invitation").Exists(0)):
            return '[链接](群聊邀请)'
        if not self.control.PaneControl().Exists(0):
            return None
        link_control_list = self.control.PaneControl().GetChildren()
        if len(link_control_list) < 2:
            return None
        link_control = link_control_list[1]
        if not link_control.ButtonControl().Exists(0):
            return None

        self.control.TextControl().Click()
        t0 = time.time()
        while not FindWindow('Chrome_WidgetWin_0', '微信'):
            if time.time() - t0 > timeout:
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)
        wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)

        while not wxbrowser.DocumentControl().GetChildren():
            if time.time() - t0 > timeout:
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)

        wxbrowser.PaneControl(searchDepth=1, ClassName='').MenuItemControl(Name="更多").Click()
        wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)
        copyurl = wxbrowser.PaneControl(ClassName='Chrome_WidgetWin_0').MenuItemControl(Name='复制链接')
        copyurl.Click()
        url = ReadClipboardData()['13']
        wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
        return url
    
    @tenacity.retry(wait=tenacity.wait_fixed(0.2), stop=tenacity.stop_after_delay(10))
    def forward(self, friend):
        """转发该消息
        
        Args:
            friend (str): 转发给的好友昵称、备注或微信号
        
        Returns:
            bool: 是否成功转发
        """
        wxlog.debug(f'转发消息：{self.sender} --> {friend} | {self.content}')
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        roll_win = self.chatbox.ListControl()
        RollIntoView(roll_win, headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=xbias, move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        headcontrol.RightClick(x=-xbias, simulateMove=False)
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        forward_option = menu.MenuItemControl(Name="转发...")
        if not forward_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法转发')
            return WxResponse.failure('该消息当前状态无法转发')
        forward_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        contactwnd = self._winobj.UiaAPI.WindowControl(ClassName='SelectContactWnd')
        edit = contactwnd.EditControl()

        if isinstance(friend, str):
            SetClipboardText(friend)
            while not edit.HasKeyboardFocus:
                edit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                time.sleep(0.1)
            edit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            edit.ShortcutPaste(move=WxParam.MOUSE_MOVE)
            checkbox = contactwnd.ListControl().CheckBoxControl()
            if checkbox.Exists(1):
                checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                contactwnd.ButtonControl(Name='发送').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return WxResponse.success()
            else:
                contactwnd.SendKeys('{Esc}')
                wxlog.debug(f'未找到好友：{friend}')
                return WxResponse.failure(f'未找到好友：{friend}')
            
        elif isinstance(friend, list):
            n = 0
            fail = []
            multiselect = contactwnd.ButtonControl(Name='多选')
            if multiselect.Exists(0):
                multiselect.Click()
            for i in friend:
                SetClipboardText(i)
                while not edit.HasKeyboardFocus:
                    edit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    time.sleep(0.1)
                edit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                edit.ShortcutPaste(move=WxParam.MOUSE_MOVE, click=False)
                checkbox = contactwnd.ListControl().CheckBoxControl()
                if checkbox.Exists(1):
                    checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    n += 1
                else:
                    fail.append(i)
                    wxlog.debug(f"未找到转发对象：{i}")
            if n > 0:
                contactwnd.ButtonControl(RegexName='分别发送（\d+）').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                if n == len(friend):
                    return WxResponse.success()
                else:
                    return WxResponse.success('存在未转发成功名单', data=fail)
            else:
                contactwnd.SendKeys('{Esc}')
                wxlog.debug(f'所有好友均未未找到：{friend}')
                return WxResponse.failure(f'所有好友均未未找到：{friend}')
    
    def parse(self):
        """解析合并消息内容，当且仅当消息内容为合并转发的消息时有效"""
        wxlog.debug(f'解析合并消息内容：{self.sender} | {self.content}')
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=-xbias, simulateMove=False)
        chatrecordwnd = ChatRecordWnd()
        time.sleep(2)
        msgs = chatrecordwnd.GetContent()
        return msgs
    
    def tickle(self):
        """拍一拍好友"""
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        headcontrol.RightClick(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        tickle_option = menu.MenuItemControl(Name="拍一拍")
        if not tickle_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法拍一拍')
            return WxResponse.failure('该消息当前状态无法拍一拍')
        tickle_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        return WxResponse.success()

class FriendMessage(Message):
    type = 'friend'
    
    def __init__(self, info, control, obj):
        self.info = info
        self.control = control
        self._winobj = obj
        self.sender = info[0][0]
        self.sender_remark = info[0][1]
        self.content = info[1]
        self.id = info[-1]
        self.info[0] = info[0][0]
        _is_main_window = hasattr(obj, 'ChatBox')
        self.chatbox = obj.ChatBox if _is_main_window else obj.UiaAPI
        
        if self.sender == self.sender_remark:
            wxlog.debug(f"【好友消息】{self.sender}: {self.content}")
        else:
            wxlog.debug(f"【好友消息】{self.sender}({self.sender_remark}): {self.content}")
    
    # def __repr__(self):
    #     return f'<wxauto FriendMessage at {hex(id(self))}>'

    def click(self):
        """点击该消息"""
        wxlog.debug(f'点击消息：{self.sender} | {self.content}')
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=False)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=xbias, move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def quote(self, msg, at=None):
        """引用该消息

        Args:
            msg (str): 引用的消息内容

        Returns:
            bool: 是否成功引用
        """
        wxlog.debug(f'发送引用消息：{msg}  --> {self.sender} | {self.content}')
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        filecontrol = self.chatbox.ButtonControl(Name='发送文件')
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=False)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        headcontrol.RightClick(x=xbias, simulateMove=False)
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        quote_option = menu.MenuItemControl(Name="引用")
        if not quote_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法引用')
            return WxResponse.failure('该消息当前状态无法引用')
        quote_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        editbox = self.chatbox.EditControl(searchDepth=15)
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                raise TimeoutError(f'发送消息超时 --> {msg}')
            SetClipboardText(msg)
            filecontrol.Click(y=int(xbias/1.5), move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE, click=False)
            if editbox.GetValuePattern().Value.replace('\r￼', ''):
                break

        if at:
            if isinstance(at, str):
                at = [at]
            for i in at:
                editbox.Input('@'+i)
                atwnd = self._winobj.UiaAPI.PaneControl(ClassName='ChatContactMenu')
                if atwnd.Exists(maxSearchSeconds=0.1):
                    self._winobj.UiaAPI.SendKeys('{ENTER}')

        time.sleep(0.1)
        editbox.SendKeys(WxParam.SHORTCUT_SEND)
        return WxResponse.success()
    
    def parse_url(self, timeout=10):
        """解析消息中的链接

        Args:
            timeout (int, optional): 超时时间

        Returns:
            str: 链接地址
        """
        if self.content not in ('[链接]', '[音乐]'):
            return WxResponse.failure('消息内容不是链接')
        if self.content == '[链接]' and (
            self.control.TextControl(Name="邀请你加入群聊").Exists(0)\
            or self.control.TextControl(Name="Group Chat Invitation").Exists(0)):
            return '[链接](群聊邀请)'
        if not self.control.PaneControl().Exists(0):
            return None
        link_control_list = self.control.PaneControl().GetChildren()
        if len(link_control_list) < 2:
            return None
        link_control = link_control_list[1]
        if not link_control.ButtonControl().Exists(0):
            return None

        self.control.TextControl().Click()
        t0 = time.time()
        while not FindWindow('Chrome_WidgetWin_0', '微信'):
            if time.time() - t0 > timeout:
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)
        wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)

        while not wxbrowser.DocumentControl().GetChildren():
            if time.time() - t0 > timeout:
                raise TimeoutError('Open url timeout')
            time.sleep(0.01)

        wxbrowser.PaneControl(searchDepth=1, ClassName='').MenuItemControl(Name="更多").Click()
        wxbrowser = uia.PaneControl(ClassName="Chrome_WidgetWin_0", Name="微信", searchDepth=1)
        copyurl = wxbrowser.PaneControl(ClassName='Chrome_WidgetWin_0').MenuItemControl(Name='复制链接')
        copyurl.Click()
        url = ReadClipboardData()['13']
        wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
        return url
    
    @tenacity.retry(wait=tenacity.wait_fixed(0.2), stop=tenacity.stop_after_delay(10))
    def forward(self, friend):
        """转发该消息
        
        Args:
            friend (str): 转发给的好友昵称、备注或微信号
        
        Returns:
            bool: 是否成功转发
        """
        wxlog.debug(f'转发消息：{self.sender} --> {friend} | {self.content}')
        # self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        roll_win = self.chatbox.ListControl()
        RollIntoView(roll_win, headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=-xbias, move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        headcontrol.RightClick(x=xbias, simulateMove=False)
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        forward_option = menu.MenuItemControl(Name="转发...")
        if not forward_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法转发')
            return WxResponse.failure('该消息当前状态无法转发')
        forward_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        contactwnd = self._winobj.UiaAPI.WindowControl(ClassName='SelectContactWnd')
        edit = contactwnd.EditControl()

        if isinstance(friend, str):
            SetClipboardText(friend)
            while not edit.HasKeyboardFocus:
                edit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                time.sleep(0.1)
            edit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            edit.ShortcutPaste(move=WxParam.MOUSE_MOVE)
            checkbox = contactwnd.ListControl().CheckBoxControl()
            if checkbox.Exists(1):
                checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                contactwnd.ButtonControl(Name='发送').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return WxResponse.success()
            else:
                contactwnd.SendKeys('{Esc}')
                wxlog.debug(f'未找到好友：{friend}')
                return WxResponse.failure(f'未找到好友：{friend}')
            
        elif isinstance(friend, list):
            n = 0
            fail = []
            multiselect = contactwnd.ButtonControl(Name='多选')
            if multiselect.Exists(0):
                multiselect.Click()
            for i in friend:
                SetClipboardText(i)
                while not edit.HasKeyboardFocus:
                    edit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    time.sleep(0.1)
                edit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                edit.ShortcutPaste(move=WxParam.MOUSE_MOVE, click=False)
                checkbox = contactwnd.ListControl().CheckBoxControl()
                if checkbox.Exists(1):
                    checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    n += 1
                else:
                    fail.append(i)
                    wxlog.debug(f"未找到转发对象：{i}")
            if n > 0:
                contactwnd.ButtonControl(RegexName='分别发送（\d+）').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                if n == len(friend):
                    return WxResponse.success()
                else:
                    return WxResponse.success('存在未转发成功名单', data=fail)
            else:
                contactwnd.SendKeys('{Esc}')
                wxlog.debug(f'所有好友均未未找到：{friend}')
                return WxResponse.failure(f'所有好友均未未找到：{friend}')
    
    def parse(self):
        """解析合并消息内容，当且仅当消息内容为合并转发的消息时有效"""
        wxlog.debug(f'解析合并消息内容：{self.sender} | {self.content}')
        # self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        xbias = int(headcontrol.BoundingRectangle.width()*1.5)
        headcontrol.Click(x=xbias, simulateMove=False)
        chatrecordwnd = ChatRecordWnd()
        time.sleep(2)
        msgs = chatrecordwnd.GetContent()
        # chatrecordwnd = uia.WindowControl(ClassName='ChatRecordWnd', searchDepth=1)
        # msgitems = chatrecordwnd.ListControl().GetChildren()
        # msgs = []
        # for msgitem in msgitems:
        #     textcontrols = [i for i in GetAllControl(msgitem) if i.ControlTypeName == 'TextControl']
        #     who = textcontrols[0].Name
        #     time = textcontrols[1].Name
        #     try:
        #         content = textcontrols[2].Name
        #     except IndexError:
        #         content = ''
        #     msgs.append(([who, content, ParseWeChatTime(time)]))
        # chatrecordwnd.SendKeys('{Esc}')
        return msgs
    
    def sender_info(self):
        """获取好友信息"""
        wxlog.debug(f"获取好友信息：{self.sender}")
        if hasattr(self, 'contact_info'):
            return self.contact_info
        # contact_info = {
        #     "nickname": None,
        #     "id": None,
        #     "remark": None,
        #     "tags": None,
        #     "source": None,
        #     "signature": None,
        # }
        
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        headcontrol.Click(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))
        profile = ProfileWnd(self._winobj)
        self.contact_info = profile.contact_info
        profile.Close()
        return self.contact_info
        # contactwnd = self._winobj.UiaAPI.PaneControl(ClassName='ContactProfileWnd')
        # if not contactwnd.Exists(1):
        #     return 
        
        # def extract_info(contactwnd):
        #     if contactwnd.ControlTypeName == "TextControl":
        #         text = contactwnd.Name
        #         if text.startswith("昵称："):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling:
        #                 contact_info["nickname"] = sibling.Name.strip()
        #         elif text.startswith("微信号："):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling:
        #                 contact_info["id"] = sibling.Name.strip()
        #         elif text.startswith("备注"):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling and sibling.TextControl().Exists(0):
        #                 contact_info["remark"] = sibling.TextControl().Name.strip()
        #         elif text.startswith("标签"):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling:
        #                 contact_info["tags"] = sibling.Name.strip()
        #         elif text.startswith("来源"):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling:
        #                 contact_info["source"] = sibling.Name.strip()
        #         elif text.startswith("个性签名"):
        #             sibling = contactwnd.GetNextSiblingControl()
        #             if sibling:
        #                 contact_info["signature"] = sibling.Name.strip()

        #     for child in contactwnd.GetChildren():
        #         extract_info(child)
        # extract_info(contactwnd)
        # contactwnd.SendKeys('{Esc}')
        # return contact_info
    
    def modify(self, remark=None, tags=None):
        """修改好友信息

        Args:
            remark (str, optional): 备注名
            tags (list, optional): 标签列表

        Returns:
            bool: 是否修改成功
        """
        if all([not remark, not tags]):
            return WxResponse("修改失败，请至少输入一个参数")
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        headcontrol.Click(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))
        profile = ProfileWnd(self._winobj)
        self.contact_info = profile.contact_info
        result = profile.ModifyRemarkOrTags(remark, tags)
        profile.Close()
        return result

    def tickle(self):
        """拍一拍好友"""
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        headcontrol.RightClick(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))
        menu = self._winobj.UiaAPI.MenuControl(ClassName='CMenuWnd')
        tickle_option = menu.MenuItemControl(Name="拍一拍")
        if not tickle_option.Exists(maxSearchSeconds=0.1):
            wxlog.debug('该消息当前状态无法拍一拍')
            return WxResponse.failure('该消息当前状态无法拍一拍')
        tickle_option.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        return WxResponse.success()
    

    
    def add_friend(self, addmsg=None, remark=None, tags=None, permission='朋友圈'):
        """添加新的好友

        Args:
            addmsg (str, optional): 添加好友的消息
            remark (str, optional): 备注名
            tags (list, optional): 标签列表
            permission (str, optional): 朋友圈权限, 可选值：'朋友圈', '仅聊天'

        Returns:
            int
            0 - 添加失败
            1 - 发送请求成功
            2 - 已经是好友
            3 - 对方不允许通过群聊添加好友
                
        Example:
            >>> addmsg = '你好，我是xxxx'      # 添加好友的消息
            >>> remark = '备注名字'            # 备注名
            >>> tags = ['朋友', '同事']        # 标签列表
            >>> msg.add_friend(keywords, addmsg=addmsg, remark=remark, tags=tags)
        """
        returns = {
            '添加失败': 0,
            '发送请求成功': 1,
            '已经是好友': 2,
            '对方不允许通过群聊添加好友': 3
        }
        self._winobj._show()
        headcontrol = [i for i in self.control.GetFirstChildControl().GetChildren() if i.ControlTypeName == 'ButtonControl'][0]
        RollIntoView(self.chatbox.ListControl(), headcontrol, equal=True)
        headcontrol.Click(simulateMove=False, move=True)
        contactwnd = self._winobj.UiaAPI.PaneControl(ClassName='ContactProfileWnd')
        if not contactwnd.Exists(1):
            return returns['添加失败']
        addbtn = contactwnd.ButtonControl(Name='添加到通讯录')
        isfriend = not addbtn.Exists(0.2)
        if isfriend:
            contactwnd.SendKeys('{Esc}')
            return returns['已经是好友']
        else:
            addbtn.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            if type(self._winobj) == ChatWnd:
                NewFriendsWnd = self._winobj._wx.UiaAPI.WindowControl(ClassName='WeUIDialog')
                AlertWnd = self._winobj._wx.UiaAPI.WindowControl(ClassName='AlertDialog')
            else:
                NewFriendsWnd = self._winobj.UiaAPI.WindowControl(ClassName='WeUIDialog')
                AlertWnd = self._winobj.UiaAPI.WindowControl(ClassName='AlertDialog')
            


            t0 = time.time()
            status = 0
            while time.time() - t0 < 5:
                if NewFriendsWnd.Exists(0.1):
                    status = 1
                    break
                elif AlertWnd.Exists(0.1):
                    status = 2
                    break
            
            if status == 0:
                return returns['添加失败']
            elif status == 2:
                AlertWnd.ButtonControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return returns['对方不允许通过群聊添加好友']
            elif status == 1:
                tipscontrol = NewFriendsWnd.TextControl(Name="你的联系人较多，添加新的朋友时需选择权限")

                permission_sns = NewFriendsWnd.CheckBoxControl(Name='聊天、朋友圈、微信运动等')
                permission_chat = NewFriendsWnd.CheckBoxControl(Name='仅聊天')
                if tipscontrol.Exists(0.5):
                    permission_sns = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='朋友圈')
                    permission_chat = tipscontrol.GetParentControl().GetParentControl().TextControl(Name='仅聊天')

                if addmsg:
                    msgedit = NewFriendsWnd.TextControl(Name="发送添加朋友申请").GetParentControl().EditControl()
                    msgedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    msgedit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                    msgedit.Input(addmsg)

                if remark:
                    remarkedit = NewFriendsWnd.TextControl(Name='备注名').GetParentControl().EditControl()
                    remarkedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    remarkedit.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                    remarkedit.Input(remark)

                if tags:
                    tagedit = NewFriendsWnd.TextControl(Name='标签').GetParentControl().EditControl()
                    for tag in tags:
                        tagedit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                        tagedit.Input(tag)
                        NewFriendsWnd.PaneControl(ClassName='DropdownWindow').TextControl().Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

                if permission == '朋友圈':
                    permission_sns.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                elif permission == '仅聊天':
                    permission_chat.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

                NewFriendsWnd.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return returns['发送请求成功']
            return returns['添加失败']



message_types = {
    'SYS': SysMessage,
    'Time': TimeMessage,
    'Recall': RecallMessage,
    'Self': SelfMessage
}

def ParseMessage(data, control, wx):
    return message_types.get(data[0], FriendMessage)(data, control, wx)

class LoginWnd:
    _clsname = 'WeChatLoginWndForPC'

    def __init__(self, app_path=None):
        self.app_path = app_path
        self.UiaAPI = uia.PaneControl(ClassName=self._clsname, searchDepth=1)
        if not self.UiaAPI.Exists(0):
            self.open()

    def __repr__(self) -> str:
        return f"<wxauto LoginWnd Object at {hex(id(self))}>"

    def _show(self):
        self.HWND = self.UiaAPI.NativeWindowHandle
        win32gui.ShowWindow(self.HWND, 1)
        win32gui.SetWindowPos(self.HWND, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.HWND, -2, 0, 0, 0, 0, 3)
        self.UiaAPI.SwitchToThisWindow()

    def exists(self, wait=3):
        return self.UiaAPI.Exists(wait)

    @property
    def _app_path(self):
        if self.app_path:
            return self.app_path
        try:
            registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Tencent\WeChat", 0, winreg.KEY_READ)
            path, _ = winreg.QueryValueEx(registry_key, "InstallPath")
            winreg.CloseKey(registry_key)
            wxpath = os.path.join(path, "WeChat.exe")
            if os.path.exists(wxpath):
                self.app_path = wxpath
                return wxpath
            else:
                raise Exception('nof found')
        except WindowsError:
            print("未找到微信安装路径，请先打开微信启动页面再次尝试运行该方法！")

    def login(self, timeout=15):
        enter_button = self.UiaAPI.ButtonControl(Name='进入微信')
        qrcode = self.UiaAPI.ButtonControl(Name='二维码')
        if enter_button.Exists(0.5):
            enter_button.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

            dialog = self.UiaAPI.PaneControl(ClassName='WeUIDialog')
            wx = uia.WindowControl(ClassName='WeChatMainWndForPC', searchDepth=1)
            
            t0 = time.time()
            while True:
                if time.time() - t0 > timeout:
                    raise Exception('微信登录超时')
                if wx.Exists(0):
                    break
                elif dialog.Exists(0):
                    dialog_text = dialog.TextControl().Name
                    wxlog.debug(f"识别到弹窗：{dialog_text}")
                    dialog.SendKeys('{Esc}')
                    time.sleep(0.5)
                    return self.login(timeout)
                elif qrcode.Exists(0):
                    return WxResponse.failure("需扫码登录")
            return WxResponse.success()
        elif qrcode.Exists(0):
            return WxResponse.failure("需扫码登录")
        else:
            return WxResponse.failure("未找到登录按钮")

    def get_qrcode(self, path=None):
        """获取登录二维码

        Args:
            path (str): 二维码图片的保存路径，默认为None，即本地目录下的wxauto_qrcode文件夹

        
        Returns:
            str: 二维码图片的保存路径
        """
        self._show()
        if path is None:
            default_dir = os.path.realpath('wxauto_qrcode')
            if not os.path.exists(default_dir):
                os.mkdir(default_dir)
            path = os.path.join(default_dir, f'qrcode_{now_time()}.png')
        elif os.path.exists(path) and os.path.isdir(path):
            path = os.path.join(path, f'qrcode_{now_time()}.png')
        elif os.path.exists(os.path.dirname(path)) and path.endswith('.png'):
            pass
        else:
            raise ValueError('请输入正确的路径，或不指定path参数以使用默认路径')
        switch_account_button = self.UiaAPI.ButtonControl(Name='切换账号')
        if switch_account_button.Exists(0.5):
            switch_account_button.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        
        qrcode_control = self.UiaAPI.ButtonControl(Name='二维码')
        qrcode = qrcode_control.ScreenShot(path)
        return qrcode
    
    def shutdown(self):
        """关闭进程"""
        pid = self.UiaAPI.ProcessId
        os.system(f'taskkill /f /pid {pid}')

    def reopen(self):
        """重新打开"""
        self.shutdown()
        self.open()

    def open(self):
        path = self._app_path
        os.system(f'"start "" "{path}""')
        self.UiaAPI = uia.PaneControl(ClassName=self._clsname, searchDepth=1)
        if self.UiaAPI.Exists(10):
            return
        else:
            raise Exception('打开微信失败，请指定微信路径')

class WeChatMoments:
    _clsname = 'SnsWnd'

    def __init__(self, language='cn') -> None:
        self.language = language
        self.api = uia.WindowControl(ClassName=self._clsname, searchDepth=1)
        self.api.Exists(5)
        MainControl1 = [i for i in self.api.GetChildren() if not i.ClassName][0]
        self.ToolsBox = MainControl1.ToolBarControl(searchDepth=1)
        self.SnsBox = MainControl1.PaneControl(searchDepth=1)

        # Tools
        self.t_refresh = self.ToolsBox.ButtonControl(Name='刷新')

        self.GetMoments()

    def __repr__(self) -> str:
        return f'<WeChat Moments object at {hex(id(self))}>'

    def _show(self):
        self.HWND = FindWindow(classname=self._clsname)
        win32gui.ShowWindow(self.HWND, 1)
        win32gui.SetWindowPos(self.HWND, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.HWND, -2, 0, 0, 0, 0, 3)
        self.api.SwitchToThisWindow()

    def Close(self):
        self.api.SendKeys('{ESC}')
    
    def Refresh(self):
        self.t_refresh.Click(simulateMove=False)

    def GetMoments(self, next_page=False, speed1=3, speed2=1):
        if next_page:
            while True:
                self.api.WheelDown(wheelTimes=speed1)
                moments_controls = [i for i in self.SnsBox.ListControl(Name='朋友圈').GetChildren() if i.ControlTypeName=='ListItemControl']
                moments = [Moments(i, self) for i in moments_controls]
                if [i.GetRuntimeId() for i in moments_controls][0] == self._ids[-1]:
                    break
                time.sleep(0.05)

            while True:
                self.api.WheelDown(wheelTimes=speed2)
                moments_controls = [i for i in self.SnsBox.ListControl(Name='朋友圈').GetChildren() if i.ControlTypeName=='ListItemControl']
                moments = [Moments(i, self) for i in moments_controls]
                if [i.GetRuntimeId() for i in moments_controls][0] != self._ids[-1]:
                    break
                time.sleep(0.01)

        moments_controls = [i for i in self.SnsBox.ListControl(Name='朋友圈').GetChildren() if i.ControlTypeName=='ListItemControl']
        moments = [Moments(i, self) for i in moments_controls]
        self._ids = [i.GetRuntimeId() for i in moments_controls]
        return moments

class Moments:
    def __init__(self, control: uia.Control, parent) -> None:
        self.api = control
        self.parent = parent
        self.language = parent.language
        self.cmt = self.api.ButtonControl(Name='评论')
        self._parse()
    
    def __repr__(self) -> str:
        return f'<Moments {self.content}>'
    
    def __getattr__(self, name):
        try:
            return self.info[name]
        except KeyError:
            raise AttributeError(f"'Sns' object has no attribute '{name}'")
    
    def _parse(self):
        self.info = {
            'type': 'moment',
            'id': ''.join([str(i) for i in self.api.GetRuntimeId()]),
            'sender': '',
            'content': '',
            'time': '',
            'img_count': 0,
            'comments': [],
            'addr': '',
            'likes': []
        }
        content_control = self.api.GetProgenyControl(4, control_type='TextControl')
        self.info['sender'] = self.api.GetProgenyControl(4, control_type='ButtonControl').Name
        self.info['content'] = content_control.Name if content_control else ''
        self.info['time'] = self.api.ButtonControl(Name='评论').GetParentControl().TextControl().Name
        img_info_control = self.api.PaneControl(RegexName='包含\d+张图片')
        if img_info_control.Exists(0):
            self._img_controls = img_info_control.GetChildren()
            self.info['img_count'] = len(self._img_controls)
        else:
            self.info['img_count'] = 0
            self._img_controls = []

        if self.api.ListControl(Name='评论').Exists(0):
            self.info['comments'] = [i.Name for i in self.api.ListControl(Name='评论').GetChildren()]

        if self.api.TextControl(Name='广告').Exists(0):
            self.info['type'] = 'advertise'

        for i in range(10):
            text_control = self.api.GetProgenyControl(5, i, control_type='ButtonControl')
            if not text_control:
                break
            text_height = text_control.BoundingRectangle.height()
            if text_height == 18 and not self.info['addr']:
                self.info['addr'] = text_control.Name
            # elif text_height == 22:
            #     self.info['sender'] = text_control.Name

        like_control = self.api.GetProgenyControl(6, 0, control_type='TextControl')
        if like_control:
            self.info['likes'] = like_control.Name.split('，')
        
    def _download_pic(self, msgitem, savepath=''):
        RollIntoView(self.parent.api.ListControl(), msgitem, bias=self.parent.ToolsBox.BoundingRectangle.height()*2)
        msgitem.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        imgobj = WeChatImage()
        save_path = imgobj.Save(savepath=savepath)
        imgobj.Close()
        return save_path
    
    def SaveImages(self, save_index=None, save_path=''):
        """保存图片
        
        Args:
            save_index (int|list): 保存第几张图片（从0开始），默认为None，保存所有图片
            save_path (str): 绝对路径，包括文件名和后缀，例如："D:/Images/微信图片_xxxxxx.jpg"
                        （如果不填，则默认为当前脚本文件夹下，新建一个“微信图片(或视频)”的文件夹，保存在该文件夹内）
        """
        images = []
        if save_index:
            if isinstance(save_index, int):
                save_index = [save_index]
            elif not isinstance(save_index, list):
                raise TypeError("save_index must be int or list")

        for i, msgitem in enumerate(self._img_controls):
            if save_index and i not in save_index:
                continue
            imgpath = self._download_pic(msgitem, savepath=save_path)
            images.append(imgpath)
        return images
            
    def Like(self, like=True):
        RollIntoView(self.parent.SnsBox, self.cmt)
        self.cmt.Click(simulateMove=False)
        like_btn = self.parent.api.PaneControl(ClassName='SnsLikeToastWnd').ButtonControl(Name='赞')
        cancel_btn = self.parent.api.PaneControl(ClassName='SnsLikeToastWnd').ButtonControl(Name='取消')
        if like and like_btn.Exists(0):
            like_btn.Click(simulateMove=False)
        elif not like and cancel_btn.Exists(0):
            cancel_btn.Click(simulateMove=False)

    def Comment(self, text):
        RollIntoView(self.parent.SnsBox, self.cmt)
        self.cmt.Click(simulateMove=False)
        cmt_btn = self.parent.api.PaneControl(ClassName='SnsLikeToastWnd').ButtonControl(Name='评论')
        cmt_btn.Click(simulateMove=False)
        edit_control = self.parent.api.EditControl(Name='评论')
        SetClipboardText(text)
        edit_control.ShortcutPaste(click=True)
        edit_control.GetParentControl().ButtonControl(Name='发送').Click(simulateMove=False)

    def sender_info(self):
        
        contact_info = {
            "nickname": None,
            "id": None,
            "remark": None,
            "tags": None,
            "source": None,
            "signature": None,
        }
        
        self.parent._show()
        headcontrol = self.api.ButtonControl(Name=self.sender)
        RollIntoView(self.parent.api.ListControl(), headcontrol, equal=True, bias=self.parent.ToolsBox.BoundingRectangle.height()*2)
        headcontrol.Click(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))
        contactwnd = self.parent.api.PaneControl(ClassName='ContactProfileWnd')
        if not contactwnd.Exists(1):
            return 
        contact_info["nickname"] = contactwnd.ButtonControl().Name
        def extract_info(contactwnd):
            if contactwnd.ControlTypeName == "TextControl":
                text = contactwnd.Name
                if text.startswith("昵称："):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["nickname"] = sibling.Name.strip()
                elif text.startswith("微信号："):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["id"] = sibling.Name.strip()
                elif text.startswith("备注"):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling and sibling.TextControl().Exists(0):
                        contact_info["remark"] = sibling.TextControl().Name.strip()
                elif text.startswith("标签"):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["tags"] = sibling.Name.strip()
                elif text.startswith("来源"):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["source"] = sibling.Name.strip()
                elif text.startswith("个性签名"):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["signature"] = sibling.Name.strip()

            for child in contactwnd.GetChildren():
                extract_info(child)
        extract_info(contactwnd)
        contactwnd.SendKeys('{Esc}')
        return contact_info
    
class ProfileWnd:
    _clsname = 'ContactProfileWnd'

    def __init__(self, parent):
        self.api = parent.UiaAPI.PaneControl(ClassName=self._clsname)
        # PrintAllControlTree(self.api)
        self.contact_info = {
            "nickname": None,
            "id": None,
            "remark": None,
            "tags": None,
            "source": None,
            "signature": None,
        }
        self._extract_info(self.api)

    def _extract_info(self, contactwnd):
        if contactwnd.ControlTypeName == "TextControl":
            text = contactwnd.Name
            if text.startswith("昵称："):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["nickname"] = sibling.Name.strip()
            elif text.startswith("微信号："):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["id"] = sibling.Name.strip()
            elif text.startswith("群昵称："):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["group_nickname"] = sibling.Name.strip()
            elif text.startswith("地区："):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["area"] = sibling.Name.strip()
            elif text.startswith("备注"):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling and sibling.TextControl().Exists(0):
                    self.contact_info["remark"] = sibling.TextControl().Name.strip()
            elif text.startswith("标签"):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["tags"] = sibling.Name.strip()
            elif text.startswith("来源"):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["source"] = sibling.Name.strip()
            elif text.startswith("个性签名"):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["signature"] = sibling.Name.strip()
            elif text.startswith("共同群聊"):
                sibling = contactwnd.GetNextSiblingControl()
                if sibling:
                    self.contact_info["same_group"] = sibling.Name.strip()
            if self.contact_info['nickname'] is None:
                self.contact_info['nickname'] = self.api.ButtonControl().Name

        for child in contactwnd.GetChildren():
            self._extract_info(child)

    def _choose_menu(self, menu_name):
        more_control = self.api.ButtonControl(Name='更多')
        if not more_control.Exists(0):
            return 
        more_control.Click()
        menu_control = self.api.MenuControl()
        menu_dict = {item.Name: item for item in menu_control.ListControl().GetChildren() if item.Name}
        if menu_name in menu_dict:
            time.sleep(1)
            menu_dict[menu_name].Click(move=True, simulateMove=False)
            return WxResponse.success()
        else:
            return WxResponse.failure(f"未找到菜单{menu_name}")
        
    def ModifyRemarkOrTags(self, remark: str=None, tags: list=None):
        if all([not remark, not tags]):
            return WxResponse.failure("请至少传入一个参数")
        if not self._choose_menu("设置备注和标签"):
            wxlog.debug('该用户不支持修改备注和标签')
            return WxResponse.failure("该用户不支持修改备注和标签")
        dialogwnd = self.api.WindowControl(ClassName='WeUIDialog')
        if remark:
            edit = dialogwnd.TextControl(Name='备注名').GetParentControl().EditControl()
            edit.Click()
            edit.ShortcutSelectAll()
            SetClipboardText(remark)
            edit.ShortcutPaste(click=False)
        if tags:
            edit_btn = dialogwnd.TextControl(Name='标签').GetParentControl().ButtonControl()
            edit_btn.Click()
            tagwnd = dialogwnd.WindowControl(ClassName='StandardConfirmDialog')
            edit = tagwnd.EditControl(Name="输入标签")
            edit.Click()
            for tag in tags:
                edit.Input(tag)
                edit.SendKeys('{Enter}')
            tagwnd.ButtonControl(Name='确定').Click()
        dialogwnd.ButtonControl(Name='确定').Click()
        return WxResponse.success()
        
    def Close(self):
        self.api.SendKeys('{Esc}')

class SessionChatRoomDetailWnd:
    _cls = 'SessionChatRoomDetailWnd'

    def __init__(self, parent):
        self.parent = parent
        self.api = parent.UiaAPI.Control(ClassName=self._cls, searchDepth=1)

    def _edit(self, key, value):
        wxlog.debug(f'修改{key}为`{value}`')
        btn = self.api.TextControl(Name=key).GetParentControl().ButtonControl(Name=key)
        if btn.Exists(0):
            RollIntoView(self.api, btn)
            btn.Click()
        else:
            wxlog.debug(f'当前非群聊，无法修改{key}')
            return WxResponse.failure(f'当前非群聊，无法修改{key}')
        while True:
            edit_hwnd_list = [i[0] for i in GetAllWindowExs(self.api.NativeWindowHandle) if i[1] == 'EditWnd']
            if edit_hwnd_list:
                edit_hwnd = edit_hwnd_list[0]
                break
            btn.Click()
        edit_win32 = uia.Win32(edit_hwnd)
        edit_win32.shortcut_select_all()
        edit_win32.send_keys_shortcut('{DELETE}')
        edit_win32.input(value)
        edit_win32.send_keys_shortcut('{ENTER}')
        return WxResponse.success()

    def edit_group_name(self, new_name):
        """编辑群聊名称
        
        Args:
            new_name (str): 新的群聊名称
        """
        key = '群聊名称'
        return self._edit(key, new_name)

    def edit_remark(self, remark):
        """编辑群聊备注
        
        Args:
            remark (str): 新的群聊备注
        """
        if self.api.TextControl(Name='仅群主或管理员可以修改').Exists(0):
            wxlog.debug('当前用户无权限修改群聊备注')
            return False
        key = '备注'
        return self._edit(key, remark)

    def edit_my_name(self, my_name):
        """编辑我在本群的昵称
        
        Args:
            my_name (str): 新的昵称
        """
        key = '我在本群的昵称'
        return self._edit(key, my_name)

    def edit_group_notice(self, notice):
        """编辑群公告
        
        Args:
            notice (str | list): 新的群公告
        """
        self.api.TextControl(Name='群公告').GetParentControl().ButtonControl(Name='点击编辑群公告').Click()
        announcementwnd = uia.WindowControl(ClassName='ChatRoomAnnouncementWnd', searchDepth=1)
        if announcementwnd.TextControl(Name="仅群主和管理员可编辑").Exists(0):
            wxlog.debug('当前用户无权限修改群公告')
            announcementwnd.SendKeys('{Esc}')
            return False
        edit_btn_control = announcementwnd.ButtonControl(Name='编辑')
        if edit_btn_control.Exists(0):
            edit_btn_control.Click()
        edit = announcementwnd.EditControl()
        edit.Click()
        edit.ShortcutSelectAll(click=False)
        if isinstance(notice, str):
            # edit.Input(notice)
            SetClipboardText(notice)
            edit.ShortcutPaste(click=False)
        elif isinstance(notice, list) and notice:
            SetClipboardText(notice[0])
            edit.ShortcutPaste(click=False)
            for i in notice[1:]:
                announcementwnd.ButtonControl(Name='分隔线').Click()
                SetClipboardText(i)
                edit.ShortcutPaste(click=False)
        announcementwnd.ButtonControl(Name='完成').Click()
        announcementwnd.PaneControl(ClassName='WeUIDialog').ButtonControl(Name='发布').Click()

    def quit(self):
        """退出群聊"""
        quit_btn = self.api.ButtonControl(Name='退出群聊')
        if quit_btn.Exists(0):
            quit_btn.Click()
        else:
            wxlog.debug('当前非群聊，无法退出')
            return
        RollIntoView(self.api, quit_btn)
        quit_btn.Click()
        dialog = self.parent.UiaAPI.PaneControl(ClassName='WeUIDialog')
        if dialog.TextControl(RegexName='将退出群聊“.*?”').Exists(0):
            dialog.ButtonControl(Name='退出').Click()

    def close(self):
        try:
            self.api.SendKeys('{ESC}')
        except Exception as e:
            pass

class CMenuWnd:
    _clsname = 'CMenuWnd'

    def __init__(self, wx):
        self.parent = wx
        self.api = self.parent.UiaAPI.MenuControl(ClassName=self._clsname)

    def __repr__(self) -> str:
        return f"<wxauto CMenuWnd Object at {hex(id(self))}>"
    
    @property
    def option_controls(self):
        return self.api.ListControl().GetChildren()
    
    @property
    def option_names(self):
        return [c.Name for c in self.option_controls]
    
    def choose(self, item):
        if isinstance(item, int):
            self.option_controls[item].Click()
            return
        for c in self.option_controls:
            if c.Name == item:
                c.Click()
                break
        else:
            raise ValueError(f'未找到选项 {item}')
    
    def close(self):
        try:
            self.api.SendKeys('{ESC}')
        except Exception as e:
            pass

class ChatHistoryWnd:
    _clsname = 'FileManagerWnd'

    def __init__(self):
        self.api = uia.WindowControl(ClassName=self._clsname)

    def __repr__(self) -> str:
        return f"<wxauto ChatHistoryWnd Object at {hex(id(self))}>"
    
    def GetHistory(self):
        msgids = []
        msgs = []
        listcontrol = self.api.ListControl()
        while True:
            listitems = listcontrol.GetChildren()
            listitemids = [item.GetRuntimeId() for item in listitems]
            try:
                msgids = msgids[msgids.index(listitemids[0]):]
            except:
                pass
            for item in listitems:
                msgid = item.GetRuntimeId()
                if msgid not in msgids:
                    msgids.append(msgid)
                    msgs.append(parse_msg(item))
            topcontrol = listitems[-1]
            top = topcontrol.BoundingRectangle.top
            self.api.WheelDown(wheelTimes=3)
            time.sleep(0.1)
            if topcontrol.Exists(0.1) and top == topcontrol.BoundingRectangle.top and listitemids == [item.GetRuntimeId() for item in listcontrol.GetChildren()]:
                self.api.SendKeys('{Esc}')
                break
        return msgs

class WxBrowser:
    _cls_name = 'Chrome_WidgetWin_0'

    def __init__(self):
        self.UiaAPI = uia.PaneControl(ClassName=self._cls_name, Name="微信", searchDepth=1)
        self.more_button = self.UiaAPI.PaneControl(searchDepth=1, ClassName='').MenuItemControl(Name="更多")
        self.close_button = self.UiaAPI.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭")

    def search(self, url):
        search_btn_eles = [
            i for i in self.UiaAPI.TabControl().GetChildren() 
            if i.BoundingRectangle.height() == i.BoundingRectangle.width()
        ]
        if search_btn_eles:
            search_btn_eles[0].Click()
            edit = self.UiaAPI.TabControl().EditControl(Name='地址和搜索栏')
            SetClipboardText(url)
            edit.ShortcutPaste()
            edit.SendKeys('{Enter}')
            return True
        return False
    
    def forward(self, friend):
        t0 = time.time()
        while True:
            if time.time() - t0 > 10:
                # wxbrowser.PaneControl(searchDepth=1, ClassName='').ButtonControl(Name="关闭").Click()
                raise # '[链接]无法获取url'
            self.more_button.Click()
            time.sleep(0.5)
            copyurl = self.UiaAPI.PaneControl(ClassName='Chrome_WidgetWin_0').MenuItemControl(Name='转发给朋友')
            if copyurl.Exists(0):
                copyurl.Click()
                break
            self.UiaAPI.PaneControl(ClassName='Chrome_WidgetWin_0').SendKeys('{Esc}')
        sendwnd = SelectContactWnd()
        return sendwnd.send(friend)

    def send_card(self, url, friend):
        try:
            self.search(url)
            return self.forward(friend)
        except Exception as e:
            return WxResponse.failure(msg=str(e))
        finally:
            self.close()

    def close(self):
        try:
            self.close_button.Click()
        except:
            pass

class SelectContactWnd:
    _cls_name = 'SelectContactWnd'

    def __init__(self, parent=None):
        if parent:
            self.UiaAPI = parent.UiaAPI.PaneControl(ClassName=self._cls_name, searchDepth=1)
        else:
            self.UiaAPI = uia.WindowControl(ClassName=self._cls_name, searchDepth=1)
        self.UiaAPI.Exists(5)
        self.editbox = self.UiaAPI.EditControl()

    def send(self, friend):
        if isinstance(friend, str):
            SetClipboardText(friend)
            while not self.editbox.HasKeyboardFocus:
                self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                time.sleep(0.1)
            self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            self.editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE)
            checkbox = self.UiaAPI.ListControl().CheckBoxControl()
            if checkbox.Exists(1):
                checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                self.UiaAPI.ButtonControl(Name='发送').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                return WxResponse.success()
            else:
                self.UiaAPI.SendKeys('{Esc}')
                wxlog.debug(f'未找到好友：{friend}')
                return WxResponse.failure(f'未找到好友：{friend}')
            
        elif isinstance(friend, list):
            n = 0
            fail = []
            multiselect = self.UiaAPI.ButtonControl(Name='多选')
            if multiselect.Exists(0):
                multiselect.Click()
            for i in friend:
                SetClipboardText(i)
                while not self.editbox.HasKeyboardFocus:
                    self.editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    time.sleep(0.1)
                self.editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                self.editbox.ShortcutPaste(move=WxParam.MOUSE_MOVE, click=False)
                checkbox = self.UiaAPI.ListControl().CheckBoxControl()
                if checkbox.Exists(1):
                    checkbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    n += 1
                else:
                    fail.append(i)
                    wxlog.debug(f"未找到转发对象：{i}")
            if n > 0:
                self.UiaAPI.ButtonControl(RegexName='分别发送（\d+）').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                if n == len(friend):
                    return WxResponse.success()
                else:
                    return WxResponse.success('存在未转发成功名单', data=fail)
            else:
                self.UiaAPI.SendKeys('{Esc}')
                wxlog.debug(f'所有好友均未未找到：{friend}')
                return WxResponse.failure(f'所有好友均未未找到：{friend}')
            
class OfficialMenu:
    def __init__(self, control, official, wx):
        self.official = official
        self._wx = wx
        self.control = control
        self.name = control.Name

    def go(self, option=None):
        self.control.Click()
        if option:
            menu = CMenuWnd(self._wx)
            menu.choose(option)

    def __repr__(self):
        return f'<OfficialMenu({self.official} - {self.name})>'