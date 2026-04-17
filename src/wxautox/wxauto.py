# cython: language_level=3
"""
Author: Cluic
Update: 2025-01-16
Version: Plus Version 3.9.11.17.24
"""

from . import uiautomation as uia
from .languages import *
from .utils import *
from .elements import *
from .errors import *
from .color import *
import traceback
import threading
import time
import os
import re
try:
    from typing import Literal
except:
    from typing_extensions import Literal

class WeChat(WeChatBase, Listener):
    VERSION: str = '3.9.11.17'
    _clsname: str = 'WeChatMainWndForPC'
    lastmsgid: str = None
    _cache_chat_name: str = None
    listen: dict = dict()
    SessionItemList: list = []

    def __init__(
            self, 
            nickname: str = None,
            mouse_move: bool = False,
            myinfo: bool = False,
            language: Literal['cn', 'cn_t', 'en'] = 'cn', 
            debug: bool = False
        ) -> None:
        """微信UI自动化实例

        Args:
            language (str, optional): 微信客户端语言版本, 可选: cn简体中文  cn_t繁体中文  en英文, 默认cn, 即简体中文
        """
        self._nickname = nickname
        WxParam.MOUSE_MOVE = mouse_move
        set_debug(debug)
        self.language = language
        self.UiaAPI
        self._show()
        if myinfo:
            self.myinfo = self._my_info()
        self.usedmsgid = self._get_now_msgid()
        self._listener_start()
        print(f'初始化成功，获取到已登录窗口：{self.nickname}')

    def __repr__(self) -> str:
        return f'<wxauto object {self.nickname}>'
    
    def _checkversion(self):
        self.HWND = FindWindow(classname=self._clsname)
        wxpath = GetPathByHwnd(self.HWND)
        wxversion = GetVersionByPath(wxpath)
        if wxversion != self.VERSION:
            Warnings.lightred(self._lang('版本不一致', 'WARNING').format(wxversion, self.VERSION), stacklevel=2)
            return False
    
    def _show(self):
        win32gui.ShowWindow(self.HWND, 1)
        win32gui.SetWindowPos(self.HWND, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.HWND, -2, 0, 0, 0, 0, 3)
        self.UiaAPI.SwitchToThisWindow()

    def _refresh(self):
        self.UiaAPI.SendKeys('{Ctrl}{Alt}w', api=False)
        self.UiaAPI.SendKeys('{Ctrl}{Alt}w', api=False)

    def _get_listen_messages(self):
        # self.AddSubWindowListen(force=True)
        temp_listen = self.listen.copy()
        for who in temp_listen:
            # wxlog.debug(f"获取监听消息：{who}")
            chat = temp_listen.get(who, None)
            try:
                if chat is None or not chat.UiaAPI.Exists(0.1):
                    continue
            except:
                continue
            with self._lock:
                msg = chat.GetNewMessage(
                    savepic=chat.savepic, 
                    savevideo=chat.savevideo,
                    savefile=chat.savefile, 
                    savevoice=chat.savevoice,
                    parseurl=chat.parseurl
                )
                if msg:
                    wxlog.debug(f"获取到消息：{msg}")
                    if chat not in self._listener_messages:
                        self._listener_messages[chat] = msg
                    else:
                        self._listener_messages[chat].extend(msg)

    def _get_friend_details(self):
        params = ['昵称：', '微信号：', '地区：', '备注', '电话', '标签', '共同群聊', '个性签名', '来源', '朋友权限', '描述', '实名', '企业']
        info = {}
        controls = GetAllControlList(self.ChatBox)
        for _, i in enumerate(controls):
            rect = i.BoundingRectangle
            text = i.Name
            if text in params or (rect.width() == 57 and rect.height() == 20):
                info[text.replace('：', '')] = controls[_+1].Name
        if '昵称' not in info:
            info['备注'] = ''
            info['昵称'] = controls[0].Name
        wxlog.debug(f'获取到好友详情：{info}')
        return info
    

    def _goto_friends_tag(self, specific='A'):
        def find_letter_tag(self):
            items = self.SessionBox.ListControl().GetChildren()
            tags = []
            for index, item in enumerate(items[:-1]):
                tag_control = item.TextControl(RegexName='^[A-Z]$')
                if tag_control.Exists(0):
                    tags.append((items[index], items[index+1]))
            for tag in tags:
                if tag[0].TextControl().Name.upper() == specific.upper():
                    return tag
            if tags:
                return tags[0]
                
        now_letter = '0'
        action = []
        while True:
            if len(action) >= 6 and action[-6:] in ([0, 1] * 3, [1, 0] * 3):
                # print('检测到上下滚动切换，退出')
                return False
            items = find_letter_tag(self)
            if items is not None:
                item, item1 = items
                now_letter = item.TextControl().Name
                if specific.upper() == now_letter.upper():
                    RollIntoView(self.SessionBox.ListControl(), item1)
                    item1.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                    # print('找到指定字母，退出')
                    return True
            if specific.upper() < now_letter.upper():
                self.SessionBox.WheelUp(wheelTimes=3)
                time.sleep(0.001)
                action.append(0)
                continue
            else:
                self.SessionBox.WheelDown(wheelTimes=3)
                time.sleep(0.001)
                action.append(1)
                continue

    def _my_info(self):
        """获取当前登录用户信息"""
        
        contact_info = {
            "id": None,
            "area": None
        }

        self._show()
        self.A_MyIcon.Click(
            simulateMove=False, 
            move=WxParam.MOUSE_MOVE, 
            show_window=(not WxParam.MOUSE_MOVE), 
            return_pos=(not WxParam.MOUSE_MOVE)
        )
        contactwnd = self.UiaAPI.PaneControl(ClassName='ContactProfileWnd')
        if not contactwnd.Exists(1):
            return contact_info
        def extract_info(contactwnd):
            if contactwnd.ControlTypeName == "TextControl":
                text = contactwnd.Name
                if text.startswith("微信号："):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["id"] = sibling.Name.strip()
                elif text.startswith("地区："):
                    sibling = contactwnd.GetNextSiblingControl()
                    if sibling:
                        contact_info["area"] = sibling.Name.strip()

            for child in contactwnd.GetChildren():
                extract_info(child)
        extract_info(contactwnd)
        contactwnd.SendKeys('{Esc}')
        return contact_info
    
    @property
    def UiaAPI(self):
        if hasattr(self, '_uia') and self._uia.Exists(0):
            return self._uia
        self._uia = uia.WindowControl(ClassName=self._clsname, searchDepth=1)
        self.HWND = FindWindow(classname=self._clsname)
        wxwins = None
        while True:
            self.HWND = self._uia.NativeWindowHandle
            MainControl1 = [i for i in self._uia.GetChildren() if not i.ClassName][0]
            MainControl2 = MainControl1.GetFirstChildControl()
            self.NavigationBox, self.SessionBox, self.ChatBox  = MainControl2.GetChildren()
            
            # 初始化导航栏，以A开头 | self.NavigationBox  -->  A_xxx
            self.A_MyIcon = self.NavigationBox.ButtonControl()
            self.A_ChatIcon = self.NavigationBox.ButtonControl(Name=self._lang('聊天'))
            self.A_ContactsIcon = self.NavigationBox.ButtonControl(Name=self._lang('通讯录'))
            self.A_FavoritesIcon = self.NavigationBox.ButtonControl(Name=self._lang('收藏'))
            self.A_FilesIcon = self.NavigationBox.ButtonControl(Name=self._lang('聊天文件'))
            self.A_MomentsIcon = self.NavigationBox.ButtonControl(Name=self._lang('朋友圈'))
            self.A_BrowserIcon = self.NavigationBox.ButtonControl(Name=self._lang('搜一搜'))
            self.A_MiniProgram = self.NavigationBox.ButtonControl(Name=self._lang('小程序面板'))
            self.A_Phone = self.NavigationBox.ButtonControl(Name=self._lang('手机'))
            self.A_Settings = self.NavigationBox.ButtonControl(Name=self._lang('设置及其他'))
            
            # 初始化聊天列表，以B开头
            self.B_Search = self.SessionBox.EditControl(Name=self._lang('搜索'))
            
            # 初始化聊天栏，以C开头
            self.C_MsgList = self.ChatBox.ListControl(Name=self._lang('消息'))
            
            self.nickname = self.A_MyIcon.Name
            if self._nickname:
                if self.nickname == self._nickname:
                    break
                elif wxwins:
                    self._uia = wxwins.pop(0)
                elif wxwins is None:
                    wxwins = [i for i in uia.GetRootControl().GetChildren() if i.ClassName == self._clsname]
                else:
                    raise WeChatNotFoundError(f'未找到微信客户端：{self._nickname}')
            else:
                break

    # def _listen_msg(self):
    
    def IsOnline(self):
        """判断微信是否在线
        
        Returns:
            bool: 在线True，掉线False
        """
        return self.UiaAPI.Exists(0.1)
    
    def Moments(self):
        """进入朋友圈"""
        if self.A_MomentsIcon.Exists(0.1):
            self.A_MomentsIcon.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return WeChatMoments()

    def GetFriendDetails(self, n=None, tag='A', timeout=0xFFFFF):
        """获取所有好友详情信息
        
        Args:
            n (int, optional): 获取前n个好友详情信息, 默认为None，获取所有好友详情信息
            tag (str, optional): 从指定标签开始获取好友详情信息，如'A'，默认为'A'
            timeout (int, optional): 获取超时时间（秒），超过该时间则直接返回结果

        Returns:
            list: 所有好友详情信息
            
        注：1. 该方法运行时间较长，约0.5~1秒一个好友的速度，好友多的话可将n设置为一个较小的值，先测试一下
            2. 如果遇到企业微信的好友且为已离职状态，可能导致微信卡死，需重启（此为微信客户端BUG）
            3. 该方法未经过大量测试，可能存在未知问题，如有问题请微信群内反馈
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        t0 = time.time()
        self.SwitchToContact()
        ContactListItem = self.SessionBox.ButtonControl(Name='ContactListItem')
        while True:
            if ContactListItem.Exists(0):
                break
            self.SessionBox.WheelUp(wheelTimes=100)
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if tag.upper() not in letters:
            raise ValueError(f"tag参数错误，应为{letters}中的一个字符")
        useletters = letters[letters.index(tag.upper()):]
        for i in useletters:
            if self._goto_friends_tag(i):
                break
        details = []
        duplicate = 0
        while True:
            if time.time() - t0 > timeout:
                wxlog.debug('获取好友详情超时，返回结果')
                return details
            _detail = self._get_friend_details()
            if details and _detail == details[-1] and duplicate > 3:
                wxlog.debug('获取好友详情完成，返回结果')
                if duplicate > 3:
                    details = details[:-4]
                return details
            if details and _detail == details[-1]:
                duplicate += 1
            else:
                duplicate = 0
            details.append(_detail)
            self.SessionBox.SendKeys('{DOWN}')
            if n and len(details) >= n:
                wxlog.debug(f"获取前{n}个好友详情完成，返回结果")
                return details

    def GetSessionAmont(self, SessionItem):
        """获取聊天对象名和新消息条数
        
        Args:
            SessionItem (uiautomation.ListItemControl): 聊天对象控件
            
        Returns:
            sessionname (str): 聊天对象名
            amount (int): 新消息条数
        """
        matchobj = re.search('.*?条新消息', SessionItem.Name)
        amount = 0
        if matchobj:
            try:
                amount_text = [i for i in SessionItem.GetFirstChildControl().GetChildren() if type(i) == uia.TextControl][0].Name
                if amount_text == '...':
                    amount_text = '99'
                    # SessionItem.RightClick()
                    # self.UiaAPI.MenuControl(ClassName='CMenuWnd').MenuItemControl(Name='消息免打扰').Click()
                    # ele = SessionElement(SessionItem)
                    # content = ele.content
                    # SessionItem.RightClick()
                    # self.UiaAPI.MenuControl(ClassName='CMenuWnd').MenuItemControl(Name='开启新消息提醒').Click()
                    # amount_text = re.findall(r'\[(\d+)条\]', content)[0]
                amount = int(amount_text)
            except:
                pass
        sessionname = SessionItem.Name if SessionItem.ButtonControl().Name == 'SessionListItem' else SessionItem.ButtonControl().Name
        return sessionname, amount
    
    def CheckNewMessage(self):
        """是否有新消息"""
        
        return IsRedPixel(self.A_ChatIcon)
    
    def GetNextNewMessage(self, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False, timeout=10):
        """获取下一个新消息"""
        if WxParam.MOUSE_MOVE:
            self._show()
        msgids = self._get_now_msgid()
        current_chat = self.CurrentChat()

        if not self.usedmsgid:
            self.usedmsgid = msgids

        if not self._cache_chat_name:
            self._cache_chat_name = current_chat
        
        wxlog.debug(f"当前已使用消息id: {self.usedmsgid}")
        if self.C_MsgList.Exists(0):
            MsgItems = self.C_MsgList.GetChildren()
            msgids = [''.join([str(i) for i in i.GetRuntimeId()]) for i in MsgItems]
            wxlog.debug(f"当前消息列表id: {msgids}")
            new1 = [x for x in msgids if x not in set(self.usedmsgid)]
            wxlog.debug(f"新增消息id: {new1}")
            a_set = set(self.usedmsgid)
            last_in_a = max((x for x in msgids if x in a_set), key=self.usedmsgid.index, default=None)
            new2 = msgids[msgids.index(last_in_a) + 1 :] if last_in_a is not None else []
            wxlog.debug(f"后置位新增消息id: {new2}")
            new = [i for i in new1 if i in new2]
            if new and self._cache_chat_name == current_chat:
                wxlog.debug('获取当前窗口新消息')
                NewMsgItems = [i for i in MsgItems if ''.join([str(j) for j in i.GetRuntimeId()]) in new]
                msgs = self._getmsgs(NewMsgItems, savepic, savevideo, savefile, savevoice, parseurl)
                self.ChatBox.EditControl().Click()
                self.usedmsgid = msgids
                return {self.CurrentChat(): msgs}


        if self.CheckNewMessage():
            wxlog.debug('获取其他窗口新消息')
            t0 = time.time()
            while True:
                if time.time() - t0 > timeout:
                    wxlog.debug('获取新消息超时')
                    return {}
                self.A_ChatIcon.DoubleClick(simulateMove=False)
                sessiondict = self.GetSessionList(newmessage=True)
                if sessiondict:
                    break
                else:
                    return {}
            for session in sessiondict:
                self.ChatWith(session)
                self._cache_chat_name = self.CurrentChat()
                if len(self.C_MsgList.GetChildren()) < sessiondict[session]:
                    load_new_button = self.ChatBox.ButtonControl(RegexName='.*?条新消息')
                    wxlog.debug('检测到新消息按钮，点击新消息按钮')
                    if load_new_button.Exists(0):
                        load_new_button.Click()
                # while len(self.C_MsgList.GetChildren()) < sessiondict[session]:
                #     self.LoadMoreMessage()
                #     time.sleep(0.5)
                MsgItems = self.C_MsgList.GetChildren()
                NewMsgItems = MsgItems[-sessiondict[session]:]
                msgs = self._getmsgs(NewMsgItems, savepic, savevideo, savefile, savevoice, parseurl)
                # msgs = self.GetAllMessage(savepic=savepic, savefile=savefile, savevoice=savevoice, parseurl=parseurl)
                self.usedmsgid = [''.join([str(i) for i in i.GetRuntimeId()]) for i in MsgItems]
                newtag = False
                for i, msg in enumerate(msgs):
                    if msg.type == 'sys' and '新消息' in msg.content and '以下' in msg.content:
                        newtag = True
                        msgs = msgs[i+1:]
                        break
                if not newtag:
                    msgs = msgs[-sessiondict[session]:]
                return {session:msgs}
        else:
            wxlog.debug('没有新消息')
            return {}
    
    def GetAllNewMessage(self, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False, max_round=10):
        """获取所有新消息
        
        Args:
            max_round (int): 最大获取次数  * 这里是为了避免某几个窗口一直有新消息，导致无法停止
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        newmessages = {}
        for _ in range(max_round):
            newmsg = self.GetNextNewMessage(savepic, savevideo, savefile, savevoice, parseurl)
            if newmsg:
                for session in newmsg:
                    if session not in newmessages:
                        newmessages[session] = []
                    newmessages[session].extend(newmsg[session])
            else:
                break
        return newmessages
    
    def GetSessionList(self, reset=False, newmessage=False):
        """获取当前聊天列表中的所有聊天对象
        
        Args:
            reset (bool): 是否重置SessionItemList
            newmessage (bool): 是否只获取有新消息的聊天对象
            
        Returns:
            SessionList (dict): 聊天对象列表，键为聊天对象名，值为新消息条数
        """
        self.SessionItem = self.SessionBox.ListItemControl()
        if reset:
            self.SessionItemList = []
        SessionList = {}
        for i in range(100):
            if self.SessionItem.BoundingRectangle.width() != 0:
                try:
                    name, amount = self.GetSessionAmont(self.SessionItem)
                except:
                    break
                if name not in self.SessionItemList:
                    self.SessionItemList.append(name)
                if name not in SessionList:
                    SessionList[name] = amount
            self.SessionItem = self.SessionItem.GetNextSiblingControl()
            if not self.SessionItem:
                break
            
        if newmessage:
            return {i:SessionList[i] for i in SessionList if SessionList[i] > 0}
        return SessionList
    
    def GetSession(self):
        """获取当前聊天列表中的所有聊天对象

        Returns:
            SessionElement: 聊天对象列表

        Example:
            >>> wx = WeChat()
            >>> sessions = wx.GetSession()
            >>> for session in sessions:
            ...     print(f"聊天对象名称: {session.name}")
            ...     print(f"最后一条消息时间: {session.time}")
            ...     print(f"最后一条消息内容: {session.content}")
            ...     print(f"是否有新消息: {session.isnew}", end='\n\n')
        """
        sessions = self.SessionBox.ListControl()
        return [SessionElement(i) for i in sessions.GetChildren()]
    
    def ChatWith(self, who, timeout=3, exact=False):
        '''打开某个聊天框
        
        Args:
            who ( str ): 要打开的聊天框好友名;  * 最好完整匹配，不完全匹配只会选取搜索框第一个
            timeout ( num, optional ): 超时时间，默认3秒
            exact ( bool, optional ): 是否精确匹配好友名，默认False

        Returns:
            chatname ( str ): 匹配值第一个的完整名字
        '''
        if WxParam.MOUSE_MOVE:
            self._show()
        sessiondict = self.GetSessionList(True)
        if who in list(sessiondict.keys())[:-1]:
            self.SessionBox.ListItemControl(RegexName=re.escape(who)).Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return who
        self.UiaAPI.ShortcutSearch(click=False)
        self.B_Search.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        self.B_Search.Input(who)

        search_result = self.SessionBox.ListControl(RegexName='.*?IDS_FAV_SEARCH_RESULT.*?')
        
        t0 = time.time()
        while time.time() -t0 < timeout:
            results = []
            search_result_items = search_result.GetChildren()
            for search_result_item in search_result_items:
                if (
                    search_result_item.ControlTypeName == 'PaneControl'
                    and search_result_item.TextControl(Name='聊天记录').Exists(0)
                ) or search_result_item.Name == f'搜索 {who}':
                    break
                elif (
                    search_result_item.ControlTypeName == 'ListItemControl'
                    and search_result_item.TextControl(Name=f"微信号: <em>{who}</em>").Exists(0)
                ):
                    search_result_item.Click()
                    return search_result_item.Name
                elif (
                    search_result_item.ControlTypeName == 'ListItemControl'
                    and search_result_item.TextControl(Name=f"昵称: <em>{who}</em>").Exists(0)
                ):
                    search_result_item.Click()
                    return search_result_item.Name
                elif (
                    search_result_item.ControlTypeName == 'ListItemControl'
                    and who == search_result_item.Name
                ):
                    search_result_item.Click()
                    return who
                elif (
                    search_result_item.ControlTypeName == 'ListItemControl'
                    and who in search_result_item.Name
                ):
                    results.append(search_result_item)
            
        if exact:
            if search_result.Exists(0):
                search_result.SendKeys('{Esc}')
            return None
        if results:
            results[0].Click()
            return results[0].Name
        
        if search_result.Exists(0):
            search_result.SendKeys('{Esc}')
    
    # def ChatWith(self, who, timeout=2, exact=False):
    #     '''打开某个聊天框
        
    #     Args:
    #         who ( str ): 要打开的聊天框好友名;  * 最好完整匹配，不完全匹配只会选取搜索框第一个
    #         timeout ( num, optional ): 超时时间，默认2秒
    #         exact ( bool, optional ): 是否精确匹配，默认False
            
    #     Returns:
    #         chatname ( str ): 匹配值第一个的完整名字
    #     '''
    #     if WxParam.MOUSE_MOVE:
    #         self._show()
    #     sessiondict = self.GetSessionList(True)
    #     if who in list(sessiondict.keys())[:-1]:
    #         self.SessionBox.ListItemControl(RegexName=re.escape(who)).Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
    #         return who
    #     else:
    #         self.UiaAPI.ShortcutSearch(click=False)
    #         self.B_Search.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
    #         self.B_Search.Input(who)
    #         target_control1 = self.SessionBox.TextControl(Name=f"微信号: <em>{who}</em>")
    #         target_control2 = self.SessionBox.TextControl(Name=f"昵称: <em>{who}</em>")
    #         target_control3 = self.SessionBox.TextControl(Name=f"<em>{who}</em>")
    #         if target_control1.Exists(timeout):
    #             wxlog.debug('选择微信号匹配项')
    #             target_control1.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
    #             return who
    #         elif target_control2.Exists(timeout):
    #             wxlog.debug('选择昵称匹配项')
    #             target_control2.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
    #             return who
    #         elif target_control3.Exists(timeout) and target_control3.GetParentControl():
    #         else:
    #             if exact:
    #                 wxlog.debug(f'未找到精准匹配项: {who}')
    #                 self.UiaAPI.SendKeys('{Esc}')
    #                 return False
    #             search_result_control = self.SessionBox.GetChildren()[1].GetChildren()[1].GetFirstChildControl()
    #             if not search_result_control.PaneControl(searchDepth=1).TextControl(RegexName='联系人|群聊').Exists(0.1):
    #                 wxlog.debug(f'未找到搜索结果: {who}')
    #                 self.UiaAPI.SendKeys('{Esc}')
    #                 return False
    #             wxlog.debug('选择搜索结果第一个')
    #             target_control = search_result_control.Control(RegexName=f'.*{re.escape(who)}.*')
    #             chatname = target_control.Name
    #             target_control.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
    #             return chatname
    
    def AtAll(self, msg=None, who=None, exact=False):
        """@所有人
        
        Args:
            who (str, optional): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            msg (str, optional): 要发送的文本消息
            exact (bool, optional): 是否精确匹配，默认False
        """
        
        if who and FindWindow(name=who, classname='ChatWnd'):
            chat = ChatWnd(who, self, self.language)
            chat.AtAll(msg)
            return None
        
        if WxParam.MOUSE_MOVE:
            self._show()

        if who:
            try:
                editbox = self.ChatBox.EditControl(searchDepth=10)
                if who in self.CurrentChat() and who in editbox.Name:
                    pass
                else:
                    self.ChatWith(who, exact=exact)
                    editbox = self.ChatBox.EditControl(searchDepth=10)
            except:
                self.ChatWith(who, exact=exact)
                editbox = self.ChatBox.EditControl(searchDepth=10)
        else:
            editbox = self.ChatBox.EditControl(searchDepth=10)
        editbox.Input('@')
        atwnd = self.UiaAPI.PaneControl(ClassName='ChatContactMenu')
        if atwnd.Exists(maxSearchSeconds=0.1):
            atwnd.ListItemControl(Name='所有人').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            if msg:
                if not msg.startswith('\n'):
                    msg = '\n' + msg
                self.SendMsg(msg, who=who, clear=False)
            else:
                editbox.SendKeys(WxParam.SHORTCUT_SEND)

    def SendTypingText(self, msg, who=None, clear=True, exact=False):
        """发送文本消息（打字机模式），支持换行及@功能

        Args:
            msg (str): 要发送的文本消息
            who (str): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            clear (bool, optional): 是否清除原本的内容， 默认True
            exact (bool, optional): 是否精确匹配，默认False

        Example:
            >>> wx = WeChat()
            >>> wx.SendTypingText('你好', who='张三')

            换行及@功能：
            >>> wx.SendTypingText('各位下午好\n{@张三}负责xxx\n{@李四}负责xxxx', who='工作群')
        """
        
        if who and FindWindow(name=who, classname='ChatWnd'):
            chat = ChatWnd(who, self, self.language)
            chat.SendTypingText(msg)
            return None
        
        if not msg:
            return None
        if WxParam.MOUSE_MOVE:
            self._show()
        if who:
            try:
                editbox = self.ChatBox.EditControl(searchDepth=10)
                if who in self.CurrentChat() and who in editbox.Name:
                    pass
                else:
                    self.ChatWith(who, exact=exact)
                    editbox = self.ChatBox.EditControl(searchDepth=10)
            except:
                self.ChatWith(who, exact=exact)
                editbox = self.ChatBox.EditControl(searchDepth=10)

        else:
            editbox = self.ChatBox.EditControl(searchDepth=10)

        if clear:
            editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)

        def _at(name):
            editbox.Input(name)
            atwnd = self.UiaAPI.PaneControl(ClassName='ChatContactMenu')
            if atwnd.Exists(maxSearchSeconds=0.1):
                atele = atwnd.ListItemControl(Name=name[1:])
                if atele.Exists(0):
                    RollIntoView(atwnd, atele)
                    atele.Click()
                else:
                    wxlog.debug(f"识别到@消息，但是被@对象 {name} 未找到")

        atlist = re.findall(r'{(@.*?)}', msg)
        for name in atlist:
            text, msg = msg.split(f'{{{name}}}')
            editbox.Input(text)
            _at(name)
        editbox.Input(msg)
        self.ChatBox.EditControl().SendKeys(WxParam.SHORTCUT_SEND)
        # self.ChatBox.ButtonControl(Name="发送(S)", searchDepth=11).Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def SendMsg(self, msg, who=None, clear=True, at=None, exact=False):
        """发送文本消息

        Args:
            msg (str): 要发送的文本消息
            who (str): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            clear (bool, optional): 是否清除原本的内容，
            at (str|list, optional): 要@的人，可以是一个人或多个人，格式为str或list，例如："张三"或["张三", "李四"]
            exact (bool, optional): 搜索who好友时是否精确匹配，默认False
        """
        if who and FindWindow(name=who, classname='ChatWnd'):
            chat = ChatWnd(who, self, self.language)
            return chat.SendMsg(msg, at=at)
        if not msg and not at:
            return WxResponse.failure(f"消息为空，发送失败")
        if WxParam.MOUSE_MOVE:
            self._show()
        _msg = msg
        if who:
            if not self.ChatWith(who, exact=exact):
                return WxResponse.failure(f"发送对象<{who}>不存在，发送失败")
            
        editbox = self.ChatBox.EditControl(searchDepth=10)
        
        if not editbox.HasKeyboardFocus:
            editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        if clear:
            editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
        
        if at:
            if isinstance(at, str):
                at = [at]
            for i in at:
                editbox.Input('@'+i)
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
                                editbox.SendKeys('{BACK}')
                
                    # if msg and not msg.startswith('\n'):
                    #     msg = '\n' + msg
                else:
                    for _ in range(len(i)+1):
                        editbox.SendKeys('{BACK}')
            time.sleep(1)

        if msg:
            t0 = time.time()
            while True:
                if time.time() - t0 > 10:
                    raise TimeoutError(f'发送消息超时 --> {editbox.Name} - {_msg}')
                # if clear:
                #     editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
                SetClipboardText(msg)
                editbox.ShortcutPaste(click=False)
                if editbox.GetValuePattern().Value.replace('￼', ''):
                    break
                time.sleep(0.1)
        while editbox.GetValuePattern().Value:
            wxlog.debug(f"当前输入框内容：{editbox.GetValuePattern().Value}")
            if time.time() - t0 > 10:
                raise TimeoutError(f'发送消息超时 --> {who} - {_msg}')
            if not editbox.HasKeyboardFocus:
                editbox.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            editbox.SendKeys(WxParam.SHORTCUT_SEND)
            time.sleep(0.2)
        return WxResponse.success(f"发送成功")
    
    def SendEmotion(self, emotion_index, who=None, exact=False):
        """发送自定义表情
        
        Args:
            emotion_index (str): 表情索引，从0开始
            who (str): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            exact (bool, optional): 搜索who好友时是否精确匹配，默认False
        """
        
        if who and FindWindow(name=who, classname='ChatWnd'):
            chat = ChatWnd(who, self, self.language)
            chat.SendEmotion(emotion_index)
            return None
        if WxParam.MOUSE_MOVE:
            self._show()
        if who:
            self.ChatWith(who, exact=exact)
        
        self.ChatBox.ButtonControl(RegexName='表情.*?').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

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
                    if position == last_one.BoundingRectangle.top:
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
                    if position == fourth.BoundingRectangle.top:
                        return

        emotion = next_page(emotion_index, emotion_list, emotions, last_one, idx, amount)
        if emotion is not None:
            RollIntoView(emotion_list, emotion)
            emotion.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return WxResponse.success()
        else:
            wxlog.debug(f'未找到表情索引：{emotion_index}')
            EmotionWnd.SendKeys('{Esc}')
            return WxResponse.failure(f'未找到表情索引：{emotion_index}')
        
    def SendUrlCard(self, url, friends):
        """发送链接卡片

        Args:
            url (str): 链接地址
            friends (str|list): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
        """
        if not self.A_BrowserIcon.Exists(0):
            return WxResponse.failure('请先在手机端打开`搜一搜`功能')
        self.A_BrowserIcon.Click()
        browser = WxBrowser()
        if browser.send_card(url, friends):
            return WxResponse.success()
        else:
            return WxResponse.failure(f'发送链接卡片失败：{url}')
        
    def SendFiles(self, filepath, who=None, exact=False):
        """向当前聊天窗口发送文件
        
        Args:
            filepath (str|list): 要复制文件的绝对路径  
            who (str): 要发送给谁，如果为None，则发送到当前聊天页面。  *最好完整匹配，优先使用备注
            exact (bool, optional): 搜索who好友时是否精确匹配，默认False
            
        Returns:
            bool: 是否成功发送文件
        """
        if who and FindWindow(name=who, classname='ChatWnd'):
            chat = ChatWnd(who, self, self.language)
            chat.SendFiles(filepath)
            return None
        if WxParam.MOUSE_MOVE:
            self._show()
        filelist = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                wxlog.debug(f'未找到文件：{filepath}，无法成功发送')
                return WxResponse.failure(f'未找到文件：{filepath}，无法成功发送')
            else:
                filelist.append(os.path.realpath(filepath))
        elif isinstance(filepath, (list, tuple, set)):
            for i in filepath:
                if os.path.exists(i):
                    filelist.append(i)
                else:
                    wxlog.debug(f'未找到文件：{i}', stacklevel=2)
        else:
            wxlog.debug(f'filepath参数格式错误：{type(filepath)}，应为str、list、tuple、set格式')
            return WxResponse.failure(f'filepath参数格式错误：{type(filepath)}，应为str、list、tuple、set格式')
        
        if filelist:
            
            if who:
                try:
                    if who in self.CurrentChat() and who in self.ChatBox.EditControl(searchDepth=10).Name:
                        pass
                    else:
                        self.ChatWith(who, exact=exact)
                except:
                    self.ChatWith(who, exact=exact)
                editbox = self.ChatBox.EditControl()
            else:
                editbox = self.ChatBox.EditControl()
            editbox.ShortcutSelectAll(move=WxParam.MOUSE_MOVE)
            t0 = time.time()
            while True:
                if time.time() - t0 > 10:
                    raise TimeoutError(f'发送文件超时 --> {filelist}')
                SetClipboardFiles(filelist)
                time.sleep(0.2)
                editbox.ShortcutPaste()
                t1 = time.time()
                while time.time() - t1 < 5:
                    try:
                        edit_value = editbox.GetValuePattern().Value
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
                        edit_value = editbox.GetValuePattern().Value
                        break
                    except:
                        time.sleep(0.1)
                if not edit_value:
                    break
                editbox.SendKeys(WxParam.SHORTCUT_SEND)
                time.sleep(0.1)
            return WxResponse.success()
        else:
            wxlog.debug('所有文件都无法成功发送')
            return WxResponse.failure('所有文件都无法成功发送')
            
    def GetAllMessage(self, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False):
        '''获取当前窗口中加载的所有聊天记录
        
        Args:
            savepic (bool): 是否自动保存聊天图片
            
        Returns:
            list: 聊天记录信息
        '''
        if not self.C_MsgList.Exists(0.2):
            return []
        MsgItems = self.C_MsgList.GetChildren()
        msgs = self._getmsgs(MsgItems, savepic, savevideo=savevideo, savefile=savefile, savevoice=savevoice, parseurl=parseurl)
        return msgs
    
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
    
    def CurrentChat(self, details=False):
        '''获取当前聊天对象名
        
        Args:
            details (bool): 是否获取聊天对象的详细信息
            
        Returns:
            str|dict: 当前聊天对象名或详细信息'''
        if not self.C_MsgList.Exists(0):
            return None
        if details:
            chat_info = {}

            if self.ChatBox.PaneControl(ClassName='popupshadow').Exists(0):
                chat_name_control = self.ChatBox.GetProgenyControl(12)
            else:
                chat_name_control = self.ChatBox.GetProgenyControl(11)
            chat_name_control_list = chat_name_control.GetParentControl().GetChildren()
            chat_name_control_count = len(chat_name_control_list)
            if chat_name_control_count == 1:
                if self.ChatBox.ButtonControl(Name='公众号主页').Exists(0):
                    chat_info['chat_type'] = 'official'
                else:
                    chat_info['chat_type'] = 'friend'
                chat_info['chat_name'] = chat_name_control.Name
            elif chat_name_control_count >= 2:
                try:
                    chat_info['group_member_count'] = int(chat_name_control_list[-1].Name.replace('(', '').replace(')', ''))
                    chat_info['chat_type'] = 'group'
                    chat_info['chat_name'] = chat_name_control.Name.replace(chat_name_control_list[-1].Name, '')
                except:
                    chat_info['chat_type'] = 'friend'
                    chat_info['chat_name'] = chat_name_control.Name
                
                ori_chat_name_control = chat_name_control.GetParentControl().GetParentControl().TextControl(searchDepth=1)
                if ori_chat_name_control.Exists(0):
                    chat_info['chat_remark'] = chat_info['chat_name']
                    chat_info['chat_name'] = ori_chat_name_control.Name
            return chat_info
        else:
            uia.SetGlobalSearchTimeout(1)
            try:
                currentname = self.ChatBox.TextControl(searchDepth=15).Name
                return currentname
            except:
                return None
            finally:
                uia.SetGlobalSearchTimeout(10)

    def GetNewFriends(self):
        """获取新的好友申请列表
        
        Returns:
            list: 新的好友申请列表，元素为NewFriendsElement对象，可直接调用Accept方法

        Example:
            >>> wx = WeChat()
            >>> newfriends = wx.GetNewFriends()
            >>> tags = ['标签1', '标签2']
            >>> for friend in newfriends:
            ...     remark = f'备注{friend.name}'
            ...     friend.Accept(remark=remark, tags=tags)  # 接受好友请求，并设置备注和标签
        """
        
        self.SwitchToContact()
        ContactListItem = self.SessionBox.ButtonControl(Name='ContactListItem')
        while True:
            if ContactListItem.Exists(0):
                break
            self.SessionBox.WheelUp(wheelTimes=3)
        ContactListItem.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        if not self.ChatBox.ListControl(Name='新的朋友').Exists(0):
            self.ChatBox.ButtonControl(Name='').Click()
        NewFriendsList = [NewFriendsElement(i, self) for i in self.ChatBox.ListControl(Name='新的朋友').GetChildren()]
        AcceptableNewFriendsList = [i for i in NewFriendsList if i.acceptable]
        wxlog.debug(f'获取到 {len(AcceptableNewFriendsList)} 条新的好友申请')
        return AcceptableNewFriendsList
    
    def AddListenChat(self, who, savepic=False, savevideo=False, savefile=False, savevoice=False, parseurl=False, exact=False):
        """添加监听对象
        
        Args:
            who (str): 要监听的聊天对象名
            savepic (bool, optional): 是否自动保存聊天图片，只针对该聊天对象有效
            savefile (bool, optional): 是否自动保存聊天文件，只针对该聊天对象有效
            savevoice (bool, optional): 是否自动保存聊天语音，只针对该聊天对象有效
            parseurl (bool, optional): 是否解析聊天消息中的链接，只针对该聊天对象有效
            exact (bool, optional): 搜索who好友时是否精确匹配，默认False
        """
        if isinstance(who, list):
            for i in who:
                self.AddListenChat(i, savepic, savefile, savevoice, parseurl)
            return None
        exists = uia.WindowControl(searchDepth=1, ClassName='ChatWnd', Name=who).Exists(maxSearchSeconds=0.1)
        if not exists:
            who = self.ChatWith(who, exact=exact)
            self.SessionBox.ListItemControl(RegexName=f"^{re.escape(who)}(?:已置顶)?$").DoubleClick(simulateMove=False)
        self.listen[who] = ChatWnd(who, self, self.language)
        self.listen[who].savepic = savepic
        self.listen[who].savevideo = savevideo
        self.listen[who].savefile = savefile
        self.listen[who].savevoice = savevoice
        self.listen[who].parseurl = parseurl

    def AddSubWindowListen(self, force=False):
        sub_wins = [i for i in uia.GetRootControl().GetChildren() if i.ClassName == 'ChatWnd']
        for win in sub_wins:
            who = win.Name
            if who in self.listen and not force:
                continue
            chat = ChatWnd(win.Name, self, self.language)
            self.listen[who] = chat
            self.listen[who].savepic = False
            self.listen[who].savevideo = False
            self.listen[who].savefile = False
            self.listen[who].savevoice = False
            self.listen[who].parseurl = False

    def GetListenMessage(self, who=None):
        """获取监听对象的新消息
        
        Args:
            who (str, optional): 要获取消息的聊天对象名，如果为None，则获取所有监听对象的消息

        Returns:
            str|dict: 如果
        """
        if who and who in self.listen:
            chat = self.listen.get(who)
            if chat in self._listener_messages:
                msgs = self._listener_messages.pop(chat)
                return {chat: msgs}
            else:
                return {}
        else:
            msgs = self._listener_messages.copy()
            self._listener_messages.clear()
        return msgs

    # def GetListenMessage(self, who=None):
    #     """获取监听对象的新消息
        
    #     Args:
    #         who (str, optional): 要获取消息的聊天对象名，如果为None，则获取所有监听对象的消息

    #     Returns:
    #         str|dict: 如果
    #     """
    #     wxlog.debug('GetListenMessage获取监听对象的新消息')
    #     temp_listen = self.listen.copy()
    #     if who and who in temp_listen:
    #         chat = temp_listen.get(who, None)
    #         try:
    #             if chat is None or not chat.UiaAPI.Exists(0.1):
    #                 try:
    #                     wxlog.debug(f"`{who}`窗口不存在，删除监听")
    #                     del self.listen[who]
    #                 except:
    #                     pass
    #                 return {}
    #         except:
    #             exc = traceback.format_exc()
    #             wxlog.debug(f"获取监听对象的新消息异常：{exc}")
    #             return {}
    #         msg = chat.GetNewMessage(
    #             savepic=chat.savepic, 
    #             savefile=chat.savefile, 
    #             savevoice=chat.savevoice, 
    #             parseurl=chat.parseurl
    #         )
    #         wxlog.debug(f"`{who}`返回监听消息{msg}")
    #         return msg
    #     msgs = {}
    #     for who in temp_listen:
    #         chat = temp_listen.get(who, None)
    #         try:
    #             if chat is None or not chat.UiaAPI.Exists(0.1):
    #                 try:
    #                     wxlog.debug(f"`{who}`窗口不存在，删除监听")
    #                     del self.listen[who]
    #                 except:
    #                     pass
    #                 continue
    #         except:
    #             continue
    #         msg = chat.GetNewMessage(
    #             savepic=chat.savepic, 
    #             savefile=chat.savefile, 
    #             savevoice=chat.savevoice,
    #             parseurl=chat.parseurl
    #         )
    #         if msg:
    #             msgs[chat] = msg
    #     wxlog.debug(f"`{who}`返回监听消息{msg}")
    #     return msgs

    def SwitchToContact(self):
        """切换到通讯录页面"""
        if WxParam.MOUSE_MOVE:
            self._show()
        self.A_ContactsIcon.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def SwitchToChat(self):
        """切换到聊天页面"""
        if WxParam.MOUSE_MOVE:
            self._show()
        self.A_ChatIcon.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

    def AddGroupMembers(self, group, members):
        """添加群成员

        Args:
            group (str): 群名或备注名
            members (list): 成员列表，列表元素可以是好友微信号、昵称、备注名
        """
        if FindWindow(name=group, classname='ChatWnd'):
            chat = ChatWnd(group, self, self.language)
            return chat.AddGroupMembers(members)
        if WxParam.MOUSE_MOVE:
            self._show()
        self.ChatWith(group)
        self.ChatBox.GetProgenyControl(10, control_type='ButtonControl').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
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

    def RemoveGroupMembers(self, group, members):
        """移除群成员
        
        Args:
            members (list): 成员列表

        Returns:
            list: 实际移除结果
        """
        if FindWindow(name=group, classname='ChatWnd'):
            chat = ChatWnd(group, self, self.language)
            return chat.RemoveGroupMembers(members)
        
        remove_result = []
        if isinstance(members, str):
            members = [members]
        if WxParam.MOUSE_MOVE:
            self._show()
        self.ChatWith(group)
        self.ChatBox.GetProgenyControl(10, control_type='ButtonControl').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        RoomDetailWndControl = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        RoomDetailWndControl.ButtonControl(Name='移出').GetParentControl().GetChildren()[0].Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        removeWnd = DeleteMemberWnd(self)
        for member in members:
            remove_target = removeWnd.Remove(member)
            if remove_target:
                remove_result.append(remove_target)
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
        return remove_result

    def MuteNotifications(self, mute=True, minimize_group=False):
        """调整消息免打扰
        
        Args:
            mute (bool, optional): 是否对**当前聊天窗口**开启消息免打扰，默认True
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        ele = self.ChatBox.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
        ele.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        roominfoWnd = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        checkbox_mute = roominfoWnd.CheckBoxControl(Name='消息免打扰')
        RollIntoView(roominfoWnd, checkbox_mute)
        mute_status = checkbox_mute.GetTogglePattern().ToggleState
        if mute != mute_status:
            checkbox_mute.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        if mute and minimize_group:
            checkbox_mini = roominfoWnd.CheckBoxControl(RegexName='折叠.*?')
            if checkbox_mini.Exists(0.5):
                RollIntoView(roominfoWnd, checkbox_mini)
                minimize_status = checkbox_mini.GetTogglePattern().ToggleState
                if minimize_group != minimize_status:
                    checkbox_mini.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)

        roominfoWnd.SendKeys('{Esc}')

    def GetGroupMembers(self, add_friend_mode=False):
        """获取当前聊天群成员

        Returns:
            list: 当前聊天群成员列表
        """
        if WxParam.MOUSE_MOVE:
            self._show()
        ele = self.ChatBox.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
        try:
            uia.SetGlobalSearchTimeout(1)
            ele.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        except:
            return 
        finally:
            uia.SetGlobalSearchTimeout(10)
        roominfoWnd = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        more = roominfoWnd.ButtonControl(Name='查看更多', searchDepth=8)
        try:
            uia.SetGlobalSearchTimeout(1)
            more.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        except:
            pass
        finally:
            uia.SetGlobalSearchTimeout(10)
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


    def GetAllFriends(self, keywords=None, speed=5):
        """获取所有好友列表
        注：
            1. 该方法运行时间取决于好友数量，约每秒6~8个好友的速度
            2. 该方法未经过大量测试，可能存在未知问题，如有问题请微信群内反馈
        
        Args:
            keywords (str, optional): 搜索关键词，只返回包含关键词的好友列表
            speed (int, optional): 滚动速度，数值越大滚动越快，但是太快可能导致遗漏，建议速度1-5之间
            
        Returns:
            list: 所有好友列表
        """
        
        self.SwitchToContact()
        self.SessionBox.ListControl(Name="联系人").ButtonControl(Name="通讯录管理").Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        contactwnd = ContactWnd()
        if keywords:
            contactwnd.Search(keywords)
        friends = contactwnd.GetAllFriends(speed)
        contactwnd.Close()
        self.SwitchToChat()
        return friends
    
    def GetAllRecentGroups(self, speed: int = 1, wait=0.05):
        """获取群列表
        
        Args:
            speed (int, optional): 滚动速度，数值越大滚动越快，但是太快可能导致遗漏，建议速度1-3之间
            wait (float, optional): 滚动等待时间，建议和speed一起调整，直至适合你电脑配置和微信群数量达到平衡，不遗漏数据
            
        Returns:
            list: 群列表
        """
        
        self.SwitchToContact()
        self.SessionBox.ListControl(Name="联系人").ButtonControl(Name="通讯录管理").Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        contactwnd = ContactWnd()
        groups = contactwnd.GetAllRecentGroups(speed, wait)
        contactwnd.Close()
        self.SwitchToChat()
        return groups
    
    def GetAllListenChat(self):
        """获取所有监听对象"""
        return self.listen
    
    def RemoveListenChat(self, who):
        """移除监听对象
        
        Args:
            who (str): 要移除的监听对象名
        """
        if who in self.listen:
            self.listen[who].Close()
            del self.listen[who]
        else:
            Warnings.lightred(f'未找到监听对象：{who}', stacklevel=2)

    def AddNewFriend(self, keywords, addmsg=None, remark=None, tags=None, permission='朋友圈'):
        """添加新的好友

        Args:
            keywords (str): 搜索关键词，微信号、手机号、QQ号
            addmsg (str, optional): 添加好友的消息
            remark (str, optional): 备注名
            tags (list, optional): 标签列表

        Example:
            >>> wx = WeChat()
            >>> keywords = '13800000000'      # 微信号、手机号、QQ号
            >>> addmsg = '你好，我是xxxx'      # 添加好友的消息
            >>> remark = '备注名字'            # 备注名
            >>> tags = ['朋友', '同事']        # 标签列表
            >>> wx.AddNewFriend(keywords, addmsg=addmsg, remark=remark, tags=tags)
        """
        returns = {
            0: '未知原因添加失败',
            1: '发送请求成功',
            2: '已经是好友',
            3: '已被对方拉黑',
            4: '找不到相关账号或内容',
        }
        def return_func(code, msg=None):
            msg = msg if msg else returns.get(code, '未知原因添加失败')
            wxlog.debug(msg)
            return code, returns.get(code, msg)
        
        self.SwitchToContact()
        self.SessionBox.ButtonControl(Name='添加朋友').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        edit = self.SessionBox.EditControl()
        edit.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        edit.Input(keywords)
        self._show()
        self.SessionBox.TextControl(Name=f'搜索：{keywords}').Click(simulateMove=False, move=WxParam.MOUSE_MOVE, show_window=(not WxParam.MOUSE_MOVE))

        ContactProfileWnd = uia.PaneControl(ClassName='ContactProfileWnd', searchDepth=1)
        if ContactProfileWnd.Exists(maxSearchSeconds=2):
            # 点击添加到通讯录
            add_btn = ContactProfileWnd.ButtonControl(Name='添加到通讯录')
            if add_btn.Exists(0):
                add_btn.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            else:
                self.SwitchToChat()
                return return_func(2)
        elif self.SessionBox.ListItemControl(Name="找不到相关账号或内容").Exists(0):
            self.SwitchToChat()
            return return_func(4)
        elif self.SessionBox.ListItemControl(Name="无法找到该用户，请检查你填写的账号是否正确。").Exists(0):
            self.SwitchToChat()
            return return_func(4)
        elif not self.SessionBox.ListControl().GetChildren()[0].Name.startswith('搜索：'):
            msg = self.SessionBox.ListControl().GetChildren()[0].TextControl().Name
            self.SwitchToChat()
            return return_func(0, msg)
        else:
            self.SwitchToChat()
            return return_func(0)

        NewFriendsWnd = self.UiaAPI.WindowControl(ClassName='WeUIDialog', searchDepth=4)
        AlertWnd = self.UiaAPI.WindowControl(ClassName='AlertDialog')
        if NewFriendsWnd.Exists(maxSearchSeconds=2):
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
            self.SwitchToChat()
            return return_func(1)
        elif AlertWnd.Exists(0):
            msg = AlertWnd.GetProgenyControl(4, 1, control_type='TextControl').Name
            AlertWnd.ButtonControl(Name='确定').Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            self.SwitchToChat()
            return return_func(3, msg)
        else:
            self.SwitchToChat()
            return return_func(0)
        
    def ManageFriend(self, remark=None, tags=None):
        """修改备注名或标签
        
        Args:
            remark (str, optional): 备注名
            tags (list, optional): 标签列表

        Returns:
            bool: 是否成功修改备注名或标签
        """
        if all([not remark, not tags]):
            return WxResponse.failure("请至少输入一个参数")
        chat_info = self.CurrentChat(details=True)
        if chat_info['chat_type'] != 'friend':
            wxlog.debug('当前聊天对象不是好友')
            return WxResponse.failure("当前聊天对象不是好友")
        msgs = self.GetAllMessage()
        for msg in msgs[::-1]:
            if msg.type == 'friend':
                return msg.modify(remark=remark, tags=tags)
        ele = self.ChatBox.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
        ele.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        roominfoWnd = self.UiaAPI.Control(ClassName='SessionChatRoomDetailWnd', searchDepth=1)
        members = [i for i in roominfoWnd.ListControl(Name='聊天成员').GetChildren()]
        members[0].Click(move=True, simulateMove=False, return_pos=True)
        profile = ProfileWnd(self)
        result = profile.ModifyRemarkOrTags(remark, tags)
        profile.Close()
        if roominfoWnd.Exists(0):
            roominfoWnd.SendKeys('{ESC}')
        return result
    
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
        chat_info = self.CurrentChat(details=True)
        if chat_info['chat_type'] != 'group':
            wxlog.debug('当前聊天对象不是群聊')
            return WxResponse.failure('当前聊天对象不是群聊')
        ele = self.ChatBox.PaneControl(searchDepth=7, foundIndex=6).ButtonControl(Name='聊天信息')
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
    
    def GetOfficialMenu(self):
        """获取服务号菜单

        Returns:
            list: 服务号菜单列表
        """
        wxlog.debug("获取服务号菜单")
        chatinfo = self.CurrentChat(details=True)
        input_button = self.ChatBox.ButtonControl(Name='输入', searchDepth=8)
        if input_button.Exists(1):
            eles = input_button.GetParentControl().GetChildren()[0].GetChildren()
            menu_buttons = [OfficialMenu(i, chatinfo['chat_name'], self) for i in eles]
            wxlog.debug(f"[{chatinfo['chat_name']}]找到菜单: {menu_buttons}")
            return menu_buttons
        wxlog.debug(f"[{chatinfo['chat_name']}]未找到菜单")
    
    def ChooseOfficialMenu(self, menu, option):
        """选择服务号菜单

        Args:
            menu (str | int): 菜单名，可以是菜单名或菜单索引
            option (str | int): 菜单选项名，可以是菜单选项名或菜单选项索引
        """
        menu_buttons = self.GetOfficialMenu()
        wxlog.debug(f"选择服务号菜单: {menu} -> {option}")
        if not menu_buttons:
            raise Exception('请先切换到服务号页面或未找到服务号菜单')

        if isinstance(menu, int):
            menu_buttons[menu].go(option)

        elif isinstance(menu, str):
            for i in menu_buttons:
                if i.name == menu:
                    i.go(option)
                    break
            else:
                raise Exception('菜单名不存在')
    
    def CallGroupMsg(self, group, members):
        """发起群语音通话
        
        Args:
            group (str): 群名或备注名
            members (list): 成员列表，列表元素可以是好友微信号、昵称、备注名
        """
        
        if WxParam.MOUSE_MOVE:
            self._show()
        if group and FindWindow(name=group, classname='ChatWnd'):
            chat = ChatWnd(group, self, self.language)
            return chat.CallGroupMsg(members)
        
        wxlog.debug(f"发起群语音通话：{group} - {members}")
        self.ChatWith(group)
        wxlog.debug(f"发起群语音通话：{members}")
        chat_info = self.CurrentChat(details=True)
        if chat_info['chat_type'] != 'group':
            wxlog.debug('当前聊天对象不是群聊')
            return False
        self.ChatBox.ButtonControl(Name='语音聊天').Click()
        addwnd = AddTalkMemberWnd(self)
        if not addwnd.UiaAPI.Exists(5):
            return False
        for member in members:
            addwnd.Add(member)
        addwnd.Submit()
    

class WeChatFiles:
    def __init__(self, language='cn') -> None:
        self.language = language
        self.api = uia.WindowControl(ClassName='FileListMgrWnd', searchDepth=1)
        MainControl3 = [i for i in self.api.GetChildren() if not i.ClassName][0]
        self.FileBox ,self.Search ,self.SessionBox = MainControl3.GetChildren()

        self.allfiles = self.SessionBox.ButtonControl(Name=self._lang('全部'))
        self.recentfiles = self.SessionBox.ButtonControl(Name=self._lang('最近使用'))
        self.whofiles = self.SessionBox.ButtonControl(Name=self._lang('发送者'))
        self.chatfiles = self.SessionBox.ButtonControl(Name=self._lang('聊天'))
        self.typefiles = self.SessionBox.ButtonControl(Name=self._lang('类型'))

    def GetSessionName(self, SessionItem):
        """获取聊天对象的名字

        Args:
            SessionItem (uiautomation.ListItemControl): 聊天对象控件

        Returns:
            sessionname (str): 聊天对象名
        """
        return SessionItem.Name

    def GetSessionList(self, reset=False):
        """获取当前聊天列表中的所有聊天对象的名字

        Args:
            reset (bool): 是否重置SessionItemList

        Returns:
            session_names (list): 对象名称列表
        """
        self.SessionItem = self.SessionBox.ListControl(Name='',searchDepth=3).GetChildren()
        if reset:
            self.SessionItemList = []
        session_names = []
        for i in range(len(self.SessionItem)):
            session_names.append(self.GetSessionName(self.SessionItem[i]))

        return session_names

    def __repr__(self) -> str:
        return f"<wxauto WeChat Image at {hex(id(self))}>"

    def _lang(self, text):
        return FILE_LANGUAGE[text][self.language]

    def _show(self):
        pass
        # HWND = FindWindow(classname='ImagePreviewWnd')
        # win32gui.ShowWindow(HWND, 1)
        # self.api.SwitchToThisWindow()

    def ChatWithFile(self, who):
        '''打开某个聊天会话

        Args:
            who ( str ): 要打开的聊天框好友名。

        Returns:
            chatname ( str ): 打开的聊天框的名字。
        '''
        
        self.chatfiles.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        sessiondict = self.GetSessionList(True)

        if who in sessiondict:
            # 直接点击已存在的聊天框
            self.SessionBox.ListItemControl(Name=who).Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
            return who
        else:
            # 如果聊天框不在列表中，则抛出异常
            raise TargetNotFoundError(f'未查询到目标：{who}')

    def DownloadFiles(self, who, amount, deadline=None, size=None):
        '''开始下载文件

        Args:
            who ( str )：聊天名称
            amount ( num )：下载的文件数量限制。
            deadline ( str )：截止日期限制。
            size ( str )：文件大小限制。

        Returns:
            result ( bool )：下载是否成功

        '''
        
        itemlist = self.GetSessionList()
        if who in itemlist:
            self.item = self.SessionBox.ListItemControl(Name=who)
            self.item.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
        else:
            wxlog.debug(f'未查询到目标：{who}')
        itemfileslist = []

        item = self.SessionBox.ListControl(Name='', searchDepth=7).GetParentControl()
        item = item.GetNextSiblingControl()
        item = item.ListControl(Name='', searchDepth=5).GetChildren()
        del item[0]

        for i in range(amount):
            try:

                itemfileslist.append(item[i].Name)
                self.itemfiles = item[i]
                self.itemfiles.Click(move=WxParam.MOUSE_MOVE, simulateMove=False, return_pos=False)
                time.sleep(0.5)
            except:
                pass

    def Close(self):
        
        self.api.SendKeys('{Esc}')
