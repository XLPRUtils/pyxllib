# cython: language_level=3
import time
import re
import random
import ctypes
import ctypes.wintypes
import win32gui
import win32api
import win32con
from typing_extensions import Literal

GlobalKeyNames = [
    'CONTROL', 
    'ALT', 
    'SHIFT', 
    'WIN', 
    'CTRL', 
    'LWIN', 
    'RWIN'
]

class Keys:
    """Key codes from Win32."""
    VK_LBUTTON = 0x01                       #Left mouse button
    VK_RBUTTON = 0x02                       #Right mouse button
    VK_CANCEL = 0x03                        #Control-break processing
    VK_MBUTTON = 0x04                       #Middle mouse button (three-button mouse)
    VK_XBUTTON1 = 0x05                      #X1 mouse button
    VK_XBUTTON2 = 0x06                      #X2 mouse button
    VK_BACK = 0x08                          #BACKSPACE key
    VK_TAB = 0x09                           #TAB key
    VK_CLEAR = 0x0C                         #CLEAR key
    VK_RETURN = 0x0D                        #ENTER key
    VK_ENTER = 0x0D
    VK_SHIFT = 0x10                         #SHIFT key
    VK_CONTROL = 0x11                       #CTRL key
    VK_MENU = 0x12                          #ALT key
    VK_PAUSE = 0x13                         #PAUSE key
    VK_CAPITAL = 0x14                       #CAPS LOCK key
    VK_KANA = 0x15                          #IME Kana mode
    VK_HANGUEL = 0x15                       #IME Hanguel mode (maintained for compatibility; use VK_HANGUL)
    VK_HANGUL = 0x15                        #IME Hangul mode
    VK_JUNJA = 0x17                         #IME Junja mode
    VK_FINAL = 0x18                         #IME final mode
    VK_HANJA = 0x19                         #IME Hanja mode
    VK_KANJI = 0x19                         #IME Kanji mode
    VK_ESCAPE = 0x1B                        #ESC key
    VK_CONVERT = 0x1C                       #IME convert
    VK_NONCONVERT = 0x1D                    #IME nonconvert
    VK_ACCEPT = 0x1E                        #IME accept
    VK_MODECHANGE = 0x1F                    #IME mode change request
    VK_SPACE = 0x20                         #SPACEBAR
    VK_PRIOR = 0x21                         #PAGE UP key
    VK_PAGEUP = 0x21
    VK_NEXT = 0x22                          #PAGE DOWN key
    VK_PAGEDOWN = 0x22
    VK_END = 0x23                           #END key
    VK_HOME = 0x24                          #HOME key
    VK_LEFT = 0x25                          #LEFT ARROW key
    VK_UP = 0x26                            #UP ARROW key
    VK_RIGHT = 0x27                         #RIGHT ARROW key
    VK_DOWN = 0x28                          #DOWN ARROW key
    VK_SELECT = 0x29                        #SELECT key
    VK_PRINT = 0x2A                         #PRINT key
    VK_EXECUTE = 0x2B                       #EXECUTE key
    VK_SNAPSHOT = 0x2C                      #PRINT SCREEN key
    VK_INSERT = 0x2D                        #INS key
    VK_DELETE = 0x2E                        #DEL key
    VK_HELP = 0x2F                          #HELP key
    VK_0 = 0x30                             #0 key
    VK_1 = 0x31                             #1 key
    VK_2 = 0x32                             #2 key
    VK_3 = 0x33                             #3 key
    VK_4 = 0x34                             #4 key
    VK_5 = 0x35                             #5 key
    VK_6 = 0x36                             #6 key
    VK_7 = 0x37                             #7 key
    VK_8 = 0x38                             #8 key
    VK_9 = 0x39                             #9 key
    VK_A = 0x41                             #A key
    VK_B = 0x42                             #B key
    VK_C = 0x43                             #C key
    VK_D = 0x44                             #D key
    VK_E = 0x45                             #E key
    VK_F = 0x46                             #F key
    VK_G = 0x47                             #G key
    VK_H = 0x48                             #H key
    VK_I = 0x49                             #I key
    VK_J = 0x4A                             #J key
    VK_K = 0x4B                             #K key
    VK_L = 0x4C                             #L key
    VK_M = 0x4D                             #M key
    VK_N = 0x4E                             #N key
    VK_O = 0x4F                             #O key
    VK_P = 0x50                             #P key
    VK_Q = 0x51                             #Q key
    VK_R = 0x52                             #R key
    VK_S = 0x53                             #S key
    VK_T = 0x54                             #T key
    VK_U = 0x55                             #U key
    VK_V = 0x56                             #V key
    VK_W = 0x57                             #W key
    VK_X = 0x58                             #X key
    VK_Y = 0x59                             #Y key
    VK_Z = 0x5A                             #Z key
    VK_LWIN = 0x5B                          #Left Windows key (Natural keyboard)
    VK_RWIN = 0x5C                          #Right Windows key (Natural keyboard)
    VK_APPS = 0x5D                          #Applications key (Natural keyboard)
    VK_SLEEP = 0x5F                         #Computer Sleep key
    VK_NUMPAD0 = 0x60                       #Numeric keypad 0 key
    VK_NUMPAD1 = 0x61                       #Numeric keypad 1 key
    VK_NUMPAD2 = 0x62                       #Numeric keypad 2 key
    VK_NUMPAD3 = 0x63                       #Numeric keypad 3 key
    VK_NUMPAD4 = 0x64                       #Numeric keypad 4 key
    VK_NUMPAD5 = 0x65                       #Numeric keypad 5 key
    VK_NUMPAD6 = 0x66                       #Numeric keypad 6 key
    VK_NUMPAD7 = 0x67                       #Numeric keypad 7 key
    VK_NUMPAD8 = 0x68                       #Numeric keypad 8 key
    VK_NUMPAD9 = 0x69                       #Numeric keypad 9 key
    VK_MULTIPLY = 0x6A                      #Multiply key
    VK_ADD = 0x6B                           #Add key
    VK_SEPARATOR = 0x6C                     #Separator key
    VK_SUBTRACT = 0x6D                      #Subtract key
    VK_DECIMAL = 0x6E                       #Decimal key
    VK_DIVIDE = 0x6F                        #Divide key
    VK_F1 = 0x70                            #F1 key
    VK_F2 = 0x71                            #F2 key
    VK_F3 = 0x72                            #F3 key
    VK_F4 = 0x73                            #F4 key
    VK_F5 = 0x74                            #F5 key
    VK_F6 = 0x75                            #F6 key
    VK_F7 = 0x76                            #F7 key
    VK_F8 = 0x77                            #F8 key
    VK_F9 = 0x78                            #F9 key
    VK_F10 = 0x79                           #F10 key
    VK_F11 = 0x7A                           #F11 key
    VK_F12 = 0x7B                           #F12 key
    VK_F13 = 0x7C                           #F13 key
    VK_F14 = 0x7D                           #F14 key
    VK_F15 = 0x7E                           #F15 key
    VK_F16 = 0x7F                           #F16 key
    VK_F17 = 0x80                           #F17 key
    VK_F18 = 0x81                           #F18 key
    VK_F19 = 0x82                           #F19 key
    VK_F20 = 0x83                           #F20 key
    VK_F21 = 0x84                           #F21 key
    VK_F22 = 0x85                           #F22 key
    VK_F23 = 0x86                           #F23 key
    VK_F24 = 0x87                           #F24 key
    VK_NUMLOCK = 0x90                       #NUM LOCK key
    VK_SCROLL = 0x91                        #SCROLL LOCK key
    VK_LSHIFT = 0xA0                        #Left SHIFT key
    VK_RSHIFT = 0xA1                        #Right SHIFT key
    VK_LCONTROL = 0xA2                      #Left CONTROL key
    VK_RCONTROL = 0xA3                      #Right CONTROL key
    VK_LMENU = 0xA4                         #Left MENU key
    VK_RMENU = 0xA5                         #Right MENU key
    VK_BROWSER_BACK = 0xA6                  #Browser Back key
    VK_BROWSER_FORWARD = 0xA7               #Browser Forward key
    VK_BROWSER_REFRESH = 0xA8               #Browser Refresh key
    VK_BROWSER_STOP = 0xA9                  #Browser Stop key
    VK_BROWSER_SEARCH = 0xAA                #Browser Search key
    VK_BROWSER_FAVORITES = 0xAB             #Browser Favorites key
    VK_BROWSER_HOME = 0xAC                  #Browser Start and Home key
    VK_VOLUME_MUTE = 0xAD                   #Volume Mute key
    VK_VOLUME_DOWN = 0xAE                   #Volume Down key
    VK_VOLUME_UP = 0xAF                     #Volume Up key
    VK_MEDIA_NEXT_TRACK = 0xB0              #Next Track key
    VK_MEDIA_PREV_TRACK = 0xB1              #Previous Track key
    VK_MEDIA_STOP = 0xB2                    #Stop Media key
    VK_MEDIA_PLAY_PAUSE = 0xB3              #Play/Pause Media key
    VK_LAUNCH_MAIL = 0xB4                   #Start Mail key
    VK_LAUNCH_MEDIA_SELECT = 0xB5           #Select Media key
    VK_LAUNCH_APP1 = 0xB6                   #Start Application 1 key
    VK_LAUNCH_APP2 = 0xB7                   #Start Application 2 key
    VK_OEM_1 = 0xBA                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the ';:' key
    VK_OEM_PLUS = 0xBB                      #For any country/region, the '+' key
    VK_OEM_COMMA = 0xBC                     #For any country/region, the ',' key
    VK_OEM_MINUS = 0xBD                     #For any country/region, the '-' key
    VK_OEM_PERIOD = 0xBE                    #For any country/region, the '.' key
    VK_OEM_2 = 0xBF                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '/?' key
    VK_OEM_3 = 0xC0                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '`~' key
    VK_OEM_4 = 0xDB                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '[{' key
    VK_OEM_5 = 0xDC                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '\|' key
    VK_OEM_6 = 0xDD                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the ']}' key
    VK_OEM_7 = 0xDE                         #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the 'single-quote/double-quote' key
    VK_OEM_8 = 0xDF                         #Used for miscellaneous characters; it can vary by keyboard.
    VK_OEM_102 = 0xE2                       #Either the angle bracket key or the backslash key on the RT 102-key keyboard
    VK_PROCESSKEY = 0xE5                    #IME PROCESS key
    VK_PACKET = 0xE7                        #Used to pass Unicode characters as if they were keystrokes. The VK_PACKET key is the low word of a 32-bit Virtual Key value used for non-keyboard input methods. For more information, see Remark in KEYBDINPUT, SendInput, WM_KEYDOWN, and WM_KeyUp
    VK_ATTN = 0xF6                          #Attn key
    VK_CRSEL = 0xF7                         #CrSel key
    VK_EXSEL = 0xF8                         #ExSel key
    VK_EREOF = 0xF9                         #Erase EOF key
    VK_PLAY = 0xFA                          #Play key
    VK_ZOOM = 0xFB                          #Zoom key
    VK_NONAME = 0xFC                        #Reserved
    VK_PA1 = 0xFD                           #PA1 key
    VK_OEM_CLEAR = 0xFE                     #Clear key

CharacterCodes = {
    '0': Keys.VK_0,                             #0 key
    '1': Keys.VK_1,                             #1 key
    '2': Keys.VK_2,                             #2 key
    '3': Keys.VK_3,                             #3 key
    '4': Keys.VK_4,                             #4 key
    '5': Keys.VK_5,                             #5 key
    '6': Keys.VK_6,                             #6 key
    '7': Keys.VK_7,                             #7 key
    '8': Keys.VK_8,                             #8 key
    '9': Keys.VK_9,                             #9 key
    'a': Keys.VK_A,                             #A key
    'A': Keys.VK_A,                             #A key
    'b': Keys.VK_B,                             #B key
    'B': Keys.VK_B,                             #B key
    'c': Keys.VK_C,                             #C key
    'C': Keys.VK_C,                             #C key
    'd': Keys.VK_D,                             #D key
    'D': Keys.VK_D,                             #D key
    'e': Keys.VK_E,                             #E key
    'E': Keys.VK_E,                             #E key
    'f': Keys.VK_F,                             #F key
    'F': Keys.VK_F,                             #F key
    'g': Keys.VK_G,                             #G key
    'G': Keys.VK_G,                             #G key
    'h': Keys.VK_H,                             #H key
    'H': Keys.VK_H,                             #H key
    'i': Keys.VK_I,                             #I key
    'I': Keys.VK_I,                             #I key
    'j': Keys.VK_J,                             #J key
    'J': Keys.VK_J,                             #J key
    'k': Keys.VK_K,                             #K key
    'K': Keys.VK_K,                             #K key
    'l': Keys.VK_L,                             #L key
    'L': Keys.VK_L,                             #L key
    'm': Keys.VK_M,                             #M key
    'M': Keys.VK_M,                             #M key
    'n': Keys.VK_N,                             #N key
    'N': Keys.VK_N,                             #N key
    'o': Keys.VK_O,                             #O key
    'O': Keys.VK_O,                             #O key
    'p': Keys.VK_P,                             #P key
    'P': Keys.VK_P,                             #P key
    'q': Keys.VK_Q,                             #Q key
    'Q': Keys.VK_Q,                             #Q key
    'r': Keys.VK_R,                             #R key
    'R': Keys.VK_R,                             #R key
    's': Keys.VK_S,                             #S key
    'S': Keys.VK_S,                             #S key
    't': Keys.VK_T,                             #T key
    'T': Keys.VK_T,                             #T key
    'u': Keys.VK_U,                             #U key
    'U': Keys.VK_U,                             #U key
    'v': Keys.VK_V,                             #V key
    'V': Keys.VK_V,                             #V key
    'w': Keys.VK_W,                             #W key
    'W': Keys.VK_W,                             #W key
    'x': Keys.VK_X,                             #X key
    'X': Keys.VK_X,                             #X key
    'y': Keys.VK_Y,                             #Y key
    'Y': Keys.VK_Y,                             #Y key
    'z': Keys.VK_Z,                             #Z key
    'Z': Keys.VK_Z,                             #Z key
    ' ': Keys.VK_SPACE,                         #Space key
    '`': Keys.VK_OEM_3,                         #` key
    #'~' : Keys.VK_OEM_3,                         #~ key
    '-': Keys.VK_OEM_MINUS,                     #- key
    #'_' : Keys.VK_OEM_MINUS,                     #_ key
    '=': Keys.VK_OEM_PLUS,                      #= key
    #'+' : Keys.VK_OEM_PLUS,                      #+ key
    '[': Keys.VK_OEM_4,                         #[ key
    #'{' : Keys.VK_OEM_4,                         #{ key
    ']': Keys.VK_OEM_6,                         #] key
    #'}' : Keys.VK_OEM_6,                         #} key
    '\\': Keys.VK_OEM_5,                        #\ key
    #'|' : Keys.VK_OEM_5,                         #| key
    ';': Keys.VK_OEM_1,                         #; key
    #':' : Keys.VK_OEM_1,                         #: key
    '\'': Keys.VK_OEM_7,                        #' key
    #'"' : Keys.VK_OEM_7,                         #" key
    ',': Keys.VK_OEM_COMMA,                     #, key
    #'<' : Keys.VK_OEM_COMMA,                     #< key
    '.': Keys.VK_OEM_PERIOD,                    #. key
    #'>' : Keys.VK_OEM_PERIOD,                    #> key
    '/': Keys.VK_OEM_2,                         #/ key
    #'?' : Keys.VK_OEM_2,                         #? key
}


SpecialKeyNames = {
    'LBUTTON': Keys.VK_LBUTTON,                        #Left mouse button
    'RBUTTON': Keys.VK_RBUTTON,                        #Right mouse button
    'CANCEL': Keys.VK_CANCEL,                          #Control-break processing
    'MBUTTON': Keys.VK_MBUTTON,                        #Middle mouse button (three-button mouse)
    'XBUTTON1': Keys.VK_XBUTTON1,                      #X1 mouse button
    'XBUTTON2': Keys.VK_XBUTTON2,                      #X2 mouse button
    'BACK': Keys.VK_BACK,                              #BACKSPACE key
    'TAB': Keys.VK_TAB,                                #TAB key
    'CLEAR': Keys.VK_CLEAR,                            #CLEAR key
    'RETURN': Keys.VK_RETURN,                          #ENTER key
    'ENTER': Keys.VK_RETURN,                           #ENTER key
    'SHIFT': Keys.VK_SHIFT,                            #SHIFT key
    'CTRL': Keys.VK_CONTROL,                           #CTRL key
    'CONTROL': Keys.VK_CONTROL,                        #CTRL key
    'ALT': Keys.VK_MENU,                               #ALT key
    'PAUSE': Keys.VK_PAUSE,                            #PAUSE key
    'CAPITAL': Keys.VK_CAPITAL,                        #CAPS LOCK key
    'KANA': Keys.VK_KANA,                              #IME Kana mode
    'HANGUEL': Keys.VK_HANGUEL,                        #IME Hanguel mode (maintained for compatibility; use VK_HANGUL)
    'HANGUL': Keys.VK_HANGUL,                          #IME Hangul mode
    'JUNJA': Keys.VK_JUNJA,                            #IME Junja mode
    'FINAL': Keys.VK_FINAL,                            #IME final mode
    'HANJA': Keys.VK_HANJA,                            #IME Hanja mode
    'KANJI': Keys.VK_KANJI,                            #IME Kanji mode
    'ESC': Keys.VK_ESCAPE,                             #ESC key
    'ESCAPE': Keys.VK_ESCAPE,                          #ESC key
    'CONVERT': Keys.VK_CONVERT,                        #IME convert
    'NONCONVERT': Keys.VK_NONCONVERT,                  #IME nonconvert
    'ACCEPT': Keys.VK_ACCEPT,                          #IME accept
    'MODECHANGE': Keys.VK_MODECHANGE,                  #IME mode change request
    'SPACE': Keys.VK_SPACE,                            #SPACEBAR
    'PRIOR': Keys.VK_PRIOR,                            #PAGE UP key
    'PAGEUP': Keys.VK_PRIOR,                           #PAGE UP key
    'NEXT': Keys.VK_NEXT,                              #PAGE DOWN key
    'PAGEDOWN': Keys.VK_NEXT,                           #PAGE DOWN key
    'END': Keys.VK_END,                                #END key
    'HOME': Keys.VK_HOME,                              #HOME key
    'LEFT': Keys.VK_LEFT,                              #LEFT ARROW key
    'UP': Keys.VK_UP,                                  #UP ARROW key
    'RIGHT': Keys.VK_RIGHT,                            #RIGHT ARROW key
    'DOWN': Keys.VK_DOWN,                              #DOWN ARROW key
    'SELECT': Keys.VK_SELECT,                          #SELECT key
    'PRINT': Keys.VK_PRINT,                            #PRINT key
    'EXECUTE': Keys.VK_EXECUTE,                        #EXECUTE key
    'SNAPSHOT': Keys.VK_SNAPSHOT,                      #PRINT SCREEN key
    'PRINTSCREEN': Keys.VK_SNAPSHOT,                    #PRINT SCREEN key
    'INSERT': Keys.VK_INSERT,                          #INS key
    'INS': Keys.VK_INSERT,                             #INS key
    'DELETE': Keys.VK_DELETE,                          #DEL key
    'DEL': Keys.VK_DELETE,                             #DEL key
    'HELP': Keys.VK_HELP,                              #HELP key
    'WIN': Keys.VK_LWIN,                               #Left Windows key (Natural keyboard)
    'LWIN': Keys.VK_LWIN,                              #Left Windows key (Natural keyboard)
    'RWIN': Keys.VK_RWIN,                              #Right Windows key (Natural keyboard)
    'APPS': Keys.VK_APPS,                              #Applications key (Natural keyboard)
    'SLEEP': Keys.VK_SLEEP,                            #Computer Sleep key
    'NUMPAD0': Keys.VK_NUMPAD0,                        #Numeric keypad 0 key
    'NUMPAD1': Keys.VK_NUMPAD1,                        #Numeric keypad 1 key
    'NUMPAD2': Keys.VK_NUMPAD2,                        #Numeric keypad 2 key
    'NUMPAD3': Keys.VK_NUMPAD3,                        #Numeric keypad 3 key
    'NUMPAD4': Keys.VK_NUMPAD4,                        #Numeric keypad 4 key
    'NUMPAD5': Keys.VK_NUMPAD5,                        #Numeric keypad 5 key
    'NUMPAD6': Keys.VK_NUMPAD6,                        #Numeric keypad 6 key
    'NUMPAD7': Keys.VK_NUMPAD7,                        #Numeric keypad 7 key
    'NUMPAD8': Keys.VK_NUMPAD8,                        #Numeric keypad 8 key
    'NUMPAD9': Keys.VK_NUMPAD9,                        #Numeric keypad 9 key
    'MULTIPLY': Keys.VK_MULTIPLY,                      #Multiply key
    'ADD': Keys.VK_ADD,                                #Add key
    'SEPARATOR': Keys.VK_SEPARATOR,                    #Separator key
    'SUBTRACT': Keys.VK_SUBTRACT,                      #Subtract key
    'DECIMAL': Keys.VK_DECIMAL,                        #Decimal key
    'DIVIDE': Keys.VK_DIVIDE,                          #Divide key
    'F1': Keys.VK_F1,                                  #F1 key
    'F2': Keys.VK_F2,                                  #F2 key
    'F3': Keys.VK_F3,                                  #F3 key
    'F4': Keys.VK_F4,                                  #F4 key
    'F5': Keys.VK_F5,                                  #F5 key
    'F6': Keys.VK_F6,                                  #F6 key
    'F7': Keys.VK_F7,                                  #F7 key
    'F8': Keys.VK_F8,                                  #F8 key
    'F9': Keys.VK_F9,                                  #F9 key
    'F10': Keys.VK_F10,                                #F10 key
    'F11': Keys.VK_F11,                                #F11 key
    'F12': Keys.VK_F12,                                #F12 key
    'F13': Keys.VK_F13,                                #F13 key
    'F14': Keys.VK_F14,                                #F14 key
    'F15': Keys.VK_F15,                                #F15 key
    'F16': Keys.VK_F16,                                #F16 key
    'F17': Keys.VK_F17,                                #F17 key
    'F18': Keys.VK_F18,                                #F18 key
    'F19': Keys.VK_F19,                                #F19 key
    'F20': Keys.VK_F20,                                #F20 key
    'F21': Keys.VK_F21,                                #F21 key
    'F22': Keys.VK_F22,                                #F22 key
    'F23': Keys.VK_F23,                                #F23 key
    'F24': Keys.VK_F24,                                #F24 key
    'NUMLOCK': Keys.VK_NUMLOCK,                        #NUM LOCK key
    'SCROLL': Keys.VK_SCROLL,                          #SCROLL LOCK key
    'LSHIFT': Keys.VK_LSHIFT,                          #Left SHIFT key
    'RSHIFT': Keys.VK_RSHIFT,                          #Right SHIFT key
    'LCONTROL': Keys.VK_LCONTROL,                      #Left CONTROL key
    'LCTRL': Keys.VK_LCONTROL,                         #Left CONTROL key
    'RCONTROL': Keys.VK_RCONTROL,                      #Right CONTROL key
    'RCTRL': Keys.VK_RCONTROL,                         #Right CONTROL key
    'LALT': Keys.VK_LMENU,                             #Left MENU key
    'RALT': Keys.VK_RMENU,                             #Right MENU key
    'BROWSER_BACK': Keys.VK_BROWSER_BACK,              #Browser Back key
    'BROWSER_FORWARD': Keys.VK_BROWSER_FORWARD,        #Browser Forward key
    'BROWSER_REFRESH': Keys.VK_BROWSER_REFRESH,        #Browser Refresh key
    'BROWSER_STOP': Keys.VK_BROWSER_STOP,              #Browser Stop key
    'BROWSER_SEARCH': Keys.VK_BROWSER_SEARCH,          #Browser Search key
    'BROWSER_FAVORITES': Keys.VK_BROWSER_FAVORITES,    #Browser Favorites key
    'BROWSER_HOME': Keys.VK_BROWSER_HOME,              #Browser Start and Home key
    'VOLUME_MUTE': Keys.VK_VOLUME_MUTE,                #Volume Mute key
    'VOLUME_DOWN': Keys.VK_VOLUME_DOWN,                #Volume Down key
    'VOLUME_UP': Keys.VK_VOLUME_UP,                    #Volume Up key
    'MEDIA_NEXT_TRACK': Keys.VK_MEDIA_NEXT_TRACK,      #Next Track key
    'MEDIA_PREV_TRACK': Keys.VK_MEDIA_PREV_TRACK,      #Previous Track key
    'MEDIA_STOP': Keys.VK_MEDIA_STOP,                  #Stop Media key
    'MEDIA_PLAY_PAUSE': Keys.VK_MEDIA_PLAY_PAUSE,      #Play/Pause Media key
    'LAUNCH_MAIL': Keys.VK_LAUNCH_MAIL,                #Start Mail key
    'LAUNCH_MEDIA_SELECT': Keys.VK_LAUNCH_MEDIA_SELECT,#Select Media key
    'LAUNCH_APP1': Keys.VK_LAUNCH_APP1,                #Start Application 1 key
    'LAUNCH_APP2': Keys.VK_LAUNCH_APP2,                #Start Application 2 key
    'OEM_1': Keys.VK_OEM_1,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the ';:' key
    'OEM_PLUS': Keys.VK_OEM_PLUS,                      #For any country/region, the '+' key
    'OEM_COMMA': Keys.VK_OEM_COMMA,                    #For any country/region, the ',' key
    'OEM_MINUS': Keys.VK_OEM_MINUS,                    #For any country/region, the '-' key
    'OEM_PERIOD': Keys.VK_OEM_PERIOD,                  #For any country/region, the '.' key
    'OEM_2': Keys.VK_OEM_2,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '/?' key
    'OEM_3': Keys.VK_OEM_3,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '`~' key
    'OEM_4': Keys.VK_OEM_4,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '[{' key
    'OEM_5': Keys.VK_OEM_5,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the '\|' key
    'OEM_6': Keys.VK_OEM_6,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the ']}' key
    'OEM_7': Keys.VK_OEM_7,                            #Used for miscellaneous characters; it can vary by keyboard.For the US standard keyboard, the 'single-quote/double-quote' key
    'OEM_8': Keys.VK_OEM_8,                            #Used for miscellaneous characters; it can vary by keyboard.
    'OEM_102': Keys.VK_OEM_102,                        #Either the angle bracket key or the backslash key on the RT 102-key keyboard
    'PROCESSKEY': Keys.VK_PROCESSKEY,                  #IME PROCESS key
    'PACKET': Keys.VK_PACKET,                          #Used to pass Unicode characters as if they were keystrokes. The VK_PACKET key is the low word of a 32-bit Virtual Key value used for non-keyboard input methods. For more information, see Remark in KEYBDINPUT, SendInput, WM_KEYDOWN, and WM_KeyUp
    'ATTN': Keys.VK_ATTN,                              #Attn key
    'CRSEL': Keys.VK_CRSEL,                            #CrSel key
    'EXSEL': Keys.VK_EXSEL,                            #ExSel key
    'EREOF': Keys.VK_EREOF,                            #Erase EOF key
    'PLAY': Keys.VK_PLAY,                              #Play key
    'ZOOM': Keys.VK_ZOOM,                              #Zoom key
    'NONAME': Keys.VK_NONAME,                          #Reserved
    'PA1': Keys.VK_PA1,                                #PA1 key
    'OEM_CLEAR': Keys.VK_OEM_CLEAR,                    #Clear key
}

def SetClipboardText(text: str) -> bool:
    """
    Return bool, True if succeed otherwise False.
    """
    if ctypes.windll.user32.OpenClipboard(0):
        ctypes.windll.user32.EmptyClipboard()
        textByteLen = (len(text) + 1) * 2
        hClipboardData = ctypes.windll.kernel32.GlobalAlloc(0, textByteLen)  # GMEM_FIXED=0
        hDestText = ctypes.windll.kernel32.GlobalLock(ctypes.c_void_p(hClipboardData))
        ctypes.cdll.msvcrt.wcsncpy(ctypes.c_wchar_p(hDestText), ctypes.c_wchar_p(text), ctypes.c_size_t(textByteLen // 2))
        ctypes.windll.kernel32.GlobalUnlock(ctypes.c_void_p(hClipboardData))
        # system owns hClipboardData after calling SetClipboardData,
        # application can not write to or free the data once ownership has been transferred to the system
        ctypes.windll.user32.SetClipboardData(ctypes.c_uint(13), ctypes.c_void_p(hClipboardData))  # CF_TEXT=1, CF_UNICODETEXT=13
        ctypes.windll.user32.CloseClipboard()
        return True
    return False

# Add by Cluic
class Win32:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        # print(time.time())
    #     self._check_hwnd()

    # def _check_hwnd(self):
    #     if not win32gui.IsWindow(self.hwnd):
    #         raise Exception("Window not found")

    def activate(self):
        """
        Activate the window.
        """
        win32gui.ShowWindow(self.hwnd, 1)
        win32gui.SetWindowPos(self.hwnd, -1, 0, 0, 0, 0, 3)
        win32gui.SetWindowPos(self.hwnd, -2, 0, 0, 0, 0, 3)
    
    def click(self, x: int, y: int, button: Literal["left", "right", "mid"] = "left", activate: bool = False):
        """
        Click at the specified client coordinates.
        
        Args:
            x: The x-coordinate.
            y: The y-coordinate.
            button: The button to click. Default is "left".
        """
        if activate:
            self.activate()
        x1, y1 = win32gui.ScreenToClient(self.hwnd, (x, y))
        if button.lower() == "left":
            # win32gui.SendMessage(self.hwnd, win32con.WM_NCHITTEST, 0, win32api.MAKELONG(x, y))
            # win32gui.SendMessage(self.hwnd, win32con.WM_SETCURSOR, self.hwnd, win32api.MAKELONG(win32con.HTCLIENT, 0))
            win32api.SendMessage(self.hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, win32api.MAKELONG(x1, y1))
            win32api.SendMessage(self.hwnd, win32con.WM_LBUTTONUP, 0, win32api.MAKELONG(x1, y1))
            # win32gui.SendMessage(self.hwnd, win32con.WM_ERASEBKGND, 0, 0)
            # win32gui.SendMessage(self.hwnd, win32con.WM_PAINT, 0, 0)
        elif button.lower() == "right":
            win32api.SendMessage(self.hwnd, win32con.WM_RBUTTONDOWN, win32con.MK_RBUTTON, win32api.MAKELONG(x1, y1))
            win32api.SendMessage(self.hwnd, win32con.WM_RBUTTONUP, 0, win32api.MAKELONG(x1, y1))
        elif button.lower() == "mid":
            win32api.SendMessage(self.hwnd, win32con.WM_MBUTTONDOWN, win32con.MK_MBUTTON, win32api.MAKELONG(x1, y1))
            win32api.SendMessage(self.hwnd, win32con.WM_MBUTTONUP, 0, win32api.MAKELONG(x1, y1))
        else:
            raise ValueError("Invalid button")
        
    def hover(self, x: int, y: int):
        """
        Move the mouse to the specified client coordinates.
        """
        x, y = win32gui.ScreenToClient(self.hwnd, (x, y))
        # win32api.SendMessage(self.hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(x, y))
        ctypes.windll.user32.PostMessageW(self.hwnd, win32con.WM_MOUSEMOVE, 0, win32api.MAKELONG(x, y))

    def double_click(self, x: int, y: int, activate: bool = False):
        """
        Double click at the specified client coordinates.
        """
        if activate:
            self.activate()
        x, y = win32gui.ScreenToClient(self.hwnd, (x, y))
        win32api.SendMessage(self.hwnd, win32con.WM_LBUTTONDBLCLK, win32con.MK_LBUTTON, win32api.MAKELONG(x, y))

    def hover_by_bbox(self, bbox, pos: Literal["center", "top", "bottom", "right", "left"] = "center", xbias: int = 0, ybias: int = 0):
        """
        Move the mouse to the specified position relative to the bounding box.

        Args:
            bbox: The bounding box (left, top, right, bottom).
            pos: The position to move to. Default is "center".
            xbias: The x-coordinate bias. Default is 0.
            ybias: The y-coordinate bias. Default is 0.
        """
        # Support for uiautomation package Rect object
        if type(bbox).__name__ == "Rect":
            bbox = (bbox.left, bbox.top, bbox.right, bbox.bottom)

        left, top, right, bottom = bbox
        if pos.lower() == "center":
            x = (left + right) // 2
            y = (top + bottom) // 2
        elif pos.lower() == "top":
            x = (left + right) // 2
            y = top
        elif pos.lower() == "bottom":
            x = (left + right) // 2
            y = bottom
        elif pos.lower() == "right":
            x = right
            y = (top + bottom) // 2
        elif pos.lower() == "left":
            x = left
            y = (top + bottom) // 2
        else:
            raise ValueError("Invalid position")
        if not xbias:
            xbias = 0
        if not ybias:
            ybias = 0
        x += xbias
        y += ybias
        self.hover(x, y)

    def click_by_bbox(
            self, 
            bbox, 
            pos: Literal["center", "top", "bottom", "right", "left"] = "center",
            button: Literal["left", "right", "mid"] = "left",
            double_click: bool = False,
            xbias: int = 0,
            ybias: int = 0,
            activate: bool = False
        ):
        """
        Click at the specified position relative to the bounding box.

        Args:
            bbox: The bounding box (left, top, right, bottom).
            pos: The position to click. Default is "center".
            button: The button to click. Default is "left". When `double_click` is True, this argument is ignored.
            double_click: Whether to double click. Default is False.
            xbias: The x-coordinate bias. Default is 0.
            ybias: The y-coordinate bias. Default is 0.
        """
        # Support for uiautomation package Rect object
        if type(bbox).__name__ == "Rect":
            bbox = (bbox.left, bbox.top, bbox.right, bbox.bottom)
        
        # Calculate the click position
        left, top, right, bottom = bbox
        if pos.lower() == "center":
            x = (left + right) // 2
            y = (top + bottom) // 2
        elif pos.lower() == "top":
            x = (left + right) // 2
            y = top
        elif pos.lower() == "bottom":
            x = (left + right) // 2
            y = bottom
        elif pos.lower() == "right":
            x = right
            y = (top + bottom) // 2
        elif pos.lower() == "left":
            x = left
            y = (top + bottom) // 2
        else:
            raise ValueError("Invalid position")
        if not xbias:
            xbias = 0
        if not ybias:
            ybias = 0
        x += xbias
        y += ybias
        if double_click:
            self.double_click(x, y, activate)
        else:
            self.click(x, y, button, activate)

    def scroll_wheel(self, bbox, delta=120):
        """
        Scroll the mouse wheel at the specified client coordinates.

        Args:
            client_x: The x-coordinate.
            client_y: The y-coordinate.
            delta: The amount to scroll. Default is 120. Positive values scroll up, negative values scroll down.
        """
        if type(bbox).__name__ == "Rect":
            bbox = (bbox.left, bbox.top, bbox.right, bbox.bottom)
        x = (bbox[0] + bbox[2]) // 2
        y = (bbox[1] + bbox[3]) // 2
        wParam = win32api.MAKELONG(0, delta)
        lParam = win32api.MAKELONG(x, y)
        win32api.SendMessage(self.hwnd, win32con.WM_MOUSEWHEEL, wParam, lParam)

    # def input(self, content: str, delay: float = -1):
    #     """
    #     Input text.
        
    #     Args:
    #         content: The text to input.
    #         delay: The delay between each character. Default is random between 0.01 and 0.05
    #     """
    #     for char in content:
    #         if char == '\n':
    #             self.send_keys_shortcut('{SHIFT}{ENTER}')
    #         win32api.SendMessage(self.hwnd, win32con.WM_CHAR, ord(char), 0)
    #         if delay < 0:
    #             time.sleep(random.uniform(0.01, 0.05))
    #         else:
    #             time.sleep(delay)

    def input(self, content: str, delay: float = -1):
        """
        Input text.
        
        Args:
            content: The text to input.
            delay: The delay between each character. Default is random between 0.01 and 0.05
        """
        for char in content:
            try:
                if char == '\n':
                #     win32api.PostMessage(self.hwnd, win32con.WM_CHAR, 0x0A, 0)
                #     win32api.PostMessage(self.hwnd, win32con.WM_CHAR, 0x0D, 0)
                    # self.send_keys_shortcut('{SHIFT}{ENTER}')
                    SetClipboardText('\n')
                    self.shortcut_paste()
                    continue
                
                # 判断是否是基本多语言平面字符
                if ord(char) <= 0xFFFF:
                    # 普通字符直接发送
                    win32api.SendMessage(self.hwnd, win32con.WM_CHAR, ord(char), 0)
                else:
                    # 处理超出基本多语言平面的字符 (例如 emoji)
                    utf16 = char.encode("utf-16-le")  # 转为 UTF-16
                    high_surrogate = int.from_bytes(utf16[:2], "little")  # 高代理
                    low_surrogate = int.from_bytes(utf16[2:], "little")  # 低代理
                    
                    # 分别发送高代理和低代理
                    win32api.SendMessage(self.hwnd, win32con.WM_CHAR, high_surrogate, 0)
                    win32api.SendMessage(self.hwnd, win32con.WM_CHAR, low_surrogate, 0)
            
            except Exception as e:
                print(f"Error processing character '{char}': {e}")

            if delay <= 0:
                time.sleep(random.uniform(0.01, 0.05))
            else:
                time.sleep(delay)

    def send_keys_shortcut(self, keys: str):
        """
        Send a key combination.

        Args:
            keys: The key combination. str

        Example:
            send_keys_shortcut('{LCTRL}{V}')  # Paste
        """
        keys = re.findall(r'\{(.*?)\}', keys)
        hold_keys = []
        for key in keys:
            _key = SpecialKeyNames.get(key.upper(), CharacterCodes.get(key, None))
            if key.upper() in GlobalKeyNames:
                win32api.keybd_event(_key, 0, 0, 0)
            elif _key is not None:
                win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, _key, 0)
                time.sleep(0.05)
            else:
                continue
            hold_keys.append(_key)

        for _key in hold_keys[::-1]:
            win32api.keybd_event(_key, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)

    def shortcut_paste(self):
        """
        Paste the clipboard content.
        """
        win32api.keybd_event(win32con.VK_CONTROL, 0,0,0)
        time.sleep(0.1)
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYDOWN, 86, 0)
        win32api.keybd_event(86, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)

    def shortcut_search(self):
        """
        Open the search dialog.
        """
        win32api.keybd_event(win32con.VK_CONTROL, 0,0,0)
        time.sleep(0.1)
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYDOWN, 70, 0)
        win32api.keybd_event(70, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)

    def shortcut_select_all(self):
        """
        Select all text.
        """
        win32api.keybd_event(win32con.VK_CONTROL, 0,0,0)
        time.sleep(0.1)
        win32gui.SendMessage(self.hwnd, win32con.WM_KEYDOWN, 65, 0)
        win32api.keybd_event(65, 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)
