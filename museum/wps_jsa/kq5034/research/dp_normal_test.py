from DrissionPage import Chromium
from DrissionPage.common import Keys
import time
import re

class KDocsAirScriptBot:
    """
    金山文档 AirScript 自动化助手
    支持多文档操作、新建脚本、代码写入及运行。
    """
    def __init__(self, url=None, port=None):
        self.browser = Chromium(port if port else 9222)
        self.tab = self.browser.latest_tab
        self.url = url
        
    def _click_visible_text(self, text, timeout=5, exact_match=False):
        """[内部方法] 查找并点击可见的文本元素"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            eles = self.tab.eles(f'text:{text}')
            for ele in eles:
                if ele.states.is_displayed:
                    if exact_match and len(ele.text.strip()) > len(text) + 2:
                        continue
                    try:
                        print(f"点击文本元素: {ele.text.strip() or ele.tag}")
                        ele.click(by_js=True)
                        return True
                    except:
                        continue
            time.sleep(0.5)
        return False

    def ensure_page_loaded(self, url=None):
        """确保目标文档已在当前标签页打开"""
        target_url = url if url else self.url
        if not target_url:
            print("错误: 未指定 URL")
            return

        if target_url not in self.tab.url:
            print(f"正在打开文档: {target_url}")
            self.tab.get(target_url)
            self.tab.wait.load_start()
            time.sleep(3)
        else:
            print("文档已在当前标签页打开")

    def _is_element_visible(self, locator):
        ele = self.tab.ele(locator, timeout=2)
        return ele and ele.states.is_displayed

    def open_editor(self):
        """打开 AirScript 编辑器（支持幂等操作）"""
        if self.tab.ele('css:.monaco-editor', timeout=2):
            print("检测到编辑器已打开。")
            return

        print("编辑器未打开，开始执行打开流程...")
        if not self._is_element_visible('text:高级开发'):
            if not self._click_visible_text('效率', exact_match=True):
                 self._click_visible_text('效率')
            time.sleep(1)
            
        if not self._is_element_visible('text:AirScript脚本编辑器'):
            self._click_visible_text('高级开发')
            time.sleep(1)
        
        if self._click_visible_text('AirScript脚本编辑器'):
            print("已点击编辑器入口，等待加载...")
            time.sleep(5)
        else:
            print("错误: 无法找到 'AirScript脚本编辑器' 入口")

    def create_new_script(self, version="2.0"):
        """
        新建脚本逻辑：
        1. 定位 '+' 按钮 (在 '文档共享脚本' 标题附近)
        2. 点击 '+'
        3. 模糊匹配 'AirScript 2.0' (忽略 'Beta' 等后缀)
        """
        print(f"尝试新建 AirScript {version} 脚本...")
        
        # 1. 寻找 '+' 按钮
        plus_btn = None
        
        # 方法A: 精确文本 '+'
        plus_candidates = self.tab.eles('text:+')
        for btn in plus_candidates:
             if btn.states.is_displayed:
                 plus_btn = btn
                 break
                 
        # 方法B: 如果没找到文本 '+'，找图标 class
        if not plus_btn:
            icon_candidates = self.tab.eles('css:[class*="add"]') + self.tab.eles('css:[class*="plus"]')
            for icon in icon_candidates:
                if icon.states.is_displayed and icon.rect.size[0] < 50:
                     if icon.rect.location[0] < 400:
                         plus_btn = icon
                         break

        if plus_btn:
            print(f"找到疑似 '+' 按钮 (Tag={plus_btn.tag}, Text={plus_btn.text}), 点击...")
            plus_btn.click(by_js=True)
            time.sleep(1.5)
            
            # 3. 选择版本
            target_text = f"AirScript {version}"
            
            # 策略1：精确匹配 AirScript 2.0
            menu_items = self.tab.eles(f'text:{target_text}')
            target_item = None
            for item in menu_items:
                if item.states.is_displayed:
                    target_item = item
                    break
            
            if target_item:
                print(f"找到菜单项 '{target_item.text}'，点击...")
                target_item.click(by_js=True)
                time.sleep(3)
                return True
            
            # 策略2：模糊匹配，寻找包含 "2.0" 的可见元素，且其父元素包含 "AirScript"
            print(f"精确匹配失败，尝试模糊搜索 '2.0'...")
            eles_20 = self.tab.eles('text:2.0')
            for e in eles_20:
                if e.states.is_displayed:
                    # 检查它自己或父级文本是否包含 AirScript
                    full_text = e.text
                    parent = e.parent()
                    if parent:
                        full_text += " " + parent.text
                    
                    if "AirScript" in full_text:
                        print(f"通过关联文本找到菜单项: {e.text} (Context: {full_text[:20]}...)")
                        # 往往点击最外层的容器比较稳
                        if parent:
                            parent.click(by_js=True)
                        else:
                            e.click(by_js=True)
                        time.sleep(3)
                        return True
                        
            # 策略3：更激进的模糊匹配，只找可见的 '2.0'，且位于最近弹出的菜单中
            # 假设最近的弹窗位于 DOM 树的末尾
            print("尝试直接点击可见的 '2.0' ...")
            for e in eles_20:
                if e.states.is_displayed:
                     # 简单的启发式：通常版本号文本很短
                     if len(e.text) < 20:
                         print(f"点击疑似菜单项: {e.text}")
                         e.click(by_js=True)
                         time.sleep(3)
                         return True

            print(f"未在弹出菜单中找到包含 '{target_text}' 的选项")
                        
        else:
            print("未找到 '+' 新建按钮")
        
        return False

    def run_code(self, code):
        """在当前编辑器中写入并运行代码"""
        print(f"\n--- 开始执行脚本 ---\n代码预览: {code[:50]}...")
        
        editor_area = self.tab.ele('css:textarea.inputarea')
        if not editor_area:
            print("错误: 未找到代码编辑区域")
            return None

        editor_area.focus()
        time.sleep(0.5)
        # 清空
        self.tab.actions.key_down(Keys.CTRL).type('a').key_up(Keys.CTRL).type(Keys.DELETE)
        time.sleep(0.5)
        # 写入
        self.tab.actions.type(code)
        time.sleep(1)

        # 运行
        run_btn = None
        candidates = self.tab.eles('css:[title="运行"]') + self.tab.eles('css:[title="Run"]')
        for btn in candidates:
            if btn.states.is_displayed:
                run_btn = btn
                break
        
        if not run_btn:
             text_candidates = self.tab.eles('text:运行')
             for c in text_candidates:
                 if c.tag != 'span' and c.states.is_displayed:
                     run_btn = c
                     break

        if run_btn:
            print(f"点击运行按钮...")
            run_btn.click(by_js=True)
        else:
            print("使用快捷键运行...")
            editor_area.focus()
            self.tab.actions.type(Keys.F5)
            time.sleep(0.5)
            self.tab.actions.key_down(Keys.CTRL).type(Keys.ENTER).key_up(Keys.CTRL)
        
        time.sleep(2)

        # 结果捕获
        print("正在获取运行结果...")
        match = re.search(r'console\.log\("([^"]+)"', code)
        if match:
            keyword = match.group(1).split('!')[0]
            logs = self.tab.eles(f'text:{keyword}')
            found_logs = [l.text for l in logs if l.states.is_displayed]
            if found_logs:
                print(f"运行成功! 捕获输出: {found_logs}")
                return found_logs
        return None

def main():
    # 1. 初始化，指定新的文档 URL
    new_url = 'https://www.kdocs.cn/l/ccMnE2s1F11G'
    bot = KDocsAirScriptBot(url=new_url)
    
    # 2. 确保页面加载
    bot.ensure_page_loaded()
    
    # 3. 打开编辑器
    bot.open_editor()
    
    # 4. 新建脚本 (避免覆盖已有代码)
    if bot.create_new_script(version="2.0"):
        print("新建脚本成功，准备写入测试代码")
    else:
        print("新建脚本失败，将使用当前脚本覆盖测试 (请注意备份)")
    
    # 5. 运行测试代码
    test_code = 'console.log("Hello KDocs! New Script Test. Time: " + new Date().toLocaleTimeString());'
    bot.run_code(test_code)

if __name__ == '__main__':
    main()
