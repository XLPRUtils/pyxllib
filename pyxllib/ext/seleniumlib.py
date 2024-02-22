#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2023/03/28

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchWindowException


class element_has_text(object):
    def __init__(self, locator, text):
        self.locator = locator
        self.text = text

    def __call__(self, driver):
        element = driver.find_element(*self.locator)
        if self.text in element.text:
            return element
        else:
            return False


class XlChrome(webdriver.Chrome):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximize_window()

    def locate(self, locator, seconds=10):
        """ 定位一个元素 """
        if isinstance(locator, str):  # 默认是XPATH格式
            locator = (By.XPATH, locator)
        return WebDriverWait(self, seconds).until(EC.presence_of_element_located(locator))

    def click(self, locator, seconds=10, check=True):
        """ 点击一个元素 """
        if isinstance(locator, str):
            locator = (By.XPATH, locator)
        if check:
            element = WebDriverWait(self, seconds).until(EC.element_to_be_clickable(locator))
        else:
            element = self.locate(locator, seconds)
        time.sleep(0.5)  # 最好稍微等一下再点击
        try:
            element.click()
        except ElementClickInterceptedException:
            # 特殊情况，例如小鹅通下载页面的"下载"按钮没法正常click，要用js脚本去click
            self.execute_script("arguments[0].click();", element)

    def locate_text(self, locator, text, seconds=10):
        """ 判断指定元素位置是否含有指定文本 """
        if isinstance(locator, str):
            locator = (By.XPATH, locator)
        return WebDriverWait(self, seconds).until(element_has_text(locator, text))

    def __bool__(self):
        """ 判断driver是否还存在，如果已被手动关闭，这个值会返回False """
        try:
            self.title
            return True
        except NoSuchWindowException:
            return False


def get_global_driver(_driver_store=[None]):  # trick
    """ 通过这个接口可以固定一个driver来使用 """
    if _driver_store[0] is None:
        _driver_store[0] = XlChrome()
    if not _driver_store[0]:  # 如果驱动没了，重新启动
        _driver_store[0] = XlChrome()
    return _driver_store[0]
