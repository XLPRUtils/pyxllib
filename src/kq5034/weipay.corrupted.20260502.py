"""閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌ｉ埡鍐剧劸闁告鍛亾閸︻厼浠ф鐐叉喘婵?"""`r`n
from .common import *  # noqa: F403
from .wechat_runtime import KqWechat

class Weipay(DpWebBase):
    """ 閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢?"""

    def __init__(self, users=None):
        """
        :param users: 婵犵鈧啿鈧綊鎮樻径灞惧暫濞达絿鎳撻ˉ鍥煟椤旀槒鍏屾繛鍛劥閵囨劙寮撮悩鍨儯闂佺懓鐏濈粔宕囩礊閺冨牊鏅悘鐐靛亾绗戦梺鍛婄啲缁犳帡銆呴敃鍌氱煑闁硅揪鑵归崑鎾存媴缁嬭法鍘掗梺鍦焾濞层倝鎮炬ィ鍐╁剭闁告洦浜為鍗灻瑰鍕槈濞存粍娲樼粚閬嶆晬閸曨偄寮?
        """
        super().__init__('https://pay.weixin.qq.com')
        self.user = None  # 閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌ｉ妸銉ヮ伀闁轰降鍊濋獮瀣暋閻楀牆鈧娊鎮规担瑙勭凡缂?闁哄鏅滈悷銈夋煂濠婂懏鍟哄ù锝堫潐閺夊綊鏌涢敂瑙勬ogin()婵炴垶鏌ㄩ鍛櫠閻樼粯鏅?
        if users:
            # 闂佸湱鐟抽崱鈺傛杸闂佽皫鍡╁殭缂傚秴绉归弻鍛村及韫囨洖绔奸梺鎸庣☉閼活垶銆呰瀵顭ㄩ崘顏咁嚇闂佸搫鐗滄禍鐐测枍閵夈劊浜归柡鍥╁仜閻忕喖鎮圭€ｎ亜鏆熼柡浣靛€栫粋宥呯暆閳ь剙菐椤曗偓閹秵鎷呴悜姗嗕划閻熸粎澧楀ú鏍吹闁秴鏋?
            self.login(users)

    def login(self, users=None):
        """
        濠碘槅鍋€閸嬫捇鏌＄仦璇插姎闁艰崵鍠愬鍕礋椤撶喎鈧偤鏌ｈ椤曆呯礊瀹ュ鐒婚柡鍕箳鐢棝鏌ㄥ☉妯肩劯闁逞屽墯娣囪櫣鎹㈤崘鈺冾洸鐎光偓閳ь剙菐椤曗偓閹秵鎷呴悜姗嗕划閻熸粎澧楀ú鐔煎Υ婢舵劖顥堥柕蹇嬪灪缁绢垶鏌涢幒鎾愁棆闁轰降鍊濋獮瀣煥鐎ｎ亶妫侀梺娲诲枙閻掞箓宕瑰璺虹闁绘梻琛ラ崑?

        :param users: 闂佸憡鐟﹂崹鍦垝娴兼潙浼犳い蹇撳暟閺勫倻鈧敻鍋婇崰鏇熺┍婵犲洦鍋ㄩ柕濠忕畱閻撴洟鎮介锝呮灈缂佺粯鐗犻獮宥夘敃閵堝洤鐏?
            婵?['闂佸搫鍊稿ú锝呪枎閵忊€愁嚤闁绘娅ｇ紙濠氭煕閺傜儤娅嗗?, '闂傚倸瀚悧鍡楊焽閸愭祴鏋?, '闁诲孩绋掗悷锕傚籍?]
            None, 婵犵鈧啿鈧綊鎮樻径鎰珘妞ゅ繐瀚弲鎼佹煥濞戞鐒哥紒顕呭灣閹峰濡堕崶鈺傛儯閻庡灚婢橀幊搴ｇ礊鐎ｎ喖绀堢€广儱妫欑花姘舵煕閿濆啫濮傛繛闂村嵆瀹曟粌顓奸崶銊ф殸闂佺锕﹂崢褔鎮捐缁傚秹宕崟顒€鏋€闂佺儵鏅涢悺銊ф暜閹绢喖绠ユい鎰╁€楅崹?
        """
        tab = self.tab
        if not tab.url == 'https://pay.weixin.qq.com/index.php/core/info':
            tab.get('https://pay.weixin.qq.com')

            # 闁荤姳绀佹晶浠嬫偪閸℃鈻旈柍褜鍓氱粙澶愵敂閸涱垰鐏遍柣鐘辩劍濠㈡銇愭径鎰厒闊洢鍎崇粈澶愭煟椤剙濡虹紒顭戝墴楠炴帟顦查柛鈺傤殔椤曘儵顢欑拋宕囩畾濠电偞鍨甸悧濠冨閸涘瓨鍤婇柛鎰皺濮ｅ矂鏌涘▎鎰伌闁逞屽厸濡炴帞绮╅弶鎴旀瀻?
            message_sent = False

            # 婵炶揪缍€濞夋洟寮?while 閻庣敻鍋婃禍鐐虹嵁閸℃稒鏅€光偓閳ь剙煤閸ф绀嗛柡澶婄仢閻撳倿寮堕崼銏犱壕閻庡灚姘ㄩ埀顒冾潐濮樸劌鈻?URL 闂侀潻闄勫妯侯焽?
            while not tab.url == 'https://pay.weixin.qq.com/index.php/core/info':
                # 1 婵炲瓨绮岄惉鐓幥庨鈧幆宥嗘媴閸撳弶鎼愰梺鍝勵儐缁繒鎹㈤崘顔煎珘闁绘柨鍚嬫禒姗€鎮烽弴姘鳖槮闁糕晜鐩?
                div = tab('tag:div@@class=qrcode-img')
                try:
                    is_invalid = div('tag:div@@class=alt@@text():婵炲瓨绮岄惉鐓幥庨鈧幆宥嗘媴閹肩偘鏉梺?, timeout=3)
                except DrissionPage.errors.ContextLostError:  # 婵炴垶鎸撮崑鎾绘煠閸撴彃澧俊鍙夋倐閹倻鎷犻懠顒傂梺鍛婅壘閸戠晫妲愬┑瀣闁告劦浜為ˇ閬嶆偠濮樼厧浜伴柛锝呮憸缁辨棃顢欓懡銈呮櫓
                    is_invalid = None

                if is_invalid:  # 闂傚倸娲犻崑鎾绘偡閺囨氨顦﹂柛鈺傜洴瀵剚锛愭担铏规缂傚倷绀侀顓㈡偉?
                    logger.info(self.get_recive('婵炲瓨绮岄惉鐓幥庨鈧幆宥嗘媴缁嬪灝娈ラ柡澶嗘櫅濞诧箑锕㈤敓鐘虫櫖鐎光偓閸愭儳娈查梺鍛婄懄閸ㄥ潡鍩€椤戞寧顦烽柟骞垮灲楠炲洩绠涢弮鍌傘儵鏌熼褍鐏茬紒杈ㄧ箞閺屽苯顓奸崱妯煎帎闁荤喐鐟辩粻鎴ｃ亹閸岀偞鍤旂€瑰嫭婢樼徊鍧楁煛閸艾浜鹃梺鍝勫€块。锔捐姳閳哄啰纾奸悗闈涙憸閸?))

                    tab.refresh()
                    message_sent = False

                if message_sent:
                    time.sleep(5)
                    # 婵犵鈧啿鈧綊鎮樻径濠庡晠闁肩⒈鍓涢惀鍛存煕濞嗘劕鐏撮柍褜鍏涘ù鍥╂崲閸愩劉妲堥柛顐ゅ枍缁辨牠鏌ㄥ☉妯垮婵?continue 缂備焦绋戦ˇ顖滄閻斿吋鍋ㄩ柕濠忕畱閻撴洟鏌熷▓鍨簽闁?
                    continue

                # 2 闂佸吋鍎抽崲鑼躲亹閸ャ劎顩茬€光偓閳ь剙菐椤曗偓閹秵鎷呴崨濠勵洯闂?
                div = tab('tag:div@@id=IDQrcodeImg')
                # 241223闂佸憡绋忛崝瀣博?1:16闂佹寧绋戦惌浣烘崲閺嶃劎鈻旀い蹇撳暟閻洟鏌￠崘锝嗘珖闁稿鍨介弻銊╁川椤掑倸鏅╅梺鎸庣☉閺堫剟宕濋崨顖涘枂濠㈣泛锕ょ拋鏌ユ煠鐎圭姵顥夐柟閿嬫緲椤曘儵顢欑拋宕囩畾婵炲瓨绮岄惉鐓幥庨鈧幆宥嗘媴閻戞鏆犳繛瀵稿Л閸嬫挸鈽夐弬璺ㄥ閻㈩垱妞介幆宥嗘媴閾忚鎯ｉ梺鍝勭墕閹碱偊宕哄Δ鍛珘濠㈣泛锕よぐ鐘绘煥濞戞ɑ婀伴柛銊︾矋濞艰螣缂佹锕傛煙楠炲灝鐏╅悽顖ｅ亰瀵爼濡疯閻ｉ亶鏌￠崒婊勫殌缂佹唻绻濆畷銉檪缂佽鲸绻堝畷銏ゆ偄閸涘﹪妾烽梻渚囧枦婵倝鎯€閸涙潙绀勫Δ锕佹硶缁犳牠姊婚崒銈呮珝妞?
                file = div('tag:img').save(XlPath.tempdir(), 'qrcode')  # dp闂佸湱绮崝鎺旀閻㈢鍋撴担鍐炬綈婵犫偓娑旂备c闁诲繒鍋熼崑鐐哄焵椤戭剙鎳忛悾閬嶆煕閵夈儱鈷斿褎鐗犻幊娑㈩敂閸曨倣妤€鈽夐幘鎰佸創婵炴潙娲幆鍐礋椤愩埄娼梺?

                # 3 闂佸憡鐟﹂崹鍧楀焵椤戣法顦﹀ù婊勬礃缁岄亶鍩勯崘顎儵鏌熼褍鐏茬紒杈ㄧ箞楠炴捇骞囬弶鎸庢珨闂佹椿娼块崝宥夊春濞戙垹绠ユい鎰╁€楅崹鎶芥煟瑜戦褏绱?
                if users:
                    for user in users:
                        wechat_lock_send(user, '闂佸吋婢橀崯顐も偓姘卞枎椤斿繘濡烽妶鍥┾枙闂傚倸娲犻崑鎾绘偡閺囨氨绛忕紒杈ㄧ箚缁犳盯顢曢姀鐘电泿闂佺懓鐡ㄩ崹鑸电珶閸岀偞鍎樺ù锝囧劋椤忋垻鎲搁悧鍫熺濞存粍娲樼粚閬嶅焺閸愨晜娈版繛?, files=[file])
                    time.sleep(5)  # 闂佸憡鐟﹂崹鐢告偩椤掑嫬鐐婇柛鎾楀喚鏆繛鎴炴尨閸嬫捇鏌ら崜鎻掑⒉妞も晪濡囩划鍨緞婢跺瞼鐛?
                    with get_autogui_lock():
                        KqWechat.闂佽顔栭崑鍡涙偉濠婂牊鍎岄悹鍥皺缁夎法鈧敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢?users[0])
                else:
                    print('>> 闁荤姴娲弨杈ㄧ珶閸岀偞鍎樺ù锝囧劋椤忋垻鎲搁悧鍫熺妞も晝绮妵鍕濞戞瑥鈧敻鏌ㄥ☉妯肩伇闁宠鍚嬮幆鏃囩疀鎼达絿鐛ラ梺鐓庮殠娴滄粍鎱ㄩ埡鍐＜鐟滃繘鎮界紒妯讳氦闁归偊鍨奸弨?..')

                # 闁荤姳绀佹晶浠嬫偪閸℃稑鍐€闁搞儺浜堕崬鍫曟煕濞嗘劧鑰块柛锝囧劋缁?True闂佹寧绋戦惉濂稿灳濡崵鈹嶆繝闈涳工濞堥箖鎮樿箛鏂跨仴鐟滄澘鍊块弻鍛媴妞嬪寒浼囧┑鐐叉閸撴繄绮旈悜钘夌畳?
                message_sent = True

        self.user = tab('tag:a@@class=username').text.split('@')[0]

    def __1_闂佺硶鏅炲▍锝夈€侀崨鏉戠闁绘棁顕ч崢?self):
        pass

    def 闂備焦褰冪粔椋庢崲濞戙垹鍐€闁搞儮鏅╅崝顔碱渻?self):
        """DrissionPage 闂?tab 闁诲海鏁搁、濠囨寘閸曨垰纾婚悹楦挎濮ｆ粌霉閸忚壈澹樻繝鈧鍛簻闁汇垹鎲″銊╂偣閸濆嫮鏋冩繛鏉戞瀹曘儲鎯旀幊娴滃ジ鏌℃担鍝ュ⒊缂佽鲸绻堥弻灞筋吋閸℃鍘愰梺鍛婄懄閻楁梻绮╃€涙鈻旀い蹇撳缁夊ジ鏌涢幘宕囆㈢憸鐗堢叀閹?tab闂?""
        try:
            tab = get_latest_not_dev_tab(self.browser)
            if tab:
                self.tab = tab
                return tab
        except Exception:
            pass

        self.tab = self.browser.latest_tab
        return self.tab

    def download_monthly_records(self, month, save_dir=True):
        """ 婵炴垶鎸搁鍫澝归崶顒€瀚夐柛顐ｇ矊閺佹粓鎮硅鐎氫即宕抽崜褎瀚婚柨鏇楀亾鐎?

        :param month: 闂佸搫鐗嗛悧鍛村船閵堝鏅悘鐐插⒔婢规劗鈧鍠栫换瀣嫻?2024-03"
            闂佸搫鐗滈崜姘辨暜閹绢喖鐭楁い蹇撴噺閺嗘粓鏌熼梹鎰妽閻庡灚绮撳顐も偓娑櫳戠粻鎴濃槈閹炬剚鍎撴繛?
        :param save_dir: 婵烇絽娲︾换鍌炴偤閵娾晛鎹堕柕濠忓閸樻瑩鏌ｉ埡濠傛灈缂傚秴绉电粙?
            True, 婵烇絽娲︾换鍌炴偤閵娾晛绀嗗ù鐓庮嚟鐢盯鎮规担闈涒偓鎾舵嫻閻斿娼?
            str, 闂佺缈伴崕鐢稿极婵犲洤绀嗛柣妤€鐗嗛惁褰掓倵鐟欏嫭鐨戞繛鍙夊閵?

        >> Weipay().download_monthly_records('2024-07')
        """
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cfund_bill_nc/funds_bill_nc#/')  # 闁荤姍鍐伃闁革絽澧介幏褰掓晝閳ь剙鐣?

        start_day = month + '-01'
        end_day = month + f'-{str(pd.Period(month).end_time.day)}'

        tab.wait(3)
        tab.wait.ele_displayed('tag:input@@placeholder=閻庢鍠掗崑鎾斥攽椤旂⒈鍎愭俊顐熸櫊瀵?)
        tab.action_type('tag:input@@placeholder=閻庢鍠掗崑鎾斥攽椤旂⒈鍎愭俊顐熸櫊瀵?, start_day)
        tab.action_type('tag:input@@placeholder=缂傚倷鐒﹂幐璇差焽椤愶箑绫嶉柕澶涢檮閸?, end_day + '\n')
        tab.wait(3)
        tab('tag:button@@text()=闂佸搫琚崕鎾敋?).click()
        tab.wait(5)

        if save_dir:
            tab('tag:a@@class=popups download').click()
            ele0 = tab('tag:div@@class=el-dialog__wrapper new-capital-down-dialog@@text():闁荤姵鍔х粻鎴濈暦閻旂厧绠ラ柟鎯у暱閻﹀爼鎮楅悷鐗堟拱闁搞劍宀搁弫宥呯暆閸愭儳娈茬紒缁㈠弾閸犳盯顢樼紒妯尖枖閻庯綆鍘界粊?)
            src_file = XlPath(ele0('.el-button el-button--primary', timeout=60).click.to_download().wait(show=False))

            if save_dir is True:
                save_dir = xlhome_dir('data/m2112kq5034/闂佽桨鑳舵晶妤€鐣垫担鐑樺仒?)
            else:
                save_dir = XlPath(save_dir)
            name = re.sub(r'_\d+\.csv$', '.csv', src_file.name)  # 闂佸憡甯炴繛鈧繛鍛叄瀹曘儲鎯旈敐鍥╋紱闂佸憡鐟崹鐢稿礂濡吋鏆滈柨鏃囧Г缁犳帡鏌ｉ妸銉ヮ仼妞わ箑娼￠幃褔宕奸悢鍛婂闂佽桨鐒﹀姗€鎮鸿瀵粙宕堕宥呮暪
            dst_file = save_dir / name
            shutil.copy(src_file, dst_file)
            return dst_file

    def daily_update(self, today=None):
        """ 濠殿噯绲界换鎰板Φ婢舵劕鍗抽悗娑櫳戦悡鈧柣鐘冲姧缁犳垵鐣烽悢鐓庢瀬闁绘鐗嗙粊?

        - 婵犵鈧啿鈧綊鎮樻径鎰強?闂佸憡鐟╅弨閬嶅垂?闂佸憡鐟ラ崵鏍濠靛洨鈻旈悗锝庡幗缁佹澘鈽夐幘绛规闂佸弶绮撳鐢稿醇濠靛洤顏悷婊呭閹歌锕㈤埀顒勬煟閵娿儱顏璺哄瀹?
        - 闂佺绻戝﹢鍦垝椤掑嫬绫嶉柕澶涢檮閸╁倿鏌涘▎妯圭胺缂佹鎳忓顏堟寠婢跺瞼歇闂佸搫鐗嗛悧鎰緞閸曨垰纭€?
        """
        dst_files = []

        # 1 闁荤姳绶ょ槐鏇㈡偩缂佹顩峰┑鐘插€舵禍顖炴煙绾版ê浜鹃柣蹇曞仧閸嬫盯宕熼悙顒傤浄濠殿喗鍔忛崑鎾存媴鐟欏嫮鐣辨繛瀵稿Т椤洟鍩€椤戣法鍔嶆俊顐熸櫊瀵?
        today = today or pd.Timestamp.now()
        current_year = today.year
        current_month = today.month
        current_day = today.day

        # 2 婵犵鈧啿鈧綊鎮樻径鎰睄闁靛闄勯崺鍌炴煛?闂佸憡鐟╅弨閬嶅垂?闂佸憡鐟ラ崵鏍濠靛牊鍟哄ù锝夘棑閻熸捇寮堕悙鏉戠亰缂佹鍊圭粙澶愵敂閸涱喚鐣遍梺姹囧妼鐎氼參寮抽悢鐓庣?
        if current_day in [1, 2]:
            # 闁荤姳绶ょ槐鏇㈡偩缂佹鈻斿┑鐘辩窔閸ゅ鏌￠崼婵堝ⅱ婵炲牊鍨块悰顔锯偓娑櫳戠粻?
            if current_month == 1:
                last_month_year = current_year - 1
                last_month = 12
            else:
                last_month_year = current_year
                last_month = current_month - 1

            # 婵炴垶鎸搁鍫澝归崶銊р枖濠电姳绶氶崵瀣煛閸繄澧涢柡鍡欏枛楠?
            last_month_str = f"{last_month_year}-{str(last_month).zfill(2)}"
            # 缂傚倸鍟崹鍧楀Υ婢舵劕鐭楁い鏍ㄧ箓閸樻潙鈽夐幘宕囆ら懣娆撴倵鐟欏嫮鍟茬紒杈ㄧ箞瀹曪綁顢涘▎搴ｉ瀺婵犮垼鍩栨穱娲儊椤栫偛绀勯柣妯诲墯閸?
            for i in range(4):
                try:
                    dst_file = self.download_monthly_records(last_month_str)
                    break
                except DrissionPage.errors.NoRectError:
                    pass
            else:
                dst_file = self.download_monthly_records(last_month_str)

            if dst_file is not None:
                dst_files.append(dst_file)

        # 3 婵炴垶鎸搁鍫澝归崶顏備汗闁规儳鐡ㄧ粻鎴︽煛娴ｅ搫顣肩€?
        if current_day == 1:  # 閻熸粎澧楅幐璇诧耿閳ь剛绱?婵犮垹鐏堥弲婊冣枔韫囨稑绫嶉柡鍫㈡暩閻熸繈鏌ら搹顐㈢亰缂佹鎳忓顏堝棘閵堝洨顦柣鐔告磻缁€渚€宕埀顒傜磼濞戞﹩妲风紒鏂跨摠瀵?
            return dst_files

        current_month_str = f"{current_year}-{str(current_month).zfill(2)}"
        for i in range(4):
            try:
                dst_file = self.download_monthly_records(current_month_str)
                break
            except DrissionPage.errors.NoRectError:
                pass
        else:
            dst_file = self.download_monthly_records(current_month_str)

        if dst_file is not None:
            dst_files.append(dst_file)

        return dst_files

    def search_refund(self, voucher_id):
        """ 闂佸搫琚崕鍙夌珶濡吋濯奸柕蹇曞Т缁€瀣⒑椤愮喎浜惧┑鐐叉瑜扮偞绌辨繝鍥х畳?"""
        # 1 闂佸搫琚崕鍙夌珶濡吋濯奸柕蹇曞Т缁€?        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/trade/search_new')

        voucher_id = str(voucher_id).lstrip("`'")
        input_name = 'mmpay_order_id' if re.match(r'\d+$', voucher_id) else 'merchant_order_id'
        tab.find_ele_with_refresh(f't:input@@name={input_name}').input(voucher_id)
        tab('t:a@@id=idQueryButton').click()

        # 2 闂佺缈伴崕鐢稿极?
        tips = tab('t:div@@class=tips-error').text
        # 闁荤姴娲ㄩ弻澶愬吹閿斿彞鐒婇柟瀛樼箰缂嶆牕鈽夐幘顖氫壕婵炴垶鎼╂禍婵嬫偂閿涘嫭瀚氶柕蹇曞Т缁€瀣煕?
        # 闂佸搫琚崕鎾敋濡や礁绶為弶鍫亯琚?閻庣敻鍋婇崰鏇熺┍婵犲嫭濯奸柕蹇曞Т缁€瀣煕濞嗘垶鐒跨紒棰濆弮瀹曟濡烽妶鍥╂喒濠殿喗绻愮徊鍧楀灳?
        if tips:  # 闂佺缈伴崕鐢稿极?
            return {'error': tips}

        # 3 闁荤姳闄嶉崹鐟扮暦閻斿摜鈹嶉柍鈺佸暕缁?
        html = tab.find_ele_with_refresh('t:div@@class=table-wrp with-border')('t:table').html
        # 闂佺儵鎳囬弳鍧媋utifulsoup闁荤喐鐟辩徊楣冩倵閼恒儲浜ゆ繛鍡樺姦閸炵禑tml闂佹寧绋戦張顒佷繆缁嬪晝鎺曠疀鎼淬劌娈漷h闁诲海鏁搁幊鎾惰姳閺屻儲鍎嶉柛鎾虫▔婵炲濮伴崕閬嶆偤瑜斿畷妤呭川婵犲啰鍘甸悗娈垮枛缁绘顣鹃梺鍝勵儐閸旀洟宕甸銏犵骇?
        soup = BeautifulSoup(html, 'lxml')
        trs = soup.find_all('tr')
        row = {}
        for tr in trs:
            th = tr.find('th')
            if th:
                th = th.text.strip()
                td = tr.find('td')
                if td:
                    td = td.text.strip()
                    row[th] = td

        # {'闂佸摜鍠庡Λ娆撳春濞戞碍濯奸柕蹇曞Т缁€瀣煕?: 'SX2TGC-0OZRE8O-EFG9',
        # '闂佽　鍋撴い鏍ㄧ懅鐢盯鏌涘Δ浣圭鐟?: '4200002706202505304477731244',
        # '婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?: '闂備緡鍠撻崝宀勫垂鎼淬劍鐒婚柍褜鍓欓埢搴ㄥ箚瑜忛弳姘舵煙?,  # 婵炴垶姊瑰姗€顢欏鍜佸晠闁告瑥顦伴弳婊兠? 闂佺绻堥崝鎴β烽崒鐐寸劵闁逞屽墮閳诲酣骞嗚閺嗘岸鏌?
        # '闁荤姳闄嶉崹鐟扮暦閻斿吋鐓傞柟杈惧瘜閺?: '620.00闂?,
        # '婵炲瓨鍤庨崐鏍ｅΔ鍛睄闁割偅娲橀敍?: '2025-05-30 21:29:06'
        row['闁荤姳闄嶉崹鐟扮暦閻斿吋鐓傞柟杈惧瘜閺?] = float(row['闁荤姳闄嶉崹鐟扮暦閻斿吋鐓傞柟杈惧瘜閺?].strip('闂?))

        # 4 閻庤鐡曠亸顏嗘崲閹存績鏋?
        match row['婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?]:
            case '婵炴垶姊瑰姗€顢欏鍜佸晠闁告瑥顦伴弳婊兠?:
                row['閻庤鐡曠亸顏嗘崲閹存績鏋?] = 0
            case '闂備緡鍠撻崝宀勫垂鎼淬劍鐒婚柍褜鍓欓埢搴ㄥ箚瑜忛弳姘舵煙?:
                details = self.search_refund_details(row.get('闂佸摜鍠庡Λ娆撳春濞戞碍濯奸柕蹇曞Т缁€瀣煕?) or row.get('闂佽　鍋撴い鏍ㄧ懅鐢盯鏌涘Δ浣圭鐟?),
                                                     query_type='auto',
                                                     raise_err=False)
                if details:
                    row['閻庤鐡曠亸顏嗘崲閹存績鏋?] = sum(item['闂備緡鍋€閸嬫挻绻涢崱娑氱暫闁革絿鍋撻敍?] for item in details)
                else:
                    tab2 = tab('t:a@@id=reqReturnBn').click.for_new_tab()
                    row['閻庤鐡曠亸顏嗘崲閹存績鏋?] = float(tab2('t:div@@class=form').eles('t:div@@class=form-item')[3]('t:span').text)
                    tab2.close()
            case '闂佺绻堥崝鎴β烽崒鐐寸劵闁逞屽墮閳诲酣骞嗚閺嗘岸鏌?:
                row['閻庤鐡曠亸顏嗘崲閹存績鏋?] = row['闁荤姳闄嶉崹鐟扮暦閻斿吋鐓傞柟杈惧瘜閺?]
            case _:
                logger.warning('闂佸搫鐗滄禍鐐烘偂閳╁啰顩查柕鍫濇椤粓鏌ｅΟ鍨厫闁逞屽厸缁躲倗妲? + row['婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?])
                row['閻庤鐡曠亸顏嗘崲閹存績鏋?] = row['婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?]

        闁诲繐绻戠换鍡涙儊椤栫偛绀傞柟鎯板Г閿涙棃姊洪幓鎺斝㈡い锕€寮堕妵鍕偨閸涘﹥銆?self.browser, reason='闂佸搫琚崕鎾敋濡も偓椤曘儵顢欑拋宕囩畾闂佽　鍋撴い鏍ㄧ懅鐢盯鎮规担闈涚仼鐎规洜鍠栧畷銉︽償閵忊剝姣嗛柣?,
                 keep_tab_ids=[getattr(self.tab, 'tab_id', None)])
        return row

    @staticmethod
    def _normalize_refund_query_type(voucher_id, query_type='auto'):
        query_type = (query_type or 'auto').strip().lower()
        if query_type != 'auto':
            return query_type

        voucher_id = str(voucher_id).lstrip("`'").strip()
        if re.fullmatch(r'\d+', voucher_id):
            return 'pay_order' if voucher_id.startswith('42') else 'refund_id'
        return 'merchant_order'

    def _fill_precise_refund_query(self, voucher_id, query_type='auto'):
        """闂備緡鍋€閸嬫挻绻涢崱娆忎壕闁绘搫绱曢幏鐘诲閵堝啠鍋撴径鎰剭闁告洦鍠氱紙濠氭煕韫囧鍔滄い銏狀儏閳讳粙鍩勯崘鈺冪暢缂備礁顑呴崯鍧楁偩?name闂佹寧绋戦懟顖濄亹瑜旈幊妤呮嚍閵夈儳妯嗙紓渚囧枛婢т粙鍨惧鈧濠氬Ψ椤垵娈戦梺鍛婄墬濡炰粙宕抽悜鑺ュ剭闁告洦鍨埀顒€瀛╅幆鏃囩疀閹垮嫮绋忛梺鍛婂姈閻燂綁鍩€?""
        voucher_id = str(voucher_id).lstrip("`'").strip()
        query_type = self._normalize_refund_query_type(voucher_id, query_type)
        input_index_map = {
            'pay_order': 0,
            'merchant_order': 1,
            'refund_id': 2,
        }
        if query_type not in input_index_map:
            raise ValueError(f'闂佸搫鐗滄禍鐐烘偂閿熺姵鐒婚柍褜鍓欓埢搴ㄦ倷椤掑倸寮ㄩ柣鐘叉储閸ㄨ崵鎮锕€鍨傞悗锝呭缁愭⒋query_type}')

        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/refundquery')
        tab.wait(2)
        tab.ele('tag:a@@id=preciseRefundSearchBtn', timeout=10).click(by_js=True)
        tab.wait(1)

        js = r"""
const voucherId = arguments[0];
const targetIndex = arguments[1];
const box = document.querySelector('.preciseRefundSearch.preciseQuery');
if (!box) return 'NO_BOX';
const inputs = [...box.querySelectorAll('input.ipt')];
if (inputs.length < 3) return `BAD_INPUTS:${inputs.length}`;
for (const input of inputs) {
  input.value = '';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  input.dispatchEvent(new Event('change', {bubbles: true}));
}
const target = inputs[targetIndex];
target.focus();
target.value = voucherId;
target.dispatchEvent(new Event('input', {bubbles: true}));
target.dispatchEvent(new Event('change', {bubbles: true}));
const btn = box.querySelector('a.btn.btn-primary');
if (!btn) return 'NO_QUERY_BTN';
btn.click();
return 'OK';
"""
        result = tab.run_js(js, voucher_id, input_index_map[query_type])
        if result != 'OK':
            raise RuntimeError(f'闂備緡鍋€閸嬫挻绻涢崱娆忎壕闁绘搫绱曢幏鐘诲閵堝啠鍋撴径瀣畽妞ゆ劑鍨归弲绋款熆閹壆绨块悷娆欑畵閺佸秴顫㈢缓鐠璼ult}')

        deadline = time.time() + 20
        while time.time() < deadline:
            table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=3)
            if not table:
                time.sleep(0.5)
                continue

            table_text = table.text.strip()
            if '濠殿喗绻愮徊钘夛耿椤忓牆钃熼柕澶樼厛閸? in table_text:
                time.sleep(0.5)
                continue

            return
        raise RuntimeError('闂備緡鍋€閸嬫挻绻涢崱娆忎壕闁绘搫绱曢幏鐘诲閵忋垺灏濋梺鍝勵儐缁秴锕㈤銏犳嵍闁靛ň鏅╅弳鏇㈡煛閸垹鏋戞俊鐐插€垮濠氬级閹存繃鏆ラ梺鍛婃⒒婵儳霉閸モ斁鍋撻悷鐗堟拱闁?)

    @staticmethod
    def _parse_refund_query_table_html(html):
        soup = BeautifulSoup(html, 'lxml')
        records = []
        for tbody in soup.select('tbody'):
            rows = tbody.find_all('tr', recursive=False)
            if len(rows) < 2:
                continue

            summary_row, detail_row = rows[0], rows[1]
            if 'sub-th' not in (summary_row.get('class') or []):
                continue

            summary = {}
            for span in summary_row.select('span.th-item'):
                text = span.get_text(' ', strip=True)
                if '闂? not in text:
                    continue
                key, value = text.split('闂?, 1)
                summary[key.strip()] = value.strip()

            cells = detail_row.find_all('td', recursive=False)
            if len(cells) < 5:
                continue

            row = {
                '婵炲瓨鍤庨崐鏍ｅΔ鍛闁哄洨鍋涙繛?: summary.get('婵炲瓨鍤庨崐鏍ｅΔ鍛闁哄洨鍋涙繛?, ''),
                '闂佸摜鍠庡Λ娆撳春濞戙垹纭€闁哄洨鍋涙繛?: summary.get('闂佸摜鍠庡Λ娆撳春濞戙垹纭€闁哄洨鍋涙繛?, ''),
                '闂備緡鍋€閸嬫挻绻涢崱妯哄姢闁伙綆鍓熼獮瀣箛椤撶噥妲梻?: summary.get('闂備緡鍋€閸嬫挻绻涢崱妯哄姢闁伙綆鍓熼獮瀣箛椤撶噥妲梻?, ''),
                '闂備緡鍋€閸嬫挻绻涢崱妯哄姢鐎规洜鍠栧畷?: cells[0].get_text(' ', strip=True),
                '闂備緡鍋€閸嬫挻绻涢崱娑氱暫闁革絿鍋撻敍?: float(cells[1].get_text(' ', strip=True) or 0),
                '闂備緡鍋€閸嬫挻绻涢崱妤€鈷斿┑顔芥倐楠炩偓?: cells[2].get_text(' ', strip=True),
                '闂佹眹鍨归悿鍥敋椤掍胶顩?: cells[3].get_text(' ', strip=True),
                '闂佸湱绮崝鎺戭潩閿曞倸绫嶉柛顐ｆ礃閿?: cells[4].get_text(' ', strip=True),
            }
            records.append(row)
        return records

    def _get_refund_query_page_state(self):
        tab = self.tab
        pager = tab.ele('tag:div@@class=pagination fr', timeout=2)
        if not pager:
            return 1, 1

        labels = pager.eles('tag:label')
        if len(labels) >= 2:
            try:
                return int(labels[0].text.strip()), int(labels[1].text.strip())
            except Exception:
                pass
        return 1, 1

    def _goto_next_refund_query_page(self, previous_first_refund_id=''):
        tab = self.tab
        next_btn = tab.ele('tag:a@@class=btn page-next', timeout=5)
        if not next_btn:
            return False

        next_btn.click(by_js=True)
        deadline = time.time() + 15
        while time.time() < deadline:
            table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=2)
            if table:
                records = self._parse_refund_query_table_html(table.html)
                if records and records[0]['闂備緡鍋€閸嬫挻绻涢崱妯哄姢鐎规洜鍠栧畷?] != previous_first_refund_id:
                    return True
            time.sleep(0.5)
        raise RuntimeError('闂備緡鍋€閸嬫挻绻涢崱娆忎壕闁绘搫绱曢幏鐘诲閵忋垹鈧啿顪冮妶澶嬫锭闁诡喗顨堢槐鎺楀箻鐎甸晲鍑介梺鍝勭墱娴滄繂煤閸ф妫?)

    def search_refund_details(self, voucher_id, query_type='auto', raise_err=True):
        """闂佸搫琚崕鎾敋濡ゅ懏鐒婚柍褜鍓欓埢搴ㄦ倷椤掍緡娼犵紓鍌欑濡宕归鍡樺仒闁靛绲洪崑?
        婵帗绋掗…鍫ヮ敇婵犳艾绠板璺猴功閺嗘岸鏌熺€涙ê濮堟俊鐐插€垮濠氬级閹存繄锛撻柟鍏兼綑缁绘鎹㈤幋锕€鐐婇柣鎴濇川缁€澶嬫叏閿濆懐绠叉繛纰卞灣閹峰寮剁捄銊梺鍛婄墪閹冲酣骞嗛崼銉︾劵闁逞屽墮閳诲酣骞嗚缁€瀣煕濞嗘劗浠涢柍褜鍏涚欢姘跺焵椤戣棄浜惧┑鐐叉缁犳牠宕抽悙顒婄矗婵犻潧瀚ч崑鎾寸瑹閳ь剟鍩€椤戣棄浜惧┑鐐叉婢ц姤鎱ㄩ幖浣哥畱濞达絿顣介崑鎾存媴閻戞ɑ娈奸柣鐘叉穿濞撶懓效婢舵劕违濞达絿鎳撶徊鐟懊瑰┃鍨偓鏍ь渻閸岀偞鈷掓繛锝庡厴閸嬫挻绗熼埀顒勫焵椤戣棄浜惧┑鐐叉閸庢娊鎮鹃鍕闁归偊鍘介ˇ褔姊婚崒娑氭殬闁?        """
        try:
            self._fill_precise_refund_query(voucher_id, query_type)
            tab = self.tab
            body_text = tab('tag:body').text
            if '闂佸搫妫楅崐鐟拔涢妶澶婃瀬闁绘鐗嗙粊? in body_text:
                return []

            all_records = []
            seen_refund_ids = set()

            while True:
                table = tab.ele('tag:div@@class=table-wrp with-border table-receive', timeout=5)
                if not table:
                    break

                records = self._parse_refund_query_table_html(table.html)
                for row in records:
                    refund_id = row['闂備緡鍋€閸嬫挻绻涢崱妯哄姢鐎规洜鍠栧畷?]
                    if refund_id in seen_refund_ids:
                        continue
                    seen_refund_ids.add(refund_id)
                    all_records.append(row)

                current_page, total_pages = self._get_refund_query_page_state()
                if current_page >= total_pages:
                    break

                first_refund_id = records[0]['闂備緡鍋€閸嬫挻绻涢崱妯哄姢鐎规洜鍠栧畷?] if records else ''
                self._goto_next_refund_query_page(first_refund_id)

            def sort_key(row):
                try:
                    return pd.to_datetime(row['闂備緡鍋€閸嬫挻绻涢崱妯哄姢闁伙綆鍓熼獮瀣箛椤撶噥妲梻?])
                except Exception:
                    return pd.Timestamp.max

            all_records.sort(key=sort_key)
            return all_records
        except Exception:
            if raise_err:
                raise
            return []

    def request_single_refund(self, voucher_id, refund_amount=0, refund_reason=''):
        """ 闂佹眹鍨归悿鍥敋椤掑嫬纭€闁哄洨鍠愰拏瀣⒑椤愮喎浜惧┑?        :param voucher_id: 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楅梺鍨儏閻忔鏌ら弶鎸庡櫣闁哄懌鍨介獮瀣偪椤栵絽鎮呴梺鍛婎殕濞叉牞銇愰崸妤佺劸闁兼亽鍎查弳婊堟煙?
        >> wp.request_single_refund('SFW1WL-0OZRE8O-KX63', 0.01, '濠电偞娼欓鍫ユ儊椤栫偞鐒婚柍褜鍓欓埢?)
        >> wp.request_single_refund('4200002199202406302648230239', 0.01, '濠电偞娼欓鍫ユ儊椤栫偞鐒婚柍褜鍓欓埢?)
        """
        baseline_refunded_amount = 0
        try:
            row = self.search_refund(voucher_id)
            if 'error' not in row:
                baseline_refunded_amount = float(row.get('閻庤鐡曠亸顏嗘崲閹存績鏋?) or 0)
        except Exception:
            pass

        # 1 闂佸搫琚崕鎾敋濡ゅ啯濯奸柕蹇曞Т缁€?        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/core/refundapply')
        input_name = 'wxOrderNum' if re.match(r'\d+$', voucher_id) else 'mchOrderNum'

        tab.find_ele_with_refresh(f't:input@@name={input_name}').input(voucher_id)
        tab('#applyRefundBtn').click()

        # '闁荤姴娲ㄩ弻澶屾椤撱垹绀傞柕澶涘瘜閸斺偓缂佺虎鍙庨崰妤呮偋缁嬫鍤曢煫鍥ㄦ礃閻ｅ崬霉濠у灝鈧牕危濡ゅ懎纭€闁哄洨鍋涙繛?  # 闂佸搫绉堕崢褏妲愰敓鐘崇叆婵炲棙甯╅崵?
        # '闁荤姳鐒﹀妯肩礊瀹ュ棛鈻旂€广儱鎳愰幗鐘绘煕?  # 婵炴垶鎸搁敃顏勵焽娴煎瓨鐓ユ繛鍡樺俯閸?
        # '闁荤姳闄嶉崹鐟扮暦閻斿摜鈻旂€广儱鎳愰幗鐘绘煕?  # 婵炴垶鎸搁澶婎焽娴煎瓨鐓ユ繛鍡樺俯閸?
        # '闁荤姳闄嶉崹鐟扮暦閻斿鍟呴柟缁樺笒瀵灝螞閻楀牏绠為柍褜鍏橀崑鎾寸箾?

        # 2 婵犻潧顦介崑鍕疮閹惧瓨浜ら柡鍐ｅ亾妞ゆ垶鐟ч幃浼村Ω閿旀儳顥?
        tab.find_ele_with_refresh('t:input@@name=refund_amount').input(refund_amount)
        tab('#textInput').input(refund_reason)
        tab('#commitRefundApplyBtn').click(by_js=True)
        tab.wait(2)

        # 3 婵°倗濮撮惌渚€鎯?        self.婵犻潧顦介崑鍕疮閹惧灈鍋撻棃娑欘棤闁绘牗绮嶇粙澶嬫償閵娧冪煑闁荤姴娲ｉ懗鍫曟偉?tab)

        # 4 缂備焦绋戦ˇ顖滄閻斿鍤楁い鏃囶唺缁诲棗銆掑鈧崨顔肩劯濠殿喗绻愮徊浠嬎囬埡鍛仩闁糕剝顨堥弳姘舵煙鐎涙ê濮囬柟顔筋殜瀹曟ê顓奸崼銏㈩唹闂佹悶鍎抽崕鎴犳濠靛鐒奸柛顭戝枛鐢啿鈽夐幘绛瑰伐闁活亝澹嗛幏鐘活敍濠垫劕鏁归梺鐐藉劜缁矂宕欓敓鐘崇劵闁逞屽墮閳诲酣鏌ㄩ妤€浜?        self.wait_refund_completion(
            voucher_id=voucher_id,
            expected_refund_amount=refund_amount,
            baseline_refunded_amount=baseline_refunded_amount,
        )

    def get_recive(self, content):
        """ 闂佸吋鍎抽崲鑼鏉堛劎顩风紓浣姑竟鍫ユ煕濞嗗繒效妞も晩鍙冨畷姗€宕ㄩ褍鏅?"""

        with WeChatSingletonLock(120) as wx:
            recive_msg = None
            wx.SendMsg(content, '闂佸吋婢橀崯顐も偓姘卞枛瀹曘儲鎯旈垾鎶藉彙')

            while recive_msg is None:
                wx._show()
                wx.ChatWith('闂佸吋婢橀崯顐も偓姘卞枛瀹曘儲鎯旈垾鎶藉彙')
                msgs = wx.GetAllMessage()

                for msg in msgs[::-1]:
                    # if msg.msg_type != 'receive':  # 闂佺绻愰悧鍡涘矗瑜旈獮鎺撳緞鐎ｎ亜顦查梺琛″亾闁煎摜鏁稿暩闂佽鍙庨崹鍐测枔閹寸姷妫柨鏃囧Г鐏忓棝鏌ㄥ☉妯绘拱闁哄瞼鍠愮粭鐔虹紦閻庢epc_aw婵炴垶鎸搁敃锕傚极閵堝棛鈻旈悹鍥紦缁ㄥ啿菐閸よ翰鍊曢ˇ鈺呮煕濞嗘垶鐒块柣銊у枔閹?
                    #     continue

                    # 闂佸憡鐟禍锝夊礂濮椻偓楠炲秹骞嗚閻忕娀鏌熼棃娑毿ｇ憸棰佺窔濮婂顢欓崗鐓庘偓鐢告煟閵娿儱顏柡灞斤躬瀹曟﹢宕ㄩ褍鏅?
                    if msg.content == content and msg.sender == 'Self':
                        break

                    recive_msg = msg.content
                    if recive_msg:
                        break

                time.sleep(3)

        return recive_msg

    def get_vcode(self):
        """ 闂佺懓鍢查悘婵嬪春瀹€鍕剹妞ゆ挻绻€缁诲棗螖閻樿尙鐒烽柣锕€顦甸幆?"""

        with WeChatSingletonLock(120) as wx:
            vcode = None

            content = f'@{self.user} 閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢稓鈧灚澹嬮崑鎾趁归敐鍥ｅ褎绮撳鐢稿传閸曨偆宀涙繛瀛樼矊濡梻绮╃€涙鈻旀い蹇撴噺閺嗘粌霉閻樼鑰块柣娑欑懅閹风姵鎷呴搹鐟扮仯闂佹寧绋戦惌渚€顢氶鍕摕闁靛／鍕暯闂佹悶鍎抽崑娑⑺囬崣澶屸枖?
            wx.SendMsg(content, '闂佸吋婢橀崯顐も偓姘卞枛瀹曘儲鎯旈垾鎶藉彙')

            while True:
                wx._show()
                wx.ChatWith('闂佸吋婢橀崯顐も偓姘卞枛瀹曘儲鎯旈垾鎶藉彙')
                msgs = wx.GetAllMessage()

                for msg in msgs[::-1]:
                    # if msg.msg_type != 'receive':  # 闂佺绻愰悧鍡涘矗瑜旈獮鎺撳緞鐎ｎ亜顦查梺琛″亾闁煎摜鏁稿暩闂佽鍙庨崹鍐测枔閹寸姷妫柨鏃囧Г鐏忓棝鏌ㄥ☉妯绘拱闁哄瞼鍠愮粭鐔虹紦閻庢epc_aw婵炴垶鎸搁敃锕傚极閵堝棛鈻旈悹鍥紦缁ㄥ啿菐閸よ翰鍊曢ˇ鈺呮煕濞嗘垶鐒块柣銊у枔閹?
                    #     continue
                    vals = re.findall(r'\d+', msg.content)
                    # 闁哄鏅涘ú锕傚箮閵堝绠冲鑸靛姈濮ｆ劙骞栨潏鍓х窗缂佹顦靛?闂?
                    vals = [v for v in vals if len(v) == 6]
                    if vals:
                        vcode = vals[0]
                        # we.send_text(f'閻庡湱顭堝鍫曞极瑜版帒绀嗗ù鐓庮嚟瀹曪綁鎮归崶锕佸厡闁绘牗绮撻弫宥咁潰缁虹浛ls[0]}')
                        break

                    # 闂佸憡鐟禍锝夊礂濮椻偓楠炲秹骞嗚閻忕娀鏌熼棃娑毿ｇ憸棰佺窔濮婂顢欓崗鐓庘偓鐢告煟閵娿儱顏柡灞斤躬瀹曟﹢宕ㄩ褍鏅?
                    if msg.content == content and msg.sender == 'Self':
                        break

                if vcode:
                    break

                time.sleep(3)

        return vcode

    def wait_refund_completion(self, timeout=300, voucher_id=None,
                               expected_refund_amount=None, baseline_refunded_amount=0):
        """缂備焦绋戦ˇ顖滄閻斿憡浜ら柡鍐ｅ亾妞ゆ垶鐟ч埀顒傛嚀閺堫剟宕瑰璺何?
        闁荤姴顑呴崯鎶芥儊椤栨稒浜ゆ繛鍡樺姦閸炰粙姊洪锝嗩潡缂侀鍋婂顔炬媼閸︻厾顦繝銏″劶缁墽鎲撮敂鐐暫濞达絿鍎ょ痪顖炴煙閹帒鍔滃┑顔肩箻閺屻劌鈻庡▎鎴狀槷婵炴垶鎸哥粔鐑藉礂濮椻偓瀹曟ê顓奸崱姗堥獜闂傚倸瀚崝鏇㈠春濞戙垹妫樺ù鍏兼綑濞呫垽鎮归崶銊︾┛缂佽鲸绻堝畷銉╂晝閳ь剟宕归鐐村剳闁绘棃顥撻弶浠嬫煟濠婂嫭绶叉繝褉鍋撴繛鏉戝悑娣囨椽锝為敃鍌氱闁冲搫鍟壕浼存煏?        """
        if not voucher_id:
            return self.wait_batch_refund_completion(timeout=timeout)

        tab = self.tab
        deadline = time.time() + timeout
        last_status_text = ''

        while time.time() < deadline:
            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''

            if any(x in body_text for x in ['闂備緡鍋€閸嬫挻绻涢崱妤€鈷旈柡浣烘暩閹风姴鐣￠弶鍨闂佸湱绮崝鎺戭潩閿曞倸绠ｉ柟閭﹀墮椤?, '闂佸湱绮崝鎺戭潩閿曞倸绠ｉ柟閭﹀墮椤?]):
                self.闁诲繐绻戠换鍡涙儊椤栫偞鍊烽柣鐔告緲濮ｅ﹪寮堕埡鍌涒拹妞ゆ垶鐟╅獮鎾诲箛椤忓懎鏀梺鍛婅壘濞村嘲鈻撻幋锕€绠甸柟閭﹀枔娴犳盯鏌熺粙娆炬█闁?tab, timeout=3)
                tab.wait(1)

            if voucher_id:
                try:
                    row = self.search_refund(voucher_id)
                except Exception as e:
                    last_status_text = f'闂佸搫琚崕鍗炵暦閻斿鍤曢柛灞炬皑閸╂鏌ㄥ☉娆樺姱e}'
                    time.sleep(2)
                    continue

                if 'error' in row:
                    last_status_text = str(row['error'])
                    time.sleep(2)
                    continue

                refunded_amount = float(row.get('閻庤鐡曠亸顏嗘崲閹存績鏋?) or 0)
                trade_status = str(row.get('婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?) or '')
                last_status_text = f'婵炲瓨鍤庨崐鏍ｅΔ鍛亹闁煎摜顣介崑?{trade_status}闂佹寧绋戦懟顖炲礄閳╁啯浜ら柡鍐ｅ亾妞?{refunded_amount}'
                target_amount = baseline_refunded_amount + float(expected_refund_amount or 0)

                if refunded_amount + 1e-9 >= target_amount:
                    return
                if trade_status in ['闂備緡鍠撻崝宀勫垂鎼淬劍鐒婚柍褜鍓欓埢搴ㄥ箚瑜忛弳姘舵煙?, '闂佺绻堥崝鎴β烽崒鐐寸劵闁逞屽墮閳诲酣骞嗚閺嗘岸鏌?] and not expected_refund_amount:
                    return

                time.sleep(2)
                continue

            time.sleep(1)

        raise RuntimeError(
            '閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌涘Δ浣圭闁荤噥浜弻鍛村焵椤掆偓閳诲酣宕滆濞夈垽鏌＄€ｎ偆鐭嬫繝鈧鐘亾閻熺増婀伴柛銊﹀哺閺?
            f'url={tab.url}闂佹寧绋戦鎼僼le={tab.title}闂佹寧绋戦惉鍏兼叏閹间礁绠戝ù锝堫潐閸犲懘鎮?{last_status_text!r}'
        )

    @staticmethod
    def _normalize_page_text(text):
        return re.sub(r'\s+', ' ', str(text or '')).strip()

    @staticmethod
    def _get_element_render_state(ele):
        js = r"""
const el = this;
let hiddenAncestor = false;
let zIndex = 0;
let p = el;
while (p) {
  const style = getComputedStyle(p);
  const cls = (p.className || '').toString();
  if (cls.includes('hide') || style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity) === 0) {
    hiddenAncestor = true;
    break;
  }
  const zi = parseInt(style.zIndex, 10);
  if (!Number.isNaN(zi)) zIndex = Math.max(zIndex, zi);
  p = p.parentElement;
}
const rect = el.getBoundingClientRect();
return `${hiddenAncestor ? 1 : 0}|${rect.width}|${rect.height}|${zIndex}`;
"""
        try:
            raw = ele.run_js(js)
            hidden_flag, width, height, z_index = str(raw).split('|', 3)
            return {
                'hidden_ancestor': hidden_flag == '1',
                'width': float(width),
                'height': float(height),
                'z_index': int(float(z_index or 0)),
            }
        except Exception:
            return {'hidden_ancestor': True, 'width': 0.0, 'height': 0.0, 'z_index': -1}

    def _is_element_really_visible(self, ele):
        state = self._get_element_render_state(ele)
        return not state['hidden_ancestor'] and state['width'] > 0 and state['height'] > 0

    def _click_visible_text_action(self, tab, texts):
        for target in texts:
            candidates = []
            locators = [
                f'tag:a@@text()={target}',
                f'tag:button@@text()={target}',
                f'tag:span@@text()={target}',
                f'tag:a@@text():{target}',
                f'tag:button@@text():{target}',
                f'tag:span@@text():{target}',
            ]
            for locator in locators:
                try:
                    for ele in tab.eles(locator):
                        state = self._get_element_render_state(ele)
                        if state['hidden_ancestor'] or state['width'] <= 0 or state['height'] <= 0:
                            continue
                        candidates.append((state['z_index'], state['width'] * state['height'], ele))
                except Exception:
                    continue
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                candidates[0][2].click(by_js=True)
                return target
        return ''

    def _raise_if_weipay_auth_invalid(self, body_text):
        text = self._normalize_page_text(body_text)
        login_markers = ['闁荤姴娲╁〒瑙勭箾閸ヮ剚鍋ㄩ柕濞垮劙缁ㄥ啿菐閸ワ絽澧插鐟板€块幆宥嗘媴閻戞﹩浠悷?, '婵炲瓨绮岄惉鐓幥庨鈧幆宥嗘媴閹肩偘鏉梺?, '閻庣敻鍋婇崰鏇熺┍婵犲洤绠ユい鎰剁磿椤忛亶鏌熷▓鍨簽婵炲懌鍎撮妵?, '闁荤姴娲弨杈ㄧ珶閸岀偞鍎樺ù锝囧劋椤忋垻鎲?]
        if any(marker in text for marker in login_markers):
            raise RuntimeError('閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌ｈ椤曆呯礊瀹ュ绠戝ù锝囶焾閸ゆ帒顭块幆浼村摵闁哄懌鍎甸弫宥呯暆閸愭儳娈查梻浣瑰絻缁夌敻寮绘繝鍥х妞ゆ劑鍊楅崹鎶芥煟瑜戦褏绱炲澶婅Е閹肩补鈧櫕娅冮梺鍦懗閸♀晜鏂€闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢?)
        if '闂佸搫妫楅崐鐟邦渻閸岀偛绫嶉柣妯硅閸ゅ鏌涢弮鍌氭灆闁稿繑锕㈠鍫曞礃椤旂瓔鈧? in text and '闁荤姴娲ㄩ弻澶嬬閸垹瀵查悹鍥у级濞呭矂鏌熼弶鎴濇Щ闁? in text:
            raise RuntimeError('閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢稓鎲搁悧鍫熷碍濠⒀呭█閹倻鎷犻懠顒傂梺璇″厸閼宠泛顔忔潏鈺€鐒婇柟杈剧秵閸熷繘姊婚崒銈呮灍婵炵厧鐗撳浠嬪箛閸撲胶顦柣鐘叉川閸忔﹢宕抽幖浣告闁绘鐗嗛ˉ鍥煟椤旇崵鍔嶉柛銊ｅ妿閳ь剛鎳撻張顒勫垂濮樻墎鍋撻悷閭︽Ц闁告瑥绻戦〃銉ョ暆閸愵亝顫嶉梺鍛婅壘妤犳悂宕埀顒勬偣?)

    def _extract_batch_refund_status(self, submit_started_at=None):
        tab = self.tab
        js = r"""
const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
const isVisible = (el) => {
  if (!el) return false;
  const style = window.getComputedStyle(el);
  const rect = el.getBoundingClientRect();
  return style.display !== 'none'
    && style.visibility !== 'hidden'
    && rect.width > 0
    && rect.height > 0;
};
return [...document.querySelectorAll('table')]
  .filter(isVisible)
  .map((table, tableIndex) => ({
    tableIndex,
    text: normalize(table.innerText || table.textContent),
    rows: [...table.querySelectorAll('tr')]
      .map((row, rowIndex) => ({
        rowIndex,
        text: normalize(row.innerText || row.textContent),
      }))
      .filter(row => row.text),
  }))
  .filter(table => table.rows.length);
"""
        try:
            tables = tab.run_js(js) or []
        except Exception:
            return None

        status_tokens = [
            ('failure', '婵犮垼娉涚€氼噣骞冩繝鍐ㄧ窞閺夊牜鍋夎'),
            ('failure', '闂備緡鍠撻崝宀勫垂鎼淬垹绶為弶鍫亯琚?),
            ('failure', '闂備緡鍋€閸嬫挻绻涢崱妯哄姢闁靛洦鍨归幏?),
            ('success', '閻庣懓鎲¤ぐ鍐囬埡鍛仩?),
            ('success', '婵犮垼娉涚€氼噣骞冩繝鍕ㄥ亾閻熺増婀伴柛?),
            ('success', '閻庣懓鎲¤ぐ鍐偩椤掑嫬绠?),
            ('processing', '婵犮垼娉涚€氼噣骞冩繝鍐枖?),
            ('processing', '閻庡灚婢橀幊搴ㄋ囬埡鍛仩?),
            ('processing', '婵犮垼娉涚€氼噣骞冩繝鍐枖妞ゆ挶鍔庣粈澶愭偣閸ヮ剦妫戦柍绗哄灲瀹?),
        ]
        candidates = []
        for table_index, table in enumerate(tables):
            rows = table.get('rows') or []
            for row in rows[:10]:
                row_text = self._normalize_page_text(row.get('text'))
                if not row_text:
                    continue

                status_kind = None
                status_token = ''
                for kind, token in status_tokens:
                    if token in row_text:
                        status_kind = kind
                        status_token = token
                        break
                if not status_kind:
                    continue

                row_time = None
                matched = re.search(r'20\d{2}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}:\d{2}', row_text)
                if matched:
                    try:
                        row_time = pd.to_datetime(matched.group(0))
                    except Exception:
                        row_time = None

                score = 0
                if table_index == 0:
                    score += 20
                if row.get('rowIndex') == 1:
                    score += 50
                elif row.get('rowIndex') == 0:
                    score += 10
                if submit_started_at is not None and row_time is not None:
                    delta = abs((row_time - submit_started_at).total_seconds())
                    score += max(0, 100 - min(delta, 100))

                candidates.append({
                    'status_kind': status_kind,
                    'status_token': status_token,
                    'row_text': row_text,
                    'row_time': row_time,
                    'table_index': table_index,
                    'row_index': row.get('rowIndex'),
                    'score': score,
                    'table_text': self._normalize_page_text(table.get('text'))[:300],
                })

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item['score'], item['table_index'], item['row_index']))
        return candidates[0]

    def _goto_batch_refund_query_view(self, timeout=10):
        tab = self.tab
        locators = [
            'tag:a@@text():闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏?,
            'tag:button@@text()=闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏?,
            'tag:span@@text()=闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏?,
        ]
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                body_text = self._normalize_page_text(tab('tag:body').text)
            except Exception:
                body_text = ''

            if '闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏? in body_text and self._extract_batch_refund_status():
                return True

            clicked = False
            for locator in locators:
                try:
                    for ele in tab.eles(locator):
                        if not self._is_element_really_visible(ele):
                            continue
                        ele.click(by_js=True)
                        clicked = True
                        tab.wait(1)
                        break
                except Exception:
                    continue
                if clicked:
                    break

            if not clicked:
                clicked_text = self._click_visible_text_action(tab, ['闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏?])
                if clicked_text:
                    clicked = True
                    tab.wait(1)

            if clicked:
                continue

            time.sleep(1)
        return False

    def _refresh_batch_refund_query_view(self):
        tab = self.tab
        locators = [
            'tag:button@@text()=闂佸搫琚崕鎾敋?,
            'tag:a@@text():闂佸搫琚崕鎾敋?,
        ]
        for locator in locators:
            try:
                for ele in tab.eles(locator):
                    if not self._is_element_really_visible(ele):
                        continue
                    ele.click(by_js=True)
                    tab.wait(1)
                    return True
            except Exception:
                continue

        clicked_text = self._click_visible_text_action(tab, ['闂佸搫琚崕鎾敋?])
        if clicked_text:
            tab.wait(1)
            return True
        return False

    def wait_batch_refund_completion(self, timeout=300, submit_started_at=None, file_name='',
                                     initial_popup_result=None):
        """缂備焦绋戦ˇ顖滄閻旂厧绠ョ憸鐗堝笒濞呫倝姊洪鐔蜂壕濠电偛妫楁晶钘夛耿閳ユ剚娼伴柨婵嗘媼濡查亶鏌ｉ悙鍙夘棞闁伙綆鍓熼獮瀣箛閳规儳浜?
        闁哄鏅滈悷鈺呭闯闁垮鈻旂€广儱鐗嗛崢鏉懨归悩顔煎姉闁逞屽墯缁秷銇愭担鍦洸闁靛牆妫楅悘鍥煕閺冨倸鏋庨柤鏉戯功缁絽螖閳ь剟宕甸銏″仢闁瑰灚鏋奸崑鎾斥攽閸涱垳鈻曟繛?step4 缂傚倷鐒﹂幐璇差焽椤愶箑绾ч柍銉ュ级椤愪粙鏌ㄥ☉妯肩劮闁逞屽墮閺堫剙危閸濄儲鍟哄ù锝堟閹煎ジ鏌涢幒鎾寸凡闁诡喗顨堢槐鎺楊敇閻斿妫楀┑鐐叉４缁辨洘鎱ㄩ幖浣哥畱濞达絽顫曢埀顒€鍟粋鎺撴償閳藉棙袩闂佽崵鍋涘Λ鏃堟嚈閹达箑鐭楁俊顖濐嚙閻忓洨鈧懓鎲¤ぐ鍐囬埡鍛仩闁糕剝鍔忛崑?        """
        tab = self.tab
        deadline = time.time() + timeout
        last_status_text = ''
        submit_success_seen = bool(initial_popup_result and initial_popup_result.get('saw_submit_success', False))
        query_view_opened = bool(initial_popup_result and initial_popup_result.get('page_changed', False))
        last_refresh_at = 0

        while time.time() < deadline:
            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''
            normalized_body = self._normalize_page_text(body_text)
            self._raise_if_weipay_auth_invalid(normalized_body)

            popup_result = self.闁诲繐绻戠换鍡涙儊椤栫偞鍊烽柣鐔告緲濮ｅ﹪寮堕埡鍌涒拹妞ゆ垶鐟╅獮鎾诲箛椤忓懎鏀梺鍛婅壘濞村嘲鈻撻幋锕€绠甸柟閭﹀枔娴犳盯鏌熺粙娆炬█闁?tab, timeout=2)
            submit_success_seen = submit_success_seen or popup_result.get('saw_submit_success', False)
            query_view_opened = query_view_opened or popup_result.get('page_changed', False)

            batch_state = self._extract_batch_refund_status(submit_started_at=submit_started_at)
            if batch_state:
                last_status_text = batch_state['row_text'][:300]
                if batch_state['status_kind'] == 'success':
                    logger.info(f'闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄥ箚瑜滃Σ閬嶆煟閻愬弶顥滈柣锝庡墴楠炲骞囬崜浣虹崶{last_status_text}')
                    return
                if batch_state['status_kind'] == 'failure':
                    raise RuntimeError(
                        f'閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌熼棃娑氱Ш闁革絾妞介弻鍛村焵椤掆偓閳诲酣鎮欓鈧ˉ妤佺箾閸″繆鍋撻幇浣剐熼梺鑽ゅ仜濡濡甸幋鐘冲闁靛鍨崇粈濉甶le={file_name!r}闂?
                        f'url={tab.url}闂佹寧绋戦鎼僼le={tab.title}闂佹寧绋戦惉鍏兼叏閹间礁绠戝ù锝堫潐閸犲懘鎮?{last_status_text!r}'
                    )

                if time.time() - last_refresh_at >= 5:
                    self._refresh_batch_refund_query_view()
                    last_refresh_at = time.time()
                time.sleep(2)
                continue

            last_status_text = normalized_body[:300]
            if submit_success_seen or popup_result.get('clicked', False):
                if not query_view_opened:
                    query_view_opened = self._goto_batch_refund_query_view(timeout=5)
                    if query_view_opened:
                        tab.wait(1)
                        continue

                if time.time() - last_refresh_at >= 5:
                    self._refresh_batch_refund_query_view()
                    last_refresh_at = time.time()

            time.sleep(1)

        raise RuntimeError(
            '閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鏌熼棃娑氱Ш闁革絾妞介弻鍛村焵椤掆偓閳诲酣宕滆濞夈垽鏌＄€ｎ偆鐭嬫繝鈧鐘亾閻熺増婀伴柛銊﹀哺閺?
            f'file={file_name!r}闂佹寧绋戦惁鍧甽={tab.url}闂佹寧绋戦鎼僼le={tab.title}闂佹寧绋戦惉鍏兼叏閹间礁绠戝ù锝堫潐閸犲懘鎮?{last_status_text!r}'
        )

    def 闁诲繐绻戠换鍡涙儊椤栫偞鍊烽柣鐔告緲濮ｅ﹪寮堕埡鍌涒拹妞ゆ垶鐟╅獮鎾诲箛椤忓懎鏀梺鍛婅壘濞村嘲鈻撻幋锕€绠甸柟閭﹀枔娴犳盯鏌熺粙娆炬█闁?self, tab, timeout=15):
        """閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯寮堕埡鍌涒拹妞ゆ垶鐟╅獮鎾诲箛椤忓懎鏀梺鍛婅壘閸戠晫妲愬┑瀣；閻犻缚娅ｅВ婊兠归崗鑹板妞わ箑娼″畷娆愭姜閹殿噮浼囬柣蹇曞仜閸婂鍨惧Ο鍏煎闁靛牆鎳撻崜銊х磼閹邦収娈ｇ紒閬嶄憾瀵灚寰勬繝鍌滀户婵犮垼娉涚€氼噣骞冩繝鍥ㄦ櫖閻忕偛澧藉楣冩煛閸繍妲告慨妯稿妿缁辨帟顦撮柣銏狀煼婵?""
        success_markers = ['闂備緡鍋€閸嬫挻绻涢崱妤€鈷旈柡浣烘暩閹风姴鐣￠弶鍨闂佸湱绮崝鎺戭潩閿曞倸绠ｉ柟閭﹀墮椤?, '闂佸湱绮崝鎺戭潩閿曞倸绠ｉ柟閭﹀墮椤?]
        locators = [
            'tag:a@@text():闁哄鏅滅粙鎴﹀矗閸℃稒鐒婚柍褜鍓欓埢搴ㄦ倷椤掑倸寮ㄩ柣?,
            'tag:button@@text()=闁哄鏅滅粙鎴﹀矗閸℃稒鐒婚柍褜鍓欓埢搴ㄦ倷椤掑倸寮ㄩ柣?,
            'tag:a@@text():缂傚倷缍€閸涱垱鏆伴梻渚囧亐閸嬫挻绻涢崱娆忎壕闁瑰ジ鏀遍幏?,
            'tag:button@@text()=缂傚倷缍€閸涱垱鏆伴梻渚囧亐閸嬫挻绻涢崱娆忎壕闁瑰ジ鏀遍幏?,
            'tag:a@@text():闁哄鏅滅粙鎴﹀矗閸℃稑绠ラ悷娆忓閸嬔囨煛鐏炶鍔ユい?,
            'tag:button@@text()=闁哄鏅滅粙鎴﹀矗閸℃稑绠ラ悷娆忓閸嬔囨煛鐏炶鍔ユい?,
            'tag:a@@class=btn btn-primary close-dialog JSCloseDG',
            'tag:a@@id=FinishProtocolBn',
            'tag:a@@text()=闁哄鏅滅粙鎴﹀矗閸℃稑鐤柛鈩兠悡鏇燁殽閻愭彃鏆辩憸?,
            'tag:a@@tabindex=2',
            'tag:a@@text()=闁诲海鎳撻張顒勫垂?,
            'tag:button@@text()=闁诲海鎳撻張顒勫垂?,
            'tag:a@@text()=缂佺虎鍙庨崰娑㈩敇?,
            'tag:button@@text()=缂佺虎鍙庨崰娑㈩敇?,
            'tag:span@@text()=缂佺虎鍙庨崰娑㈩敇?,
            'tag:a@@text()=闂佹椿鍘归崕鍨閻愵剛顩?,
            'tag:a@@text()=闁哄鏅滈弻銊ッ?,
            'tag:button@@text()=闁哄鏅滈弻銊ッ?,
            'tag:i@@class=el-dialog__close el-icon el-icon-close',
            'tag:i@@class=el-dialog__close',
        ]
        deadline = time.time() + timeout
        start_url = tab.url
        clicked_any = False
        saw_submit_success = False
        while time.time() < deadline:
            clicked_this_round = False
            for locator in locators:
                try:
                    for btn in tab.eles(locator):
                        if not self._is_element_really_visible(btn):
                            continue
                        btn.click(by_js=True)
                        tab.wait(1)
                        clicked_any = True
                        clicked_this_round = True
                        break
                except Exception:
                    continue
                if clicked_this_round:
                    break

            if clicked_this_round:
                if tab.url != start_url:
                    return {'clicked': True, 'saw_submit_success': saw_submit_success, 'page_changed': True}
                continue

            try:
                body_text = tab('tag:body').text
            except Exception:
                body_text = ''

            if any(x in body_text for x in success_markers):
                saw_submit_success = True
                clicked_text = self._click_visible_text_action(
                    tab,
                    ['闁哄鏅滅粙鎴﹀矗閸℃稑绠ラ悷娆忓閸嬔囨煛鐏炶鍔ユい?, '闁哄鏅滅粙鎴﹀矗閸℃稒鐒婚柍褜鍓欓埢搴ㄦ倷椤掑倸寮ㄩ柣?, '缂傚倷缍€閸涱垱鏆伴梻渚囧亐閸嬫挻绻涢崱娆忎壕闁瑰ジ鏀遍幏?, '缂佺虎鍙庨崰娑㈩敇?, '闁诲海鎳撻張顒勫垂?, '闂佹椿鍘归崕鍨閻愵剛顩?, '闁哄鏅滈弻銊ッ?]
                )
                if clicked_text:
                    clicked_any = True
                    tab.wait(1)
                    if tab.url != start_url:
                        return {'clicked': True, 'saw_submit_success': saw_submit_success, 'page_changed': True}
                    continue

            if any(x in body_text for x in ['闂佸綊娼х紞濠囧闯濞差亝鐒婚柍褜鍓欓埢搴ㄦ倷椤掆偓椤ユ绻涢崱蹇撳⒉闁绘搫绱曢幏?, '婵犮垼娉涚€氼噣骞冩繝鍐枖?, '閻庣懓鎲¤ぐ鍐囬埡鍛仩?]):
                return {'clicked': clicked_any, 'saw_submit_success': saw_submit_success, 'page_changed': tab.url != start_url}
            time.sleep(1)
        return {'clicked': clicked_any, 'saw_submit_success': saw_submit_success, 'page_changed': tab.url != start_url}

    def 婵犻潧顦介崑鍕疮閹惧灈鍋撻棃娑欘棤闁绘牗绮嶇粙澶嬫償閵娧冪煑闁荤姴娲ｉ懗鍫曟偉?self, tab):
        # 1 婵犻潧顦介崑鍕疮閹惧灈鍋撻棃娑欘棤闁?        inputs = tab.eles('tag:input@@class=real-input')
        passwd = XlEnv.get(f'XL_KQ_PAY_PASSWORD_{self.user}', decoding=True)
        if not passwd:
            passwd = XlEnv.get('XL_KQ_PAY_PASSWORD', decoding=True)
        if passwd:
            inputs[0].input(passwd, clear=True)

        # 2 闂佹椿鍙庨崢鐑樼┍婵犲喚娈界€光偓閸愵亝顫嶉梺?
        if len(inputs) > 1:
            tab('tag:a@@text():闂佸憡鐟﹂崹鍧楀焵椤戣儻鍏岄柣鎿冨幗缁?).click()
            # vcode = self.get_vcode()
            time.sleep(10)
            with get_autogui_lock():
                vcode = KqWechat.婵炲濮寸€涒晠宕抽幐搴ｎ洸濡わ附瀵х粊顕€鏌涘▎鎰伇妤犵偛绻愰銉ノ旈崘顏勫綃婵烇絽娲犻埀顒€鍟块弫鍫曟倵?)
            inputs[1].input(vcode, clear=True)

        # 3 缂佺虎鍙庨崰娑㈩敇婵犳艾绠板鑸靛姈鐏?
        time.sleep(1)
        tab('tag:a@@text()=缂佺虎鍙庨崰鏍偩缂嶆class=btn btn-primary align-center').click()

        # 4 闂佸搫鐗嗛ˇ顔捐姳閻戞ǜ浜滈柣銏犳啞濡椼劌霉閸忕厧鍝烘い銈呭€瑰鍕冀瑜戦崜銊モ槈閹绢垰浜惧┑鐐叉閸撴繆銇愭担铏圭焼闁绘艾顕粈澶愭煛閸繍妲告慨妯稿妽瀵板嫰宕熼鐔封偓鐐烘煥濞戞ɑ婀伴柣銉ユ嚇瀵灚寰勬繝鍐闂佺儵鏅涢悺銊ф暜鐎靛摜纾肩憸蹇涙偨婵犳艾违?        return self.闁诲繐绻戠换鍡涙儊椤栫偞鍊烽柣鐔告緲濮ｅ﹪寮堕埡鍌涒拹妞ゆ垶鐟╅獮鎾诲箛椤忓懎鏀梺鍛婅壘濞村嘲鈻撻幋锕€绠甸柟閭﹀枔娴犳盯鏌熺粙娆炬█闁?tab)

    def request_file_refund(self, file=None):
        """ 闂備緡鍋呮穱铏规崲閸愵喖妫橀柛銉檮椤愪粙寮堕埡鍌滎灱妞ゃ垺鍨块獮宥堛亹閹烘垶顏￠梻渚囧亐閸嬫挻绻?

        :param file:
            str, 婵炴垶鎸搁敃锝囨閸洖绠伴柛銉戝懏姣庨梺鍝勫€稿ú锝呪枎?
            None, 闂佺厧顨庢禍婊勬叏閳哄懎绠ラ柟顖嗗啰鍘掗梺鍝勭墐閸嬫捇鏌￠崒娑橆棆婵炲牊鍨跺濠氬籍閳ь剟顢栧▎鎾虫闁搞儻闄勯?

        """
        # 1 闂佺懓鐏氶崕鎶藉春瀹€鍕珘妞ゆ巻鍋撴繝鈧幘顔煎珘闁逞屽墴瀵剟骞嶉鐣屾殸闁哄鏅滃褰掝敄濞嗘挸妫橀柛銉檮椤?
        if file is None:
            d = xlhome_dir('data/m2112kq5034/闁哄鏅滃褰掝敄濞嗘垶鍋?)
            # 闂佺懓鐏氶崕鎶藉春瀹€鍕剮妞ゆ棁鍋愮粔鍨槈閹炬剚鍎愭繛鎻掓健瀵剟鎮ч崼鐕佹Н闂傚倸鍊搁悺銊ャ€掗崼鏇炴闁规鍠楅悾閬嶆煛閸屾碍鐭楁繛?
            files = list(d.glob_files('*.csv'))
            files.sort(key=lambda f: f.mtime())
            file = files[-1]

        # 2 婵炴垶鎸搁敃锝囨閸洖妫橀柛銉檮椤?
        tab = self.tab
        tab.get('https://pay.weixin.qq.com/index.php/xphp/cbatchrefund/batch_refund#/pages/index/index')
        tab.wait(2)
        tab('tag:a@@title=婵炴垶鎸搁敃锝囨閸洖妫橀柛銉檮椤?).click.to_upload(file)
        tab.wait(2)
        tab('tag:a@@text():缂佺虎鍙庨崰鏍偩缂嶆class=btn btn-primary@@href=javascript:void(0);').click()
        tab.wait(2)

        # 3 婵犻潧顦介崑鍕疮閹惧灈鍋撻棃娑欘棤闁?        submit_started_at = pd.Timestamp.now()
        popup_result = self.婵犻潧顦介崑鍕疮閹惧灈鍋撻棃娑欘棤闁绘牗绮嶇粙澶嬫償閵娧冪煑闁荤姴娲ｉ懗鍫曟偉?tab)

        # 4 缂備焦绋戦ˇ顖滄閻斿憡浜ら柡鍐ｅ亾妞ゆ垶鐟ч埀顒傛嚀閺堫剟宕?        self.wait_batch_refund_completion(
            submit_started_at=submit_started_at,
            file_name=str(file),
            initial_popup_result=popup_result,
        )

    def __2_闁荤姳闄嶉崹鐟扮暦閻旂厧绀夐柣鏃囶嚙閸?self):
        pass

    @classmethod
    def _闂佹眹鍨婚崰鎰板垂?O闂佸搫娲︾€笛冪暦閼碱剛纾奸柛鏇ㄥ亝閸?cls, 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?:
        """闂佹眹鍨婚崰鎰板垂濮樺彉鐒婇柛鈩兩戝▓鍫曟倵?闂佸搫娲︾€笛冪暦閸欏鈻旈柛婵嗗閹界喐鎱ㄩ敐鍛€€闂佹眹鍔岀€氼厽鏅跺澶婂珘濠㈣泛锕ょ拋鏌ユ煠瀹曞洦娅曠紒顔肩Ч瀹?""
        # 闂佺懓鐏氶崕鎶藉春瀹€鍕闁逞屽墴瀵灚寰勭€ｎ偅顔嶉柣?闂佹眹鍔岀€氼亞绱為崨顖滅＞?
        zero_positions = [i for i, char in enumerate(闁荤姳闄嶉崹鐟扮暦閻旂厧鐭? if char == '0']

        if not zero_positions:
            return [闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楃紒?

        combinations = []
        # 闂佹椿娼块崝瀣礊閸涙潙绠肩€广儱瀚粙濠囨煟閵忋垹鏋戦柛銊﹀哺楠炲秹鍩€椤掑嫬瀚夊璺猴工鐠佹煡鏌ゅ畷鍥ㄦ珪婵炲牊鍨圭槐鎺楀礋椤愶絽鈧?
        # 0闁荤偞绋忛崝搴ㄥΦ濮橆厾鈹嶆繝闈涙搐閻︻喖鈽?0'闂?闁荤偞绋忛崝搴ㄥΦ濮樿泛鍗抽柟绋块鎼村﹤鈽?O'
        for i in range(2 ** len(zero_positions)):
            闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楃紒灞芥ist = list(闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?
            for j, pos in enumerate(zero_positions):
                if i & (1 << j):  # 濠碘槅鍋€閸嬫捇鏌＄仦璇插姢妞ゆ垵妾繛杈剧到缁夐潧危閹间礁瑙﹂柨鏃囧劵缁€?
                    闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楃紒灞芥ist[pos] = 'O'
            combinations.append(''.join(闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楃紒灞芥ist))

        return combinations

    @classmethod
    def 闂佹眹鍨婚崰鎰板垂濮樿泛纾规繛鍡樻尨閸嬫挻寰勬惔顔兼倕闂佸憡顨嗗ú妯肩博婵犳艾纭€?cls, 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?:
        闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?= str(闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?.lstrip("`'")
        闂佺锕ラ悷鈺呭焵椤掆偓椤︽娊顢樿ぐ鎺戠闁哄洨鍠撻鎼佹煕?= [闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楃紒?
        if '-' in 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?and '0' in 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?
            # 闁荤姳闄嶉崹鐟扮暦閻旂厧鐭楅柧姘€介崢顒勬煟閵娿儱顏柣锔瑰墲缁?闂備緡鍠涙慨銈咃耿娓氣偓瀹曪綁顢涘┑鍡楀箣闂佸搫瀚弸楣冩煥濞戞鐒锋い鏇ㄥ墴瀵顫濋銏╂喘闂佸湱顣介崑鎾绘煛閸繍妲归柛搴＄箻瀹曟ɑ鎷呴崘顏嗩啋闁荤偞绋戦張顒勬偂閿熺姴绠ラ柟鎯ф噽缁€澶愭煙閸偄鍔ら柛鈺佺灱缁參顢栫捄顭戜紘婵炴垶鎼╂禍婵嗩嚕瑜忛幖楣冨礃閸欏娈ら梻渚囧亐閸嬫捇鏌?
            闂佺锕ラ悷鈺呭焵椤掆偓椤︽娊顢樿ぐ鎺戠闁哄洨鍠撻鎼佹煕?= cls._闂佹眹鍨婚崰鎰板垂?O闂佸搫娲︾€笛冪暦閼碱剛纾奸柛鏇ㄥ亝閸?闁荤姳闄嶉崹鐟扮暦閻旂厧鐭?
        return 闂佺锕ラ悷鈺呭焵椤掆偓椤︽娊顢樿ぐ鎺戠闁哄洨鍠撻鎼佹煕?

    @classmethod
    def 婵炴潙鍚嬮敋閻庝絻灏欓幏瀣閻樿尙顦伴梺鍝勭Ф閸樠呮?cls, row):
        row2 = {}
        row2['闁荤姳闄嶉崹鐟扮暦閻旂厧绫嶉柕澶涢檮閸?] = row['datetime'].strftime('%Y%m') if row.get('datetime') else ''
        row2['閻庣敻鍋婇崰鏇熺┍婵犲洤缁╂い鏍ㄧ懅鐢盯鎮规担闈涚仼鐎规洜鍠栧畷?] = ('`' + row['flow_order']) if row.get('flow_order') else ''
        row2['闂佸摜鍠庡Λ娆撳春濞戞碍濯奸柕蹇曞Т缁€瀣煕?] = row['voucher_id'] if row.get('voucher_id') else ''
        row2['闁荤姳闄嶉崹鐟扮暦閻斿吋鐓傞柟杈惧瘜閺?] = float(row['money']) if row.get('money') else ''
        row2['閻庤鐡曠亸顏嗘崲閹存績鏋?] = str(row['refund']) if row.get('refund') else 0
        return row2
