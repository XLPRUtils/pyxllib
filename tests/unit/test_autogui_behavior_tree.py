from pyxllib.autogui.behavior_tree import (
    ActionPlanner,
    CloseActionPlanner,
    CurView,
    DbView,
    MatchRole,
    Runtime,
    SceneNavigator,
    SceneRecognizer,
    SceneScorer,
    Shape,
    ShapeMatchPlanner,
    View,
    normalize_match_role,
)


def _image(title: str, filename: str, shapes: list[dict] | None = None) -> dict:
    return {
        "type": "image",
        "title": title,
        "filename": filename,
        "width": 900,
        "height": 1600,
        "shapes": shapes or [],
    }


def test_behavior_tree_view_action_match_and_navigation_basics():
    jump = {"id": "jump", "kind": "rect", "title": "去二", "sceneJumpTarget": "#2"}
    button = {"id": "btn", "kind": "rect", "title": "按钮", "x": 0.5, "y": 0.9, "w": 0.1, "h": 0.06}
    tree = [_image("一", "0001.png", [jump, button]), _image("二", "0002.png", [])]

    assert normalize_match_role("定") is MatchRole.decisive
    assert View(tree[0]).get_shape("按钮").box() == {
        "name": "按钮",
        "x": 450.0,
        "y": 1440.0,
        "w": 90.0,
        "h": 96.0,
    }
    assert ActionPlanner().shape_center(tree[0], button) == (495.0, 1488.0)
    assert ShapeMatchPlanner().runtime_match_payload_flags({"floating": True, "ocrMatchRole": "off"})["scan"] is True
    assert ShapeMatchPlanner().match_conditions({"imageMatchRole": "定", "ocrMatchRole": "定", "ocrText": "邮"}) == ["image", "ocr"]
    assert [edge["shape"]["id"] for edge in SceneNavigator(tree).find_scene_route(1, 2)] == ["jump"]


def test_behavior_tree_runtime_wait_click_is_generic_shape_action():
    image = _image("菜单", "0035.png", [{"id": "mail", "kind": "rect", "title": "邮件"}])
    shape = View(image).get_shape("邮件")
    calls = []

    class FakeRuntime(Runtime):
        def __init__(self):
            self.count = 0

        def get_cur_view(self, update: bool = False):
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return []

        def goto_view(self, view):
            raise NotImplementedError

        def match_shape(self, candidate):
            self.count += 1
            return self.count >= 2

        def click_shape(self, view, candidate):
            calls.append((view.title, candidate.title))
            return "clicked"

    waiter = shape.wait_click(FakeRuntime())

    assert next(waiter) == 1
    try:
        next(waiter)
    except StopIteration as exc:
        assert exc.value == "clicked"
    else:
        raise AssertionError("wait_click should finish after shape matches")
    assert calls == [("菜单", "邮件")]


def test_behavior_tree_close_action_planner_picks_safe_close_shape():
    shapes = [
        {"id": "confirm", "kind": "rect", "title": "确定"},
        {"id": "exit", "kind": "rect", "title": "任意", "sceneJumpTarget": "-1"},
        {"id": "blank", "kind": "rect", "title": "空白"},
    ]

    assert CloseActionPlanner(title_priorities=("空白", "关闭")).choose_close_shape(
        shapes,
        include_independent_exit=True,
    )["id"] == "blank"
    assert CloseActionPlanner(title_priorities=("关闭",)).choose_close_shape(
        shapes,
        include_independent_exit=True,
    )["id"] == "exit"


def test_behavior_tree_view_close_delegates_action_to_runtime():
    image = _image("提示", "0047.png", [
        {"id": "confirm", "kind": "rect", "title": "确认"},
        {"id": "blank", "kind": "rect", "title": "空白"},
    ])
    calls = []

    class FakeRuntime:
        def click_shape(self, view, shape):
            calls.append((view.id, shape.title))
            return "clicked"

    assert View(image).close(FakeRuntime()) == "clicked"
    assert calls == [(47, "空白")]


def test_behavior_tree_dbview_and_curview_separate_static_and_runtime_state():
    image = _image("提示", "0047.png", [
        {"id": "blank", "kind": "rect", "title": "空白"},
    ])
    dbview = DbView(image)
    curview = CurView(view=dbview, score=91.5, frame={"source": "live"})
    calls = []

    class FakeRuntime:
        def click_shape(self, view, shape):
            calls.append((type(view), view.id, shape.title))
            return "closed"

    assert isinstance(dbview, View)
    assert dbview.get_shape("空白").parent_view is dbview
    assert curview.id == 47
    assert curview.title == "提示"
    assert curview.raw is image
    assert curview.close(FakeRuntime()) == "closed"
    assert calls == [(DbView, 47, "空白")]


def test_behavior_tree_runtime_resolves_curview_to_static_view():
    image = _image("场景", "0121.png", [])
    dbview = DbView(image)

    class FakeRuntime(Runtime):
        def get_cur_view(self, update: bool = False):
            return CurView(dbview, score=88)

        def get_views(self, group: str = "", recursive: bool = False):
            return [dbview]

        def goto_view(self, view):
            raise NotImplementedError

        def match_shape(self, candidate):
            return False

        def click_shape(self, view, candidate):
            raise NotImplementedError

    runtime = FakeRuntime()

    assert runtime.get_view(CurView(dbview, score=88)) is dbview
    assert runtime.get_view(121) is dbview


def test_behavior_tree_runtime_centered_view_and_shape_matching():
    required = {"id": "required", "title": "必须", "isSceneIdentity": True, "sceneIdentityRole": "required"}
    decisive = {"id": "decisive", "title": "任一", "sceneIdentityRole": "decisive"}
    view = View(_image("世界", "0034.png", [required, decisive]))
    calls = []

    class FakeRuntime(Runtime):
        def __init__(self, matched_ids):
            self.matched_ids = set(matched_ids)

        def get_cur_view(self, update: bool = False):
            return "cur"

        def get_views(self, group: str = "", recursive: bool = False):
            calls.append((group, recursive))
            return [view]

        def match_shape(self, shape):
            return shape.raw.get("id") in self.matched_ids

        def click_shape(self, view, shape):
            return shape.title

        def goto_view(self, view):
            return view

    assert view.get_shape("必须").is_match(FakeRuntime({"required"})) is True
    assert view.is_match(FakeRuntime({"decisive"})) is True
    assert view.is_match(FakeRuntime({"required"})) is True
    assert view.is_match(FakeRuntime(set())) is False
    assert FakeRuntime({"required"}).find_view("日常") is view
    assert calls[-1] == ("日常", False)


def test_behavior_tree_find_view_empty_group_ignores_nested_identity_shapes():
    nested_identity = {
        "id": "panel",
        "title": "面板",
        "children": [
            {"id": "nested", "title": "子标识", "sceneIdentityRole": "decisive"},
        ],
    }
    view = View(_image("世界", "0034.png", [nested_identity]))

    class FakeRuntime(Runtime):
        def get_cur_view(self, update: bool = False):
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return [view]

        def match_shape(self, shape):
            return shape.raw.get("id") == "nested"

        def click_shape(self, view, shape):
            return None

        def goto_view(self, view):
            return None

    runtime = FakeRuntime()

    assert view.is_match(runtime) is True
    assert view.is_match(runtime, include_descendants=False) is False
    assert runtime.find_view("") is None
    assert runtime.find_view("任意分组") is view


def test_behavior_tree_shape_load_delegates_scroll_window_to_runtime():
    image = _image("邮件", "0121.png", [
        {"id": "list", "title": "邮件清单2", "x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5, "contentDirection": "down"}
    ])
    shape = View(image).get_shape("邮件清单2")
    calls = []

    class FakeRuntime(Runtime):
        def __init__(self):
            self.attrs = {}
            self.signatures = iter(["before", "after"])

        def get_cur_view(self, update: bool = False):
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return []

        def goto_view(self, view):
            return None

        def match_shape(self, shape):
            return False

        def click_shape(self, view, shape):
            return None

        def drag_shape_content(self, shape, *, ratio=0.5, duration=1.5):
            calls.append((shape.title, ratio, duration))

        def shape_load_signature(self, shape):
            return next(self.signatures)

    runtime = FakeRuntime()

    assert shape is not None
    assert list(shape.load(runtime, ratio=0.6, duration=2.0)) == [1]
    assert calls == [("邮件清单2", 0.6, 2.0)]
    assert runtime.attrs["load_new"] is True


def test_behavior_tree_shape_load_without_content_direction_does_not_drag():
    shape = Shape({"title": "按钮"})

    class FakeRuntime(Runtime):
        def __init__(self):
            self.attrs = {}

        def get_cur_view(self, update: bool = False):
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return []

        def goto_view(self, view):
            return None

        def match_shape(self, shape):
            return False

        def click_shape(self, view, shape):
            return None

        def drag_shape_content(self, shape, *, ratio=0.5, duration=1.5):
            raise AssertionError("should not drag")

    runtime = FakeRuntime()

    assert list(shape.load(runtime)) == []
    assert runtime.attrs["load_new"] is False


def test_behavior_tree_runtime_wait_view_yields_until_match():
    view = View(_image("目标", "0121.png", [{"id": "identity", "title": "目标", "isSceneIdentity": True}]))

    class FakeRuntime(Runtime):
        def __init__(self):
            self.attempts = 0

        def get_cur_view(self, update: bool = False):
            self.attempts += 1
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return [view]

        def goto_view(self, view):
            return None

        def match_shape(self, shape):
            return self.attempts >= 2

        def click_shape(self, view, shape):
            return None

    runtime = FakeRuntime()
    waiter = runtime.wait_view(view)

    assert next(waiter) == 1
    try:
        next(waiter)
    except StopIteration as exc:
        assert exc.value is view
    else:
        raise AssertionError("wait_view should stop when the target view matches")


def test_behavior_tree_runtime_wait_view_timeout(monkeypatch):
    view = View(_image("目标", "0121.png", [{"id": "identity", "title": "目标", "isSceneIdentity": True}]))
    monotonic_values = iter([0.0, 0.5, 1.2])

    class FakeRuntime(Runtime):
        def get_cur_view(self, update: bool = False):
            return None

        def get_views(self, group: str = "", recursive: bool = False):
            return [view]

        def goto_view(self, view):
            return None

        def match_shape(self, shape):
            return False

        def click_shape(self, view, shape):
            return None

    monkeypatch.setattr("pyxllib.autogui.runtime.time.monotonic", lambda: next(monotonic_values))
    waiter = FakeRuntime().wait_view(view, timeout=1.0)

    assert next(waiter) == 1
    try:
        next(waiter)
    except TimeoutError as exc:
        assert "等待 view 超时" in str(exc)
    else:
        raise AssertionError("wait_view should raise TimeoutError after timeout")


def test_behavior_tree_action_planner_drag_shape_content_points_follow_direction():
    image = {"width": 100, "height": 200}
    shape = {"x": 0.1, "y": 0.2, "w": 0.4, "h": 0.5}

    assert ActionPlanner().drag_shape_content_points(image, shape, direction="down", ratio=0.5) == (30.0, 115.0, 30.0, 65.0)
    assert ActionPlanner().drag_shape_content_points(image, shape, direction="up", ratio=0.5) == (30.0, 65.0, 30.0, 115.0)


def test_behavior_tree_scene_recognizer_is_runtime_agnostic():
    ctx = {"images": {1: {"title": "#1"}, 2: {"title": "#2"}}}
    recognizer = SceneRecognizer(
        score_image=lambda _ctx, image, _frame: 90 if image["title"] == "#2" else 70,
        threshold_for_scene_id=lambda _scene_id: 80,
    )

    assert recognizer.identify_scene_number(ctx, "frame") == (2, 90.0)


def test_behavior_tree_scene_recognizer_prefers_order_on_preferred_tie():
    ctx = {"images": {34: {"title": "#34"}, 35: {"title": "#35"}}}
    recognizer = SceneRecognizer(
        score_image=lambda *_args: 100.0,
        threshold_for_scene_id=lambda _scene_id: 80,
    )

    assert recognizer.identify_scene_number(ctx, "frame", preferred_scene_ids=[35, 34]) == (35, 100.0)
    assert recognizer.identify_scene_number(ctx, "frame") == (34, 100.0)


def test_behavior_tree_scene_scorer_combines_scene_identity_roles():
    image = _image("世界", "0034.png", [
        {"id": "weak", "title": "弱标识", "isSceneIdentity": True},
        {"id": "strong", "title": "强标识", "isSceneIdentity": True},
    ])
    scorer = SceneScorer(
        shape_score=lambda _ctx, _image, shape, _frame: 20 if shape["id"] == "weak" else 95,
        shape_ocr_score=lambda *_args: 0,
        threshold=80,
    )

    assert scorer.scene_score({}, image, "frame") == 95


def test_behavior_tree_scene_scorer_enforces_required_ocr_role():
    shape = {
        "id": "identity",
        "title": "离开",
        "isSceneIdentity": True,
        "imageMatchRole": "optional",
        "ocrEnabled": True,
        "ocrText": "离开",
        "ocrMatchRole": "required",
    }
    image = _image("离开场景", "0086.png", [shape])
    scorer = SceneScorer(
        shape_score=lambda *_args: 99,
        shape_ocr_score=lambda *_args: 0,
        threshold=80,
    )

    assert scorer.scene_score({}, image, "frame") == 0


def test_behavior_tree_scene_navigator_builds_route_candidate_ids():
    edge12 = {"id": "edge12", "title": "去二", "sceneJumpTarget": "#2"}
    edge23 = {"id": "edge23", "title": "去三", "sceneJumpTarget": "#3"}
    confirm = {"id": "confirm", "title": "确认"}
    tree = [
        _image("一", "0001.png", [edge12]),
        _image("二", "0002.png", [edge23]),
        _image("三", "0003.png", []),
        _image("离开场景", "0086.png", [confirm]),
    ]
    navigator = SceneNavigator(tree)

    confirmation_ids = navigator.confirmation_scene_ids(lambda image: image.get("title") == "离开场景")

    assert confirmation_ids == [86]
    assert navigator.route_candidate_ids(3, confirmation_scene_ids=confirmation_ids) == [3, 86, 2, 1]


def test_behavior_tree_public_exports_are_explicit():
    from pyxllib.autogui import behavior_tree as bt

    expected = {
        "ActionPlanner",
        "CloseActionPlanner",
        "CurView",
        "DbView",
        "MatchRole",
        "Runtime",
        "SceneNavigator",
        "SceneRecognizer",
        "SceneScorer",
        "Shape",
        "ShapeMatchPlanner",
        "View",
        "normalize_match_role",
    }
    namespace = {}

    exec("from pyxllib.autogui.behavior_tree import *", namespace)

    assert expected <= set(bt.__all__)
    assert namespace["CurView"] is CurView
    assert namespace["DbView"] is DbView
    assert namespace["View"] is View
    assert namespace["SceneNavigator"] is SceneNavigator


def test_autogui_package_reexports_behavior_tree_foundation():
    from pyxllib.autogui import CurView as PackageCurView
    from pyxllib.autogui import DbView as PackageDbView
    from pyxllib.autogui import MatchRole as PackageMatchRole
    from pyxllib.autogui import Runtime as PackageRuntime
    from pyxllib.autogui import SceneNavigator as PackageSceneNavigator
    from pyxllib.autogui import View as PackageView

    assert PackageCurView is CurView
    assert PackageDbView is DbView
    assert PackageMatchRole is MatchRole
    assert PackageRuntime is Runtime
    assert PackageSceneNavigator is SceneNavigator
    assert PackageView is View
