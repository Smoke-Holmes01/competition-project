from app.intent import is_image_specific_question, should_call_match_tool


def test_simple_what_is_this_does_not_call_tool():
    assert should_call_match_tool("这张图是什么？", has_image=True) is False
    assert should_call_match_tool("你能告诉我这张图片有什么内容吗", has_image=True) is False


def test_dynasty_question_calls_tool_with_image():
    assert should_call_match_tool("这个斗拱是哪个朝代的？", has_image=True) is True


def test_no_image_never_calls_tool():
    assert should_call_match_tool("这个斗拱是哪个朝代的？", has_image=False) is False


def test_image_specific_question_detection():
    assert is_image_specific_question("这张图是什么？") is True
