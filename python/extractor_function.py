def extract_function_from_start_line(cpp_file_path: str, start_line: int):
    """
    从指定的起始行提取C++函数的完整实现

    参数:
        cpp_file_path: C++文件路径
        start_line: 函数起始行行号（从1开始计数）

    返回:
        包含完整函数代码的字符串，如果未找到则返回None
    """
    # 读取文件所有行
    with open(cpp_file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    # 检查起始行是否有效
    if start_line < 1 or start_line > len(lines):
        return None

    # 起始行索引（转换为0-based）
    start_idx = start_line - 1

    # 查找函数体开始的位置（第一个'{'）
    brace_start_idx = None
    for i in range(start_idx, len(lines)):
        if "{" in lines[i]:
            # 记录'{'在该行的位置
            brace_pos = lines[i].index("{")
            brace_start_idx = i
            break

    if brace_start_idx is None:
        return None  # 未找到函数体开始

    # 从第一个'{'开始计算括号匹配
    brace_count = 1
    end_idx = brace_start_idx
    current_pos = brace_pos + 1  # 从'{'后面开始

    # 遍历每行查找匹配的结束括号
    while end_idx < len(lines) and brace_count > 0:
        # 处理当前行剩余部分
        while current_pos < len(lines[end_idx]):
            char = lines[end_idx][current_pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # 找到函数结束位置
                    end_pos = current_pos
                    # 提取从起始行到结束行（包含）的所有内容
                    function_lines = lines[start_idx : end_idx + 1]
                    # 处理最后一行，只保留到结束括号
                    function_lines[-1] = function_lines[-1][: end_pos + 1]
                    return "".join(function_lines)
            current_pos += 1

        # 移动到下一行
        end_idx += 1
        current_pos = 0

    # 如果循环结束仍未找到匹配的括号，返回找到的部分
    return "".join(lines[start_idx : end_idx + 1])


"""
# 使用示例
if __name__ == "__main__":
    cpp_file = R"E:\GIP\GIP\Source\GMPPDFEngine\Picker\GMPPdfDragBoxSelector.cpp"  # 替换为你的C++文件路径
    start_line = 97  # 替换为函数起始行号

    function_code = extract_function_from_start_line(cpp_file, start_line)

    if function_code:
        print("提取到的函数实现：\n")
        print(function_code)
    else:
        print("未找到有效的函数实现")
"""
