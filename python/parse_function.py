import json
from typing import Dict, List


def parse_function_data(
    file_path: str,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    解析函数JSON数据，提取两类映射关系

    参数:
        file_path: 包含函数信息的JSON文件路径

    返回:
        两个字典的元组：
        1. 函数名与调用函数列表的映射
        2. 函数名与所在文件、行号的映射
    """

    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 初始化两个结果字典
    call_map: Dict[str, List[str]] = {}
    location_map: Dict[str, List[str]] = {}

    # 遍历所有函数
    for func in json_data.get("functions", []):
        func_name = func.get("name")
        if not func_name:
            continue

        # 1. 处理函数调用关系映射
        called_funcs = func.get("called_functions", [])
        call_map[func_name] = called_funcs

        # 2. 处理函数位置信息映射
        location = func.get("location", {})
        file_path = location.get("file", "")
        line_num = str(location.get("line", ""))  # 转换为字符串便于统一存储
        location_map[func_name] = [file_path, line_num]

    return call_map, location_map


"""
# 使用示例
if __name__ == "__main__":
    # 提取数据
    calls, locations = parse_function_data(R"E:\AI Course\parseCov\data\callgraph.json")

    # 打印结果
    print("1. 函数调用关系映射:")
    for func, called in calls.items():
        print(f"{func}: {called}")

    print("\n2. 函数位置信息映射:")
    for func, loc in locations.items():
        print(f"{func}: 文件={loc[0]}, 行号={loc[1]}")
"""
