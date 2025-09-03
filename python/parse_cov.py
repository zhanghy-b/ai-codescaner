import xml.etree.ElementTree as ET
import sys
import json  # 导入json模块用于处理JSON数据
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils import load_config

def load_filter_list(filter_file):
    """加载过滤文件列表，每行一个路径"""
    try:
        with open(filter_file, "r", encoding="utf-8") as f:
            # 读取所有行，去除空白和换行符，过滤空行
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"读取过滤文件时出错: {str(e)}", file=sys.stderr)
        sys.exit(1)


def find_coverage_reach_funcs(xml_file, filter_list):
    """
    解析XML文件，查找所有fn_cov和cd_cov都为0的src元素，并返回其路径

    参数:
        xml_file: XML文件路径

    返回:
        符合条件的src元素路径列表
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 验证根元素是否为BullseyeCoverage
        if root.tag != "{http://www.bullseye.com/covxml}BullseyeCoverage":
            raise ValueError("XML文件的根元素必须是BullseyeCoverage")

        result = []

        # 递归遍历XML元素
        def traverse(element, current_path):
            # 处理src元素
            if element.tag == "{http://www.bullseye.com/covxml}fn":
                # 获取属性值，默认为0
                fn_cov = element.get("fn_cov", "0")
                cd_cov = element.get("cd_cov", "0")
                src_name = element.get("name", "")

                # 检查是否两个覆盖率都为0
                # if fn_cov == "0" and cd_cov == "0":
                if fn_cov != "0":
                    # 构建完整路径
                    full_path = (
                        f"{current_path}/{src_name}" if current_path else src_name
                    )
                    result.append(full_path)
                return

            # 处理folder元素，继续遍历子元素
            if (
                element.tag == "{http://www.bullseye.com/covxml}folder"
                or element.tag == "{http://www.bullseye.com/covxml}src"
            ):
                folder_name = element.get("name", "")
                # 检查当前文件夹是否在过滤列表中
                if folder_name in filter_list:
                    return
                # 更新当前路径
                new_path = (
                    f"{current_path}/{folder_name}" if current_path else folder_name
                )

                # 递归处理所有子元素
                for child in element:
                    traverse(child, new_path)

        # 从根元素开始遍历，初始路径为空
        for child in root:
            traverse(child, "")

        return result

    except Exception as e:
        print(f"解析XML文件时出错: {str(e)}", file=sys.stderr)
        return []
    
def parse_cov_filter_func(cov_config: Dict[str, Any])->List[str]:
    filter_list = cov_config["filter_dirs"]

    # 遍历覆盖率为0的文件
    coverage_reach_funcs = find_coverage_reach_funcs(cov_config['cov_xml'], filter_list)
    # 将结果写入JSON文件
    try:
        cov_filter_ouput_file = cov_config['cov_filter_output_file']
        if Path(cov_filter_ouput_file).parent.exists():
            with open(cov_filter_ouput_file, "w", encoding="utf-8") as f:
                # 写入JSON数据，使用indent参数美化输出，ensure_ascii=False确保中文正常显示
                json.dump(coverage_reach_funcs, f, ensure_ascii=False, indent=4)
            print(f"成功将{len(coverage_reach_funcs)}个结果写入到{cov_filter_ouput_file}")
        return coverage_reach_funcs
    except Exception as e:
        print(f"写入JSON文件时出错: {str(e)}", file=sys.stderr)
        return []

def modify_func_name(zero_cov_srcs: List) -> Optional[str]:
    results = []
    for src in zero_cov_srcs:
        func_name = src.split('/')[-1]
        if func_name.find('::') != -1:
            args = func_name.split('::')
            if args[-1][0] == '~':
                results.append(args[-1])
            elif args[-1].find(f'{args[-2]}'+'(') != -1:
                results.append(args[-1])
            else:
                func_name = f'{args[-2]}::{args[-1]}'
                if func_name.find(') const' ) != -1:
                    func_name = func_name.replace(') const', ')')
                results.append(func_name)
        else:
            results.append(func_name)
    return results

def search_cov_reachable_funcs(cov_config: Dict[str, Any]) -> List[str]:
    coverage_reach_funcs = parse_cov_filter_func(cov_config)
    extract_func_results = modify_func_name(coverage_reach_funcs)
    return extract_func_results

if __name__ == "__main__":
    # cov_config = load_config('config/cov_config.json')
    # coverage_reach_funcs = parse_cov_filter_func(cov_config)

    coverage_reach_srcs = load_config('./result/filtered_cov.json')
    extract_func_results = modify_func_name(coverage_reach_srcs)
    from utils import dump
    dump(extract_func_results, Path('./result') / 'modify_cov_funcs.json')
    
