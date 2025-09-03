import chardet
from pathlib import Path
import clang.cindex
import networkx as nx
import re, json

file_splict_char = '#$%'
graph_node_with_filename = False

qt_funcs = ['::metaObject()', '::tr(const char *, const char *, int)', '::trUtf8(const char *, const char *, int)',
            '::qt_metacall(QMetaObject::Call, int, void **)', '::qt_metacast(const char *)', 
            '::qt_static_metacall(QObject *, QMetaObject::Call, int, void **)']

def detect_file_encoding(file_path):
    # 读取文件的前几KB用于检测（避免读取整个大文件，提高效率）
    with open(file_path, 'rb') as f:
        # 读取10KB数据（可根据实际情况调整）
        raw_data = f.read(1024 * 10)
    
    # 检测编码
    result = chardet.detect(raw_data)
    # 结果包含编码格式和可信度（0-1）
    encoding = result['encoding']
    confidence = result['confidence']
    return encoding, confidence

def is_external_file(file_path : str):
    """
    检查文件是否位于外部目录中
    :param file_path: 文件路径
    :return: 如果文件位于外部目录中则返回True，否则返回False
    """
    external_dirs = ['Externals', 'Program Files (x86)', 'Windows Kits']
    for external_dir in external_dirs:
        if external_dir.lower() in file_path.lower():
            return True
    return False

def transfer_file_path(file_path: str, src_repo_root: str, dest_repo_root: str) -> str:
    try:
        rel_path = Path(file_path).resolve().relative_to(Path(src_repo_root).resolve())
        return str(Path(dest_repo_root).resolve() / rel_path)
    except Exception as e:
        return file_path

def transfer_cov_func_name(func_name: str) -> str:
    if func_name and func_name.find('(') != -1 and func_name.find(')') != -1:
        arg_str = func_name[func_name.find('(')+1: func_name.find(')')]
        args = arg_str.split(',')
        for i in range(len(args)):
            arg = args[i].strip()
            one_args = arg.split(' ')
            arg = ''
            for one_arg in one_args:
                if one_arg == 'const':
                    arg += one_arg + ' '
                else:
                    arg += one_arg
            args[i] = arg.strip()
        arg_str1 = ','.join(args)
        func_name = func_name.replace(arg_str, arg_str1)
    else:
        print(f"警告：函数名格式异常，无法处理参数列表：{func_name}")
    return func_name

def remove_qt_moc_func(unreachable_functions: set) -> set:
    unreachable_functions = [func for func in unreachable_functions if not any(transfer_cov_func_name(qt_func) in func for qt_func in qt_funcs)]
    return unreachable_functions

def remove_deconstructor_func(unreachable_functions: set) -> set:
    pattern = r"~[A-Za-z_][A-Za-z0-9_]*\s*\(\)"
    unreachable_functions = [func for func in unreachable_functions if not re.search(pattern, func)]
    return unreachable_functions

def extract_rela_path(file_path):
    """
    提取文件路径并去掉扩展名
    :param file_path: 文件路径
    :return: 去掉扩展名的文件路径
    """
    path = Path(file_path)
    if path.suffix in ['.h', '.hpp', '.hh', '.hxx'] and 'include' in str(path.parent).lower():
        result = str(path.parent / path.stem)
        start_index = result.lower().index('include') + len('include/')
        result = result[start_index:]
        return result
    if path.suffix in ['.cpp'] and 'source' in str(path.parent).lower():
        result = str(path.parent / path.stem)
        start_index = result.lower().index('source') + len('source/')
        result = result[start_index:]
        return result
    return path.parent / path.stem

def remove_clang_unsupport_cmds(compile_args):
    compile_args = [arg for arg in compile_args if '/we' not in arg]  # Exclude the file name
    compile_args = [arg for arg in compile_args if '-external:W0' not in arg]  # Exclude the file name
    compile_args = [arg for arg in compile_args if '/RTC1' not in arg]  # Exclude the file name
    compile_args = [arg for arg in compile_args if '--' not in arg]  # Exclude the file name
    compile_args = ['-isystem' if arg == '-external:I' else arg for arg in compile_args]  # Exclude the file name
    compile_args = compile_args[0: len(compile_args) - 1]  # Skip the first argument which is the compiler name
    # 启用微软语法扩展（如 __declspec、__stdcall）
    compile_args.insert(1, '-fms-extensions')       
    # 模拟 MSVC 行为，兼容 Windows 头文件
    compile_args.insert(1, '-fms-compatibility')              
    compile_args.insert(1, '-DWIN32_LEAN_AND_MEAN')              
    compile_args.insert(1, '-std=c++14')
    return compile_args

def generator_func_node_id(func_node : dict) -> str:
    """
    生成函数节点的唯一标识符
    :param func_node: 函数节点字典，包含函数名称和文件路径等信息
    :return: 唯一标识符字符串
    """
    func_name = func_node.get('name', 'unknown_function')
    func_name = transfer_cov_func_name(func_name)
    if graph_node_with_filename:
        if func_node.get('location') and func_node['location'].get('file'):
            location = func_node['location']
            file_path = Path(location['file']).resolve()
            file_rela_path_without_suffix = extract_rela_path(file_path)
            return f"{file_rela_path_without_suffix}{file_splict_char}{func_name}"
        return None
    else:
        return func_name
    # 去掉文件后缀

def parse_cursor_func_node(func : clang.cindex.Cursor) -> str:
    func_node_name = func.displayname
    # for class methods, prepend class name to function name
    if func.kind == clang.cindex.CursorKind.CXX_METHOD:
        parent = func.semantic_parent
        if parent and parent.kind in [clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL]:
            func_node_name = f"{parent.spelling}::{func_node_name}"
    return func_node_name

def parse_cursor_call_func_node(call_func: clang.cindex.Cursor) -> str:
    """
    解析被调用函数的名称
    :param call_func: 被调用函数的 Clang AST 节点
    :return: 被调用函数的名称字符串
    """
    reference = call_func.referenced
    if reference:
        file = reference.location.file.name
        if is_external_file(file):
            return None
        if reference.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD, 
                             clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR]:
            key = parse_cursor_func_node(reference)
            if graph_node_with_filename:
                file = Path(file).resolve()
                file_rela_without_suffix = extract_rela_path(file)
                cxx_fun = f'{file_rela_without_suffix}{file_splict_char}{key}'
                return transfer_cov_func_name(cxx_fun)
            else:
                return transfer_cov_func_name(key)
        else:
            return None
    # else:
    #     children = call_func.get_children()
    #     for c in children:
    #         c1 = list(c.get_children())[0]
    #         if c1.kind == clang.cindex.CursorKind.CALL_EXPR: 
    #             return parse_cursor_call_func_node(c1)
    return None


def parse_cursor_lamda_func_node(lambda_func: clang.cindex.Cursor) -> str:
    """
    解析Lambda函数的名称
    :param lambda_func: Lambda函数的 Clang AST 节点
    :return: Lambda函数调用的函数名称列表 
    """
    funcs = []
    if lambda_func.kind == clang.cindex.CursorKind.LAMBDA_EXPR:
        for child in lambda_func.get_children():
            if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
                for grand_child in child.get_children():
                    if grand_child.kind == clang.cindex.CursorKind.CALL_EXPR:
                        func_name = parse_cursor_call_func_node(grand_child)
                        funcs.append(func_name)
                
    return funcs
        
def parse_qt_signal_slot_func(obj: clang.cindex.Cursor, singal: clang.cindex.Cursor) -> str:
    """
    解析Qt信号槽函数的名称
    :param obj: 信号或槽所在的对象的 Clang AST 节点
    :param singal: 信号或槽的 Clang AST 节点
    :return: 信号或槽函数的名称字符串
    """
    if singal.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
        singal = list(singal.get_children())[0]
    if singal.kind == clang.cindex.CursorKind.STRING_LITERAL:
        displayname = singal.displayname.replace('"','')
        displayname = displayname[1:]
        if obj.kind in [clang.cindex.CursorKind.CXX_THIS_EXPR, clang.cindex.CursorKind.MEMBER_REF_EXPR]:
            class_name = obj.type.spelling.replace('*', '').strip()
            return [f"{class_name}::{displayname}"]
    if singal.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
        reference = singal.referenced
        func_name = parse_cursor_func_node(reference) 
        return [func_name]
    if singal.kind == clang.cindex.CursorKind.LAMBDA_EXPR:
        funcs = parse_cursor_lamda_func_node(singal)
        return funcs
    return None

def has_key(nodes: set, key: str) -> bool:
    """
    检查集合中是否存在指定的键
    :param nodes: 节点集合
    :param key: 要检查的键
    :return: 如果存在则返回True，否则返回False
    """
    for node in nodes:
        if key in node:
            return True, node
    return False, None

def load_config(config_file):
    """
    加载配置文件
    :param config_file: 配置文件路径
    :return: 配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def dump(json_data, output_file):
    if json_data and len(json_data) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            # 写入JSON数据，使用indent参数美化输出，ensure_ascii=False确保中文正常显示
            json.dump(json_data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # 测试 transfer_file_path 函数
    # func_name = 'IGMPPDFViewDrawer::inputCmdLine(const QString &)'
    # func_name = transfer_cov_func_name(func_name)
    # print(func_name)
    # d = {'key1':100, 'key2':200, 'key3':300}
    # print(d)
    # del d['key2']
    # print(d)
    except_graph_funcs = ['main']
    except_graph_funcs_file = Path('./result') / 'except_graph_funcs.json'
    dump(except_graph_funcs, except_graph_funcs_file)