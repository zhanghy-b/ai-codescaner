from code_invalidity_scanner import CodeInvalidityScanner
from utils import load_config
import argparse
from pathlib import Path
from parse_cov import search_cov_reachable_funcs

def load_ast_results(ast_output_dir):
    ast_results = []
    ast_file_paths = Path(ast_output_dir).rglob('*.ast.json')
    for ast_file_path in ast_file_paths:
        ast_data = load_config(ast_file_path)
        if ast_data:
            ast_results.append(ast_data)
    return ast_results

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Clang AST 有向图构建与分析工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    # 从AST生成并序列化有向图的命令
    build_parser = subparsers.add_parser('scan&analyze', help='从AST文件构建并序列化有向图')
    # 加载序列化有向图并分析的命令
    analyze_parser = subparsers.add_parser('analyze', help='加载序列化有向图并分析')
    args = parser.parse_args()

    ast_config = load_config('config/ast_config.json')
    scanner_config = load_config('config/scanner_config.json')
    analyze_config = load_config('config/analyze_config.json')
    cov_config = load_config('config/cov_config.json')

    cov_reachable_funcs = search_cov_reachable_funcs(cov_config)
    # cov_reachable_funcs.clear()
    # cov_reachable_funcs.append('main(int,char**)')
    scanner = CodeInvalidityScanner(ast_config, scanner_config, cov_reachable_funcs)

    if args.command == 'scan&analyze':
        # 仅分析已有的有向图
        # 创建扫描器实例
        # 运行分析
        report = scanner.run()
        print(f"扫描结果：\r\n{report}")
    elif args.command == 'analyze':
        ast_results = load_ast_results(analyze_config['ast_output'])
        from digraph_serializer import DiGraphSerializer
        serializer = DiGraphSerializer(True)
        func_graph = serializer.deserialize_digraph(analyze_config['func_graph_file'])
        report = scanner.analyze(ast_results, func_graph)
        print(f"分析结果：\r\n{report}")