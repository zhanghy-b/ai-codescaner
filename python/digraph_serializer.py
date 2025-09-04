import json
import argparse
import networkx as nx
import os
from typing import Dict, List, Set, Tuple
from joblib import dump, load
from pathlib import Path

class DiGraphSerializer:
    def __init__(self, verbose: bool = False):
        """初始化DiGraphSerializer类"""
        self.graph = nx.DiGraph()
        self.verbose = verbose

    def parse_clang_ast(self, ast_file_path: str) -> Dict:
        """
        解析Clang生成的AST JSON文件
        :param ast_file_path: AST文件路径
        :return: 解析后的AST字典
        """
        try:
            with open(ast_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {ast_file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"错误: 文件 {ast_file_path} 不是有效的JSON格式")
            return {}

    def extract_function_calls(self, ast_data: Dict) -> List[Tuple[dict, list]]:
        """
        从AST数据中提取函数调用关系
        :param ast_data: 解析后的AST字典
        :return: 函数调用关系列表，每个元素为(调用者, 被调用者)
        """
        nodes = {}
        edges = []
        if ast_data.get('functions'):
            functions = ast_data['functions']
            for func in functions:
                node_id = func.get('name')
                if not node_id:
                    continue
                nodes[node_id] = func

            for func in functions:
                call_funcs = func.get('called_functions')
                if not call_funcs or len(call_funcs) == 0:
                    continue
                for call_func in call_funcs:
                    edges.append((func['name'], call_func))
        return nodes, edges

    def build_digraph(self, nodes: dict[str, dict], edges: list[Tuple[str, str]]) -> nx.DiGraph:
        """
        根据函数调用关系构建DiGraph有向图
        :param function_calls: 函数调用关系列表
        :return: 构建好的DiGraph有向图
        """
        # 添加节点和边
        for key, val in nodes.items():
            self.graph.add_node(key, **val)
        for edge in edges:
            caller, callee = edge
            if caller not in self.graph.nodes:
                if self.verbose:
                    print(f"Warning: caller {caller} not found in graph nodes.")
            if callee not in self.graph.nodes:
                if self.verbose:
                    print(f"Warning: callee {callee} not found in graph nodes.")
            self.graph.add_edge(caller, callee)
        return self.graph

    def serialize_digraph(self, graph: nx.DiGraph, output_file_path: str) -> bool:
        """
        将DiGraph序列化到文件
        :param output_file_path: 输出文件路径
        :return: 是否序列化成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_file_path)), exist_ok=True)
            # 使用NetworkX的write_gpickle函数序列化图
            dump(graph, output_file_path)
            print(f"成功将有向图序列化到 {output_file_path}")
            return True
        except Exception as e:
            print(f"序列化有向图时出错: {str(e)}")
            return False

    def deserialize_digraph(self, input_file_path: str) -> nx.DiGraph or None:
        """
        从文件加载序列化的DiGraph
        :param input_file_path: 输入文件路径
        :return: 加载的DiGraph有向图，如果加载失败则返回None
        """
        try:
            if not os.path.exists(input_file_path):
                print(f"错误: 找不到文件 {input_file_path}")
                return None
            # 使用NetworkX的read_gpickle函数加载图
            # self.graph = nx.read_gpickle(input_file_path)
            graph = load(input_file_path)
            print(f"成功从 {input_file_path} 加载有向图")
            return graph
        except Exception as e:
            print(f"加载有向图时出错: {str(e)}")
            return None

    def analyze_function_calls(self, graph: nx.DiGraph = None) -> Dict:
        """
        分析函数调用关系
        :param graph: 要分析的有向图，如果为None则使用当前图
        :return: 分析结果字典
        """
        if graph is None:
            graph = self.graph

        analysis_results = {
            'total_nodes': len(graph.nodes()),
            'total_edges': len(graph.edges()),
            'dead_functions': [node for node in graph.nodes() if graph.in_degree(node) == 0],
            'most_called_functions': sorted(graph.in_degree(), key=lambda x: x[1], reverse=True)[:5],
            'calling_chain_example': self._find_calling_chain_example(graph)
        }
        return analysis_results

    def _find_calling_chain_example(self, graph: nx.DiGraph, max_length: int = 5) -> List[str]:
        """
        查找一个函数调用链示例
        :param graph: 有向图
        :param max_length: 最大链长度
        :return: 函数调用链列表
        """
        for node in graph.nodes():
            if graph.out_degree(node) > 0:
                chain = [node]
                current = node
                while len(chain) < max_length:
                    neighbors = list(graph.neighbors(current))
                    if not neighbors:
                        break
                    current = neighbors[0]
                    chain.append(current)
                if len(chain) > 1:
                    return chain
        return []

    def process_ast_and_serialize(self, ast_file_dir: str) -> bool:
        """
        处理AST文件并序列化生成的有向图
        :param ast_file_path: AST文件路径
        :param output_file_path: 输出文件路径
        :return: 是否成功
        """
        # 解析AST
        ast_file_paths = Path(ast_file_dir).rglob('*.ast.json')
        ast_results = []
        from utils import load_config
        for ast_file_path in ast_file_paths:
            ast_data = load_config(ast_file_path)
            if ast_data:
                ast_results.append(ast_data)

        from clang_ast_generator_threadpool import ThreadPoolClangASTGenerator
        config = load_config('./config/ast_config.json')
        generator = ThreadPoolClangASTGenerator(
            config 
        )

        import time
        start_time = time.time()
        function_call_graph = generator.build_function_call_graph(ast_results)
        succ = self.serialize_digraph(function_call_graph, os.path.join(config.get('ast_output', None), "nx_graph.gpickle"))
        end_time = time.time() 
        print(f"build function call graph time taken: {end_time - start_time:.2f} seconds")
        print(f"AST outputs saved to {config['ast_output']}")
        return succ

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Clang AST函数调用关系DiGraph序列化工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 从AST生成并序列化有向图的命令
    build_parser = subparsers.add_parser('build', help='从AST文件构建并序列化有向图')
    build_parser.add_argument('-a', '--ast-dir', required=True, help='Clang生成的AST JSON文件路径')
    build_parser.add_argument('-v', '--verbose', action='store_true', help='启用详细输出')

    # 加载序列化有向图并分析的命令
    analyze_parser = subparsers.add_parser('analyze', help='加载序列化有向图并分析')
    analyze_parser.add_argument('-i', '--input', required=True, help='序列化有向图文件路径')
    analyze_parser.add_argument('-v', '--verbose', action='store_true', help='启用详细输出')

    args = parser.parse_args()

    serializer = DiGraphSerializer(args.verbose)

    if args.command == 'build':
        # 从AST构建并序列化有向图
        serializer.process_ast_and_serialize(args.ast_dir)
    elif args.command == 'analyze':
        # 加载并分析有向图
        graph = serializer.deserialize_digraph(Path(args.input) / 'nx_graph.gpickle')
        if graph:
            analysis_results = serializer.analyze_function_calls(graph)
            print("\n分析结果:")
            print(f"总节点数: {analysis_results['total_nodes']}")
            print(f"总边数: {analysis_results['total_edges']}")
            print(f"没有被调用过的函数: {analysis_results['dead_functions']}")
            print("被调用最多的5个函数:")
            for func, count in analysis_results['most_called_functions']:
                print(f"  - {func}: {count}次")
            print(f"函数调用链示例: {' -> '.join(analysis_results['calling_chain_example'])}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()