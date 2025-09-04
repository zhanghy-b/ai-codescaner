#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLang AST Generator for C++ Code Repository

This script parses C++ files in a repository using libclang and generates AST (Abstract Syntax Tree)
for further analysis. Supports loading compilation database (compile_commands.json) to resolve
t头文件引用路径问题.
"""

import os
import argparse
from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from utils import *
from digraph_serializer import DiGraphSerializer
import threading
import concurrent.futures
from collections import defaultdict

class ThreadPoolClangASTGenerator:
    def __init__(self, kwargs : Dict[str, Any]):
        """
        Initialize the AST generator
        
        Args:
            repo_path: Path to the C++ code repository
            output_dir: Directory to save AST outputs
            verbose: Enable verbose logging
            compile_db_path: Path to the compilation database (compile_commands.json)
        """

        self.repo_root = kwargs.get("repo_root", "")
        self.repo_dest_root = kwargs.get("repo_dest_root", "")
        self.codes = kwargs.get("codes", None)
        self.output_dir = kwargs.get("ast_output", None)
        self.compile_db_path = kwargs.get("compile_commands_db", None)
        self.verbose = kwargs.get("verbose", False)
        self.ast = kwargs.get("parse_ast", False)
        self.body = kwargs.get("parse_func_body", False)
        self.max_workers = kwargs.get("workers", 1)

        self._results_lock = threading.Lock()
        self._errors_lock = threading.Lock()
        self._processed_files = 0
        self._total_files = 0
        self._thread_local = threading.local()
        
        # Initialize Clang index
        try:
            clang.cindex.Config.set_library_file('./clang+llvm-20.1.8-x86_64-pc-windows-msvc/bin/libclang.dll')
            self.index = clang.cindex.Index.create()
        except Exception as e:
            print(f"Failed to initialize Clang index: {e}")
            raise
        
        # Load compilation database if specified
        if self.compile_db_path:
            self._load_compilation_database(self.compile_db_path)
        
        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def find_cpp_files(self) -> List[str]:
        """
        Find all C++ files in the repository
        
        Returns:
            List of C++ file paths
        """
        # cpp_extensions = ['.cpp', '.cxx', '.cc', '.h', '.hpp', '.hxx']
        cpp_extensions = ['.cpp', '.cxx']
        cpp_files = []
        for repo_path in self.codes: 
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if any(file.endswith(ext) for ext in cpp_extensions):
                        cpp_files.append(os.path.join(root, file))
        
        if self.verbose:
            print(f"Found {len(cpp_files)} C++ files in repository")
        
        return cpp_files

    def _get_thread_index(self):
        """Get or create a Clang index for the current thread"""
        if not hasattr(self._thread_local, 'index'):
            self._thread_local.index = clang.cindex.Index.create()
        return self._thread_local.index

    def _load_compilation_database(self, compile_db_path: str) -> None:
        """
        Load the compilation database
        
        Args:
            compile_db_path: Path to the compilation database (compile_commands.json)
        """
        try:
            compile_db_path = os.path.abspath(compile_db_path)
            if not os.path.exists(compile_db_path):
                raise FileNotFoundError(f"Compilation database not found: {compile_db_path}")

            self.compile_db = clang.cindex.CompilationDatabase.fromDirectory(
                os.path.dirname(compile_db_path))
            if self.verbose:
                print(f"Loaded compilation database from {compile_db_path}")
        except Exception as e:
            print(f"Failed to load compilation database: {e}")
            raise

    def generate_ast(self, file_path: str) -> Dict[str, Any]:
        """
        Generate AST for a single C++ file
        
        Args:
            file_path: Path to the C++ file
        
        Returns:
            Dictionary representing the AST with extracted functions
        """
        try:
            if self.verbose and threading.current_thread() == threading.main_thread():
                print(f"Generating AST for {file_path}")

            # Get thread-specific Clang index
            index = self._get_thread_index()

            # Get compile commands for this file if compilation database is available
            compile_args = []
            if self.compile_db:
                # Normalize file path to match compilation database entries
                normalized_path = os.path.normpath(file_path)
                commands = self.compile_db.getCompileCommands(normalized_path)
                if commands:
                    compile_args = list(commands[0].arguments)
                    compile_args = remove_clang_unsupport_cmds(compile_args) 
                    # if self.verbose:
                    #     print(f"Using compile arguments for {file_path}: {compile_args}")
                else:
                    if self.verbose:
                        print(f"No compile commands found for {file_path}")

            # Parse the file with compile arguments
            translation_unit = self.index.parse(file_path, args=compile_args)
            
            # Check for parsing errors
            if translation_unit: # and translation_unit.spelling:
                if self.verbose and translation_unit.diagnostics:
                    for diag in translation_unit.diagnostics:
                        if diag.severity > 3:  # Only show errors and fatal errors
                            print(f"Clang error in {file_path}: {diag}")
            
            # Convert AST to dictionary
            ast_dict = self._node_to_dict(translation_unit.cursor) if self.ast else None
            
            # Extract functions information
            functions = self._extract_functions(translation_unit.cursor)
            
            return {
                'file_path': transfer_file_path(file_path, self.repo_root, self.repo_dest_root),
                'ast': ast_dict,
                'functions': functions,
                'compile_args': compile_args
            }
        except Exception as e:
            print(f"Failed to generate AST for {file_path}: {e}")
            return {
                'file_path': transfer_file_path(file_path, self.repo_root, self.repo_dest_root),
                'error': str(e)
            }

    def _node_to_dict(self, node: clang.cindex.Cursor) -> Dict[str, Any]:
        """
        Convert a Clang cursor node to a dictionary
        
        Args:
            node: Clang cursor node
        
        Returns:
            Dictionary representation of the node
        """
        result = {
            'kind': str(node.kind),
            'spelling': node.spelling,
            'displayname': node.displayname,
            'location': {
                'file': str(Path(node.location.file.name).resolve()) if node.location.file else None,
                'line': node.location.line,
                'column': node.location.column
            },
            'children': []
        }
        
        # Add children
        for child in node.get_children():
            if child.location is None or child.location.file is None:
                continue
            if is_external_file(child.location.file.name):
                continue
            result['children'].append(self._node_to_dict(child))
        
        return result

    def _extract_qt_connect_info(self, node: clang.cindex.Cursor) -> Optional[Dict[str, str]]:
        """
        Extract signal and slot information from a Qt connect call

        Args:
            node: Clang cursor node representing a connect call

        Returns:
            Dictionary with 'signal' and 'slot' names, or None if not a valid Qt connect
        """
        try:
            # Get all arguments of the connect function
            args = list(node.get_arguments())
            if len(args) < 4:
                return None
            # Case 1: Old syntax with SIGNAL and SLOT macros
            # Case 2: New syntax with function pointers
            # QObject::connect(sender, &Sender::signal, receiver, &Receiver::slot)
            elif args[1].kind in [clang.cindex.CursorKind.MEMBER_REF_EXPR, clang.cindex.CursorKind.UNEXPOSED_EXPR,
                                  clang.cindex.CursorKind.UNARY_OPERATOR]:
                # Get the children of UNEXPOSED_EXPR
                src_obj = list(args[0].get_children())[0]
                if src_obj.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
                    src_obj = list(src_obj.get_children())[0]
                signal = list(args[1].get_children())[0]
                singal_func = parse_qt_signal_slot_func(src_obj, signal)
                dst_obj = list(args[2].get_children())[0]
                if dst_obj.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
                    dst_obj = list(dst_obj.get_children())[0]
                slot = list(args[3].get_children())[0]
                slot_func = parse_qt_signal_slot_func(dst_obj, slot)
                if slot_func is None or singal_func is None or len(singal_func) == 0 or len(slot_func) == 0:
                    return None
                return singal_func[0], slot_func
        except Exception as e:
            if self.verbose:
                print(f"Failed to extract Qt connect info: {e}")
        return None

    def _extract_functions(self, node: clang.cindex.Cursor) -> List[Dict[str, Any]]:
        """
        extract function information from ast, including function call relationships and function bodies
        
        args:
            node: root ast node
        
        returns:
            list of function information dictionaries with call relationships and bodies
        """
        functions = []
        function_map = {}  # map from function name to function info
        current_function = None  # track current function being analyzed
        
        def traverse(node: clang.cindex.Cursor):
            nonlocal current_function
            # if node.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
            #     node = list(node.get_children())[0] if len(list(node.get_children())) > 0 else node
            #     if node.kind in [clang.cindex.CursorKind.MEMBER_REF_EXPR]:
            #         node = node.referenced
            if node.kind in [clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CXX_METHOD, 
                             clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR]:
                func_name = parse_cursor_func_node(node)
                func_info = {
                    'name': func_name,
                    'location': {
                        'file': transfer_file_path(str(Path(node.location.file.name).resolve()) if node.location.file else None, 
                                                   self.repo_root, self.repo_dest_root),
                        'line': node.location.line,
                    },
                    'called_functions': [],  # add called functions list
                    'body': ''  # add function body field
                }
                node_id = generator_func_node_id(func_info)
                func_info['name'] = node_id
                
                function_body = ''
                for child in node.get_children():
                    if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
                        try:
                            if child.location.file and os.path.exists(child.location.file.name):
                                encoding, _1 = detect_file_encoding(child.location.file.name)
                                with open(child.location.file.name, 'r', encoding=encoding) as f:
                                    f.seek(child.extent.start.offset)
                                    function_body = f.read(child.extent.end.offset - child.extent.start.offset)
                                # clean up the function body
                                function_body = function_body.strip()
                        except Exception as e:
                            if self.verbose:
                                print(f"failed to extract function body for {func_name}: {e}")
                if function_body == '':
                    return None

                # set the function body
                if self.body:
                    func_info['body'] = function_body
                
                functions.append(func_info)
                function_map[func_name] = func_info
                
                # set current function for child traversal
                old_current_function = current_function
                current_function = func_info
                
                # traverse children (function body and other nodes)
                for child in node.get_children():
                    traverse(child)
                
                # restore current function
                current_function = old_current_function
            
            # check if current node is a function call
            elif node.kind == clang.cindex.CursorKind.CALL_EXPR and current_function is not None:
                name = current_function['name']
                if name.find('GIPBridgeBeamCustomEditor::createCentralWidget') != -1:
                    if node.location and node.location.file:
                        line = node.location.line
                        file_name = Path(node.location.file.name).stem
                    print('debug')
                if node.spelling == 'connect':
                    signal_info = self._extract_qt_connect_info(node)
                    if signal_info:
                        signal_slot_map = {}
                        signal_slot_map['signal'] = signal_info[0]
                        signal_slot_map['slot'] = signal_info[1]
                        if current_function.get('emitted_signals') is None: 
                            current_function['emitted_signals'] = []
                        current_function['emitted_signals'].append(signal_slot_map)
                else:
                    called_func_name = parse_cursor_call_func_node(node)
                    if called_func_name:
                        current_function['called_functions'].append(called_func_name)
            # continue traversing children for other node types
            for child in node.get_children():
                if child.location is None or child.location.file is None:
                    continue
                if  is_external_file(child.location.file.name):
                    continue
                traverse(child)
        
        traverse(node)
        return functions
    
    def _process_directory(self, dir_path: str, files: List[str]) -> List[Dict[str, Any]]:
        """
        Process all files in a directory (executed by a single thread)
        
        Args:
            dir_path: Directory path
            files: List of files in the directory
        
        Returns:
            List of AST results for the directory
        """
        dir_results = []
        
        for file_path in files:
            ast_result = self.generate_ast(file_path)
            dir_results.append(ast_result)
            
            # Update progress
            with self._results_lock:
                self._processed_files += 1
                if self.verbose and self._processed_files % 10 == 0:
                    print(f"Progress: {self._processed_files}/{self._total_files} files processed")
            
            # Save AST if output directory is specified
            if self.output_dir and 'error' not in ast_result:
                # try:
                #     rel_path = os.path.relpath(file_path, self.repo_root)
                # except Exception as e:
                #     rel_path = Path(file_path).stem
                rel_path = Path(file_path).stem
                output_path = os.path.join(self.output_dir, f"{rel_path}.ast.json")
                
                # Create directory structure if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(ast_result, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Failed to save AST for {file_path}: {e}")
        
        return dir_results

    
    def group_files_by_directory(self, files: List[str]) -> Dict[str, List[str]]:
        """
        Group files by their parent directory
        
        Args:
            files: List of file paths
        
        Returns:
            Dictionary mapping directory paths to list of files in that directory
        """
        dir_groups = defaultdict(list)
        for file_path in files:
            dir_path = os.path.dirname(file_path)
            dir_groups[dir_path].append(file_path)
        return dir_groups
    
    def process_repository_single_thread(self) -> List[Dict[str, Any]]:
        """
        Process all C++ files in the repository using a single thread (for comparison)
        
        Returns:
            List of AST dictionaries for each file
        """
        cpp_files = self.find_cpp_files()
        results = []
        
        for file_path in cpp_files:
            ast_result = self.generate_ast(file_path)
            results.append(ast_result)
            
            # Update progress
            self._processed_files += 1
            if self.verbose and self._processed_files % 10 == 0:
                print(f"Progress: {self._processed_files}/{self._total_files} files processed")
            
            # Save AST if output directory is specified
            if self.output_dir and 'error' not in ast_result:
                # try:
                #     rel_path = os.path.relpath(file_path, self.repo_root)
                # except Exception as e:
                #     rel_path = Path(file_path).stem
                rel_path = Path(file_path).stem
                output_path = os.path.join(self.output_dir, f"{rel_path}.ast.json")
                
                # Create directory structure if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(ast_result, f, indent=2, ensure_ascii=False)
        
        return results

    def process_repository(self) -> List[Dict[str, Any]]:
        """
        Process all C++ files in the repository
        
        Returns:
            List of AST dictionaries for each file
        """
        cpp_files = self.find_cpp_files()
        if not cpp_files:
            return []
        # Group files by directory
        dir_groups = self.group_files_by_directory(cpp_files)
        if self.verbose:
            print(f"Processing {len(cpp_files)} files in {len(dir_groups)} directories")
            print(f"Using thread pool with {self.max_workers or 'default'} workers") 
        
        all_results = []

        # Process directories in parallel using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all directory processing tasks
            future_to_dir = {
                executor.submit(self._process_directory, dir_path, files): dir_path
                for dir_path, files in dir_groups.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_dir):
                dir_path = future_to_dir[future]
                try:
                    dir_results = future.result()
                    all_results.extend(dir_results)
                    
                    if self.verbose:
                        print(f"Completed processing directory: {dir_path}")
                except Exception as e:
                    print(f"Error processing directory {dir_path}: {e}")
        
        return all_results
    
    def build_function_call_graph(self, ast_results: List[Dict[str, Any]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        nodes = set()
        # 遍历所有文件的函数信息
        for file_data in ast_results:
            data = file_data
            if 'functions' in data:
                for func in data['functions']:
                    func_id = func['name']
                    # 将函数添加到图中
                    graph.add_node(func_id, **func)
                    nodes.add(func_id)

        for file_data in ast_results:
            data = file_data
            if 'functions' in data:
                functions = data['functions']
                for func in functions:
                    if 'emitted_signals' in func:
                        for emit_signal in func['emitted_signals']:
                            signal = transfer_cov_func_name(emit_signal['signal'])
                            exist, node1 = has_key(nodes, signal)
                            if not exist:
                                graph.add_node(signal)
                            slot_list = emit_signal['slot']
                            for slot in slot_list:
                                if slot:
                                    slot = transfer_cov_func_name(slot)
                                    exist, node2 = has_key(nodes, slot)
                                    if not exist:
                                        graph.add_node(slot)
                                    graph.add_edge(signal, slot)

        unadd_call_funcs = []
        for file_data in ast_results:
            data = file_data
            if 'functions' in data:
                for func in data['functions']:
                    func_id = func['name']
                    # 添加函数调用关系
                    if 'called_functions' in func:
                        for called_func in func['called_functions']:
                            called_func_id = called_func
                            if graph.has_node(called_func_id):
                                graph.add_edge(func_id, called_func_id)
                            else:
                                unadd_call_funcs.append(called_func_id)
                                
        if self.verbose:
            print(f"构建完成函数调用图，包含 {len(graph.nodes)} 个节点和 {len(graph.edges)} 条边")
            if len(unadd_call_funcs) > 0:
                unadd_call_funcs_file = Path('./result') / 'unadd_graphy_call_functions.json'
                print(f"Warning: Total {len(unadd_call_funcs)} called functions not add graph nodes, please check {unadd_call_funcs_file}")
                dump(unadd_call_funcs, unadd_call_funcs_file)

        return graph
    
    def run(self) -> nx.DiGraph:
        try:
            import time
            start_time = time.time()
            if self.max_workers == 1:
                results = self.process_repository_single_thread()
            else:
                results = self.process_repository()
            # Print summary
            success_count = sum(1 for r in results if 'error' not in r)
            error_count = len(results) - success_count
            
            print(f"Processed {len(results)} files: {success_count} successful, {error_count} failed")

            end_time = time.time() 
            print(f"ast parser time taken: {end_time - start_time:.2f} seconds")
            
            start_time = time.time()
            # 步骤2: 构建函数调用图
            function_call_graph = self.build_function_call_graph(results)

            #流化调用关系图
            graph_serializer = DiGraphSerializer()
            graph_serializer.serialize_digraph(function_call_graph, os.path.join(self.output_dir, "nx_graph.gpickle")) 
            end_time = time.time() 

            print(f"build function call graph time taken: {end_time - start_time:.2f} seconds")
            print(f"AST outputs saved to {self.output_dir}")
            return results, function_call_graph
        except Exception as e:
            print(f"Error: {e}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Generate AST for C++ code repository using Clang')
    parser.add_argument("--config", type=str, default="./config/ast_config.json", help="Path to config file")
    # parser.add_argument('repo_root', type=str, help='Path to the C++ code repository')
    # parser.add_argument('repo_dest_root', type=str, help='Path to the C++ code repository')
    # parser.add_argument('codes', nargs='+', type=str, help='Path to the C++ code repository')
    # parser.add_argument('-o', '--output', help='Directory to save AST outputs')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    # parser.add_argument('--ast', action='store_true', help='Enable verbose logging')
    # parser.add_argument('--body', action='store_true', help='Enable verbose logging')
    # parser.add_argument('-c', '--compile-db', help='Path to the compilation database (compile_commands.json)')
    # parser.add_argument('-w', '--workers', type=int, help='Number of threads to use (default: CPU count)')
    # parser.add_argument('--single-thread', action='store_true', help='Use single thread mode (for testing)')
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        generator = ThreadPoolClangASTGenerator(kwargs=config)
        ast_results, func_graph = generator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())