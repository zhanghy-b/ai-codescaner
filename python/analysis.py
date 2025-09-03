import json
import requests
import networkx as nx
import parse_function
import extractor_function
from typing import List, Dict, Any


class CodeAnalysis:
    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/generate"
        self.coverage_data = self.load_data("data/coverage.json")
        self.call_graph, self.func_info = self.build_call_graph("data/callgraph.json")
        self.entry_points = self.load_entry_points("data/entrypoints.txt")

    def load_data(self, file_path: str) -> List[str]:
        """加载覆盖率数据"""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_call_graph(
        self, file_path: str
    ) -> tuple[nx.DiGraph, Dict[str, List[str]]]:
        """构建调用图"""
        calls, locations = parse_function.parse_function_data(file_path)

        G = nx.DiGraph()
        for caller, callees in calls.items():
            if len(callees) != 0:
                for callee in callees:
                    G.add_edge(caller, callee)
            else:
                G.add_edge(caller, "")

        return G, locations

    def load_entry_points(self, file_path: str) -> List[str]:
        """加载入口点列表"""
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]

    def find_reachable_functions(self) -> set:
        """找出所有可从入口点到达的函数"""
        reachable = set()
        for entry_point in self.entry_points:
            if entry_point in self.call_graph:
                reachable.update(nx.descendants(self.call_graph, entry_point))

        return reachable

    def find_func_code(self, funcs: List[str]) -> Dict[str, str]:
        """查找函数实现代码"""
        func_code: Dict[str, str] = {}
        for func in funcs:
            if func in self.func_info:
                func_code[func] = extractor_function.extract_function_from_start_line(
                    self.func_info[func][0], int(self.func_info[func][1])
                )
        return func_code

    def query_ollama(self, prompt: str) -> str:
        """向Ollama发送查询请求"""
        payload = {"model": "gemma3:1b", "prompt": prompt, "stream": False}

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return ""

    def analyze_function(self, func: str, reachable: set) -> Dict:
        """分析单个函数是否为废弃代码"""
        # 静态分析：检查是否可从入口点到达
        is_reachable = func in reachable

        # 准备大模型分析提示
        prompt = self.create_analysis_prompt(func, is_reachable)

        # 查询大模型
        analysis_result = self.query_ollama(prompt)

        return {
            "function": func,
            "file": self.func_info.get(func, ""),
            "is_reachable": is_reachable,
            "llm_analysis": analysis_result,
            "likely_dead": not is_reachable and "是废弃代码" in analysis_result,
        }

    def create_analysis_prompt(self, func: str, is_reachable: bool) -> str:
        """创建分析提示"""
        # 获取函数的调用者和被调用者信息
        callers = list(self.call_graph.predecessors(func))
        callees = list(self.call_graph.successors(func))

        # 获取函数的源代码
        func_code: Dict[str, str] = {}
        func_code.update(self.find_func_code([func]))
        func_code.update(self.find_func_code(callers))
        func_code.update(self.find_func_code(callees))

        prompt = f"""
请分析以下函数是否可能是废弃代码，并给出理由：

函数名: {func}
源代码: {"\n".join([f"{key}: {value}" for key, value in func_code.items()])}

调用关系:
- 被以下函数调用: {', '.join(callers) if callers else '无'}
- 调用以下函数: {', '.join(callees) if callees else '无'}

静态分析结果: {'可从入口点到达' if is_reachable else '不可从入口点到达'}

请考虑以下因素：
1. 是否有框架注解(如@Controller、@RequestMapping)
2. 是否可能通过反射调用
3. 是否有条件编译指令
4. 是否是接口实现或抽象方法实现
5. 是否有被测试代码调用但未覆盖的情况

请用中文回答，并明确指出该函数是否可能是废弃代码，回答“是废弃代码”、“不是废弃代码”或“不确定是否为废弃代码”，并说明理由和置信度（高/中/低）。
"""
        return prompt

    def run_analysis(self):
        """运行完整分析流程"""
        print("开始分析废弃代码...")

        # 步骤1: 找出所有可到达的函数
        reachable = self.find_reachable_functions()
        print(f"找到 {len(reachable)} 个可到达的函数")

        # 步骤2: 获取覆盖率为0的函数
        zero_coverage_funcs = self.coverage_data
        print(f"找到 {len(zero_coverage_funcs)} 个覆盖率为0的函数")

        # 步骤3: 分析每个函数
        results = []
        for i, func in enumerate(zero_coverage_funcs):
            print(f"分析函数 {i+1}/{len(zero_coverage_funcs)}: {func}")
            result = self.analyze_function(func, reachable)
            results.append(result)

        # 步骤4: 保存结果
        with open("results/report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 步骤5: 生成摘要报告
        likely_dead = [r for r in results if r["likely_dead"]]
        print(f"\n分析完成! 发现 {len(likely_dead)} 个可能废弃的函数")

        return results


if __name__ == "__main__":
    analyzer = CodeAnalysis()
    analyzer.run_analysis()
