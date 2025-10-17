import json


from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    func_doc_language_specific_pre_processing,
)
from overrides import override
import re


class Granite4LocalHandlerV2Thinking(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        '''
        for no-thinking mode (default), use greedy but for thinking mode, it is recommended to use a sampling temperature of 0.7, a top-p value of 0.95, and a top-k value of 20 generation instead of greedy for better results. Additionally, apply a presence penalty of 1.5 to encourage the generation of more diverse content and reduce repetitions.
        '''
        print(f"HANDLER: Granite4LocalHandlerV2Thinking")
        super().__init__(model_name, temperature)

        self.temperature = 0.7
        self.thiking_parameters = True

    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, function, tokenize=False, add_generation_prompt=True, thinking=True
        )
        return formatted_prompt

    @override
    def decode_ast(self, result, language="Python"):
        # print(result)
        # import ipdb; ipdb.set_trace()

        # Remove <think>...</think> blocks
        # result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        result = re.sub(r'.*?</think>', '', result, flags=re.DOTALL).strip()

        # Extract JSON strings between <tool_call> tags
        tool_jsons = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', result, flags=re.DOTALL)

        # Convert JSON strings to dicts
        tool_calls = [json.loads(j) for j in tool_jsons]

        if isinstance(tool_calls, list) and len(tool_calls) == 1 and isinstance(tool_calls[0], list):
            # import ipdb; ipdb.set_trace()
            # print()
            tool_calls = tool_calls[0]

        result = json.dumps(tool_calls)

        decoded_outputs = []
        try:
            result = eval(result)
        except:
            result = json.loads(result)
        if type(result) == dict and 'tool_calls' in result:
            result = result['tool_calls']
        if type(result) == dict:
            result = [result]
        for res in result:
            if type(res) != dict:
                try:
                    res = json.loads(res.strip())
                except:
                    pass
            fnname = res.get("name", "").strip()
            args = res.get("arguments", {})
            if args == {}:
                args = res.get("parameters", {})
            if type(args) == str:  # TODO: needed for 8B -- non-tuned
                try:
                    args = json.loads(args.strip())
                except:
                    pass
            if fnname == "no_function":
                decoded_outputs.append("No function is called")
                continue

            decoded_outputs.append({fnname: args})
        # print(decoded_outputs)
        # ipdb.set_trace()
        return decoded_outputs

    @override
    def decode_execute(self, result):
        decoded_outputs = []
        for decoded_output in self.decode_ast(result):
            if isinstance(decoded_output, dict):
                fname = next(iter(decoded_output.keys()))
                args = next(iter(decoded_output.values()))
                args = ','.join([f'{argname}={repr(argval)}' for argname, argval in args.items()])
                decoded_outputs.append(f"{fname}({args})")
        return decoded_outputs

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)

        # Granite use its own system prompt

        return {"message": [], "function": functions}