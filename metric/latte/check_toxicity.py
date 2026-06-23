import sys
import os
import json, re
import argparse
from openai import OpenAI

from MAF.metric.latte.get_bias_score import GetBiasScore
from MAF.metric.latte.get_llm_response import GetLLMResponse
from MAF.utils.local_llm import generate_text, resolve_local_model_name, should_use_local_llm

parent_dir = os.environ["PYTHONPATH"]
MODEL_NAME = "gpt-4o-2024-05-13"


class ScoringToxicity:
    """
    - Model Selection step에서 통과한 모델을 가지고, 해당 분야에서 score를 하는 sample 코드
    - GPT-4o는 demeaning 부분에서 70%이상 달성하였기에, 관련 분야에 대한 text toxicity score를 잴 수 있음
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.use_local = should_use_local_llm(model_name)
        self.model_name = (
            resolve_local_model_name(model_name) if self.use_local else model_name
        )
        self.client = None if self.use_local else OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def get_openai_response(
        self, text: str, instruction: str, end: str, temperature: float = 0
    ):
        if self.use_local:
            return generate_text(
                instruction + text + end,
                model_name=self.model_name,
                max_new_tokens=32,
            )
        output = self.client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            messages=[
                {"role": "user", "content": instruction + text + end},
            ],
        )
        res = output.choices[0].message.content
        return res

    def get_instruction(
        self, instruct_file_path: str = parent_dir + "/MAF/metric/latte/prompt.json"
    ):
        with open(instruct_file_path, "r") as f:
            instruction = json.load(f)
        instruction, query = instruction[0]["application"]["paradetox"]
        return instruction, query

    def get_gpt4_score(self, text: str):
        pattern = r"the score is (\d+)"
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"the score is:\n\n(\d+)"
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"is(\d+)"
        text = text.replace("\n", "").replace(":", "")
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"score (\d+)\*\*"
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"is\s*(\d+)\."
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"\*\*(\d+)\*\*"
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        pattern = r"(\d+)the"
        text = (
            re.findall(pattern, text.lower())[0]
            if re.findall(pattern, text.lower()) != []
            else text
        )
        try:
            score = int(round(float(text)))
        except ValueError:
            matches = re.findall(r"\b[01]\b", text)
            if not matches:
                matches = re.findall(r"\d+", text)
            if not matches:
                raise
            score = int(round(float(matches[0])))
        score = max(0, min(1, score))
        return score


def check_toxicity(sentence: str):
    res = {"sentence": sentence, "toxicity": "non-toxic"}
    toxicity_checker = ScoringToxicity()
    instruction, query = toxicity_checker.get_instruction()
    toxicity = toxicity_checker.get_openai_response(sentence, instruction, query)
    toxicity_score = toxicity_checker.get_gpt4_score(toxicity)
    if toxicity_score == 1:
        res["toxicity"] = "toxic"
    return res


if __name__ == "__main__":
    text = "Nice weather!"
    print(check_toxicity(text))
