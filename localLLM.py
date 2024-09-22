from llama_cpp import Llama

import os
import configparser
import errno

class LLMconfig:
    def __init__(self, config_ini_path = './configs/config.ini'):
        self.config_ini = configparser.ConfigParser()
        if not os.path.exists(config_ini_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)
        self.config_ini.read(config_ini_path, encoding='utf-8')
        LLM_items = self.config_ini.items('LLM')
        self.LLM_config_dict = dict(LLM_items)

class LLM:
    def __init__(self, config_ini_path = './configs/config.ini') :
        LLM_config = LLMconfig(config_ini_path = config_ini_path)
        config_dict = LLM_config.LLM_config_dict

        self.simpleLLM_messages = []
        self.model = config_dict["llm_model"]
        self.temperature = float(config_dict["temperature"])
        self.max_tokens = int(config_dict["max_tokens"])
        self.n_gpu_layers_num = int(config_dict["n_gpu_layers_num"])
        self.n_ctx = int(config_dict["n_ctx"])
        self.cpu_mode = bool(config_dict["cpu_mode"])
        self.verbose = bool(config_dict["verbose"])

        if self.cpu_mode:
            self.llm = Llama(model_path=self.model, n_ctx=self.n_ctx, verbose=self.verbose)
        else:
            self.llm = Llama(model_path=self.model, n_gpu_layers=self.n_gpu_layers_num, n_ctx=self.n_ctx, verbose=self.verbose)

    def change_prompt(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += "system:" + message["content"] + "\n"
            elif message["role"] == "assistant":
                prompt += "assistant:" + message["content"] + "\n"
            elif message["role"] == "user":
                prompt += "user:" + message["content"] + "\n"
        prompt += "assistant:"
        return prompt

    def simpleLLM(self, user_prompt, temp_sys_prompt=''):
        self.simpleLLM_messages.append({"role": "system", "content": temp_sys_prompt})
        self.simpleLLM_messages.append({"role": "user", "content": user_prompt})

        making_prompt = self.change_prompt(self.simpleLLM_messages)

        res = self.llm.create_completion(
            making_prompt, 
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["user"]
        )

        return res['choices'][0]['text']

    def simpleLLMstream_prepare(self, user_prompt, temp_sys_prompt=None):
        self.simpleLLM_messages.append({"role": "system", "content": temp_sys_prompt})
        self.simpleLLM_messages.append({"role": "user", "content": user_prompt})

        making_prompt = self.change_prompt(self.simpleLLM_messages)

        return making_prompt, self.max_tokens, self.temperature

def main():
    # 設定ファイルのパスを指定
    config_path = './configs/config.ini'
    
    # LLMインスタンスを初期化
    llm = LLM(config_ini_path=config_path)
    
    # ユーザープロンプトとシステムプロンプトを定義
    sys_prompt = "あなたは天気予報に関する専門家です。"
    user_prompt = "今週の天気予報を教えてください。"

    # LLMを使って応答を取得
    response = llm.simpleLLM(user_prompt, temp_sys_prompt=sys_prompt)
    
    # 応答を表示
    print("LLMからの応答:")
    print(response)

if __name__ == "__main__":
    main()