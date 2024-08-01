import giskard
from typing import Sequence, Optional
from giskard.llm.client import set_default_client
from giskard.llm.client.base import LLMClient, ChatMessage
import json
from depends.model import llm_anton_llama2


class MyLLMClient(LLMClient):
    def __init__(self, my_client):
        self._client = my_client

    def complete(
            self,
            messages: Sequence[ChatMessage],
            temperature: float = 1,
            max_tokens: Optional[int] = None,
            caller_id: Optional[str] = None,
            seed: Optional[int] = None,
            format=None,
    ) -> ChatMessage:
        # Create the prompt
        prompt = ""
        for msg in messages:
            if msg.role.lower() == "assistant":
                prefix = "\n\nAssistant: "
            else:
                prefix = "\n\nHuman: "

            prompt += prefix + msg.content

        prompt += "\n\nAssistant: "

        # Create the body
        params = {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens or 1000,
            "temperature": temperature,
            "top_p": 0.9,
        }
        body = json.dumps(params)

        response = self._client.invoke_model(
            body=body,
            modelId=self._model_id,
            accept="application/json",
            contentType="application/json",
        )
        data = json.loads(response.get("body").read())

        return ChatMessage(role="assistant", message=data["completion"])

set_default_client(MyLLMClient(llm_anton_llama2))