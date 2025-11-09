# # qa/llm_runner.py
# import os
# import requests
# import time
# import re
# from dotenv import load_dotenv

# load_dotenv()

# class LLMRunner:
#     def __init__(self, model: str = None, temperature: float = 0.2, max_context_chars: int = 4000, api_base: str = "https://api.groq.com/openai/v1"):
#         self.api_key = os.getenv("GROQ_API_KEY")
#         if not self.api_key:
#             raise ValueError("GROQ_API_KEY not found in .env")
#         self.model = model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
#         self.temperature = float(temperature)
#         self.base_url = api_base.rstrip("/")
#         self.completions_url = f"{self.base_url}/chat/completions"
#         self.max_context_chars = int(max_context_chars)

#     def _truncate(self, text: str, cap: int):
#         if not text:
#             return ""
#         return text if len(text) <= cap else text[-cap:]

#     def _parse_error_limits(self, body: dict):
#         try:
#             msg = body.get("error", {}).get("message", "") or ""
#             nums = [int(n) for n in re.findall(r"\b\d+\b", msg)]
#             if len(nums) >= 2:
#                 return {"limit": nums[0], "requested": nums[1], "message": msg}
#             if len(nums) == 1:
#                 return {"limit": nums[0], "requested": None, "message": msg}
#         except Exception:
#             pass
#         return {"limit": None, "requested": None, "message": None}

#     def answer(self, query: str, context: str) -> str:
#         # aggressive initial truncation
#         context = self._truncate(context or "", self.max_context_chars)
#         prompt = f"You are an expert assistant. Use ONLY the context to answer. Context:\n{context}\n\nQuestion: {query}\nAnswer concisely and cite chunk ids."
#         payload = {
#             "model": self.model,
#             "messages": [{"role":"system","content":"You are a factual assistant."},{"role":"user","content":prompt}],
#             "temperature": self.temperature,
#             "stream": False
#         }
#         headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
#         attempts = 0
#         while attempts < 3:
#             attempts += 1
#             try:
#                 resp = requests.post(self.completions_url, headers=headers, json=payload, timeout=60)
#             except Exception as e:
#                 return f"[LLM network error] {e}"
#             if resp.status_code == 200:
#                 try:
#                     data = resp.json()
#                     choice = data.get("choices", [{}])[0]
#                     message = choice.get("message", {}) or {}
#                     content = message.get("content") or choice.get("text")
#                     return content.strip() if content else "[empty response]"
#                 except Exception as e:
#                     return f"[LLM parse error] {e}"
#             # handle token limit errors by truncating and retrying
#             if resp.status_code in (413, 429) or (resp.status_code == 400 and "tokens" in resp.text.lower()):
#                 body = {}
#                 try:
#                     body = resp.json()
#                 except Exception:
#                     body = {"error": {"message": resp.text}}
#                 parsed = self._parse_error_limits(body)
#                 limit = parsed.get("limit")
#                 if limit:
#                     # safe tokens -> chars estimate (conservative)
#                     safe_chars = int(limit * 0.7 * 4)
#                     payload["messages"][1]["content"] = f"You are an expert assistant. Use ONLY the context to answer.\n\nContext:\n{self._truncate(context, safe_chars)}\n\nQuestion: {query}\nAnswer concisely."
#                 else:
#                     # fallback halve the context each retry
#                     context = context[-(len(context)//2):]
#                     payload["messages"][1]["content"] = f"You are an expert assistant. Use ONLY the context to answer.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer concisely."
#                 time.sleep(0.5 * attempts)
#                 continue
#             # other errors: return body to help debugging
#             try:
#                 err = resp.json()
#             except Exception:
#                 err = resp.text
#             return f"[LLM API error] {resp.status_code}: {err}"
#         return "[LLM API error] failed after retries due to token limits"

# qa/llm_runner.py
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
CHROMA_TELEMETRY_ENABLED = "false"
class LLMRunner:
    def __init__(self,
                 model: str = None,
                 temperature: float = 0.2,
                 max_context_chars: int = 20000,
                 groq_api_url: str = "https://api.groq.com/openai/v1"):
        """
        REST-based Groq client with automatic handling of decommissioned models.
        - Set GROQ_API_KEY and optional GROQ_MODEL in .env
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        # allow env override; fallback to a recommended replacement
        self.model = model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"
        self.temperature = temperature
        self.max_context_chars = max_context_chars
        self.base_url = groq_api_url.rstrip("/")
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"

    def _truncate_context(self, context: str) -> str:
        if not context:
            return context
        if len(context) <= self.max_context_chars:
            return context
        return context[-self.max_context_chars :]

    def _call_api(self, payload: dict):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.completions_url, headers=headers, json=payload, timeout=60)
        return resp

    def _fetch_available_models(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            r = requests.get(self.models_url, headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            # API likely returns list like {"data": [...]} or list; normalize
            if isinstance(data, dict) and "data" in data:
                models = [m.get("id") or m for m in data["data"]]
            elif isinstance(data, list):
                models = [m.get("id") if isinstance(m, dict) else m for m in data]
            else:
                models = []
            return [m for m in models if m]
        except Exception:
            return []

    def answer(self, query: str, context: str) -> str:
        """
        Build the prompt, call Groq REST API. If model is decommissioned, query /models and retry.
        """
        context = self._truncate_context(context)
        prompt = f"""
You are an expert AI assistant. Use ONLY the provided context to answer the question truthfully and clearly. 
Don't give short answer. Specify atleast 2-3 lines.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer concisely and cite sources by chunk id when relevant.
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a factual assistant for RAG QA."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.temperature),
            "stream": False
        }

        # first attempt
        resp = self._call_api(payload)
        if resp.status_code == 200:
            return self._parse_response(resp)

        # if 400 with model_decommissioned, try to pick a replacement and retry once
        try:
            body = resp.json()
        except Exception:
            body = {"error": {"message": resp.text}}

        # detect model decommission error
        err_msg = body.get("error", {}).get("message", "") if isinstance(body, dict) else str(body)
        if "decommission" in err_msg.lower() or "model" in err_msg.lower() and "decommission" in err_msg.lower() or body.get("error", {}).get("code") == "model_decommissioned":
            # fetch available models and pick one (prefer recommended list)
            available = self._fetch_available_models()
            # prefer recommended replacements if present
            preferred = ["mistral-saba-24b", "llama-3.3-70b-versatile", "llama3-70b-8192", "gemma-7b-it"]
            for p in preferred:
                if p in available:
                    new_model = p
                    break
            else:
                new_model = available[0] if available else None

            if new_model:
                old = self.model
                self.model = new_model
                # retry once
                payload["model"] = self.model
                time.sleep(0.5)
                resp2 = self._call_api(payload)
                if resp2.status_code == 200:
                    return self._parse_response(resp2)
                else:
                    # return helpful message with both errors
                    try:
                        msg2 = resp2.json()
                    except Exception:
                        msg2 = resp2.text
                    return f"[Groq API error after retry] {resp.status_code}: {err_msg} -- retry result: {resp2.status_code}: {msg2}"
            else:
                return f"[Groq API error] model decommissioned and no replacement found: {err_msg}"

        # other errors
        return f"[Groq API error] {resp.status_code}: {err_msg}"

    def _parse_response(self, resp):
        try:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            # Groq response shape may vary; handle a couple of cases
            message = choice.get("message", {}) or {}
            content = message.get("content")
            if content:
                return content.strip()
            if "text" in choice:
                return choice["text"].strip()
            return "[Groq API error] unexpected response format"
        except Exception as e:
            return f"[Groq API error] parse error: {e}"
