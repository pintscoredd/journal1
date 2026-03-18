import json
import hashlib
import time
from typing import Dict, Any, Optional
from google import genai
from groq import Groq
from db import get_session, AICache
from secrets_store import get_api_key

class AIAdapter:
    def __init__(self, provider: str = "noop"):
        self.provider = provider.lower()
        if self.provider == "gemini":
            key = get_api_key("gemini_api_key")
            if not key:
                raise ValueError("Gemini API key is blank. Please configure it in secrets.toml or Settings.")
            self.gemini_client = genai.Client(api_key=key)
        elif self.provider == "groq":
            key = get_api_key("groq_api_key")
            if not key:
                raise ValueError("Groq API key is blank. Please configure it in secrets.toml or Settings.")
            self.groq_client = Groq(api_key=key)

    def _generate_cache_key(self, prompt: str, quant_json: str, model: str) -> str:
        data = f"{prompt}|{quant_json}|{model}"
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _round_floats(self, obj: Any, precision: int = 4) -> Any:
        if isinstance(obj, float):
            return round(obj, precision)
        if isinstance(obj, dict):
            return {k: self._round_floats(v, precision) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._round_floats(v, precision) for v in obj]
        return obj

    def get_critique(self, prompt_template: str, quant_metrics: Dict[str, Any], model: str = "", image=None) -> str:
        # Default models
        if not model:
            if self.provider == "gemini":
                model = "gemini-2.5-flash"
            elif self.provider == "groq":
                model = "llama3-70b-8192"
            else:
                model = "noop_model"

        quant_json = json.dumps(quant_metrics, sort_keys=True)
        rounded_metrics = self._round_floats(quant_metrics)
        cache_quant_json = json.dumps(rounded_metrics, sort_keys=True)
        cache_key = self._generate_cache_key(prompt_template, cache_quant_json, model)

        # Check Cache
        session = get_session()
        try:
            cached = session.query(AICache).filter_by(cache_key=cache_key).first()
            if cached and cached.response_json:
                return cached.response_json
        finally:
            session.close()

        # Construct final prompt
        final_prompt = f"{prompt_template}\n\nMetrics:\n{quant_json}"

        response_text = ""
        # Call API
        if self.provider == "noop":
            response_text = '{"summary": "No-op offline test.", "trade_quality_checklist": 100}'
        elif self.provider == "gemini":
            contents = [final_prompt]
            if image is not None:
                try:
                    # check if the user configured a vision model, otherwise flash is default 
                    # gemini-2.5-flash supports multimodal natively
                    contents.append(image)
                except Exception:
                    pass
            resp = self.gemini_client.models.generate_content(
                model=model,
                contents=contents
            )
            response_text = resp.text
        elif self.provider == "groq":
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": final_prompt}],
                model=model,
            )
            response_text = chat_completion.choices[0].message.content
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Save Cache
        session = get_session()
        try:
            prompt_hash = hashlib.sha256(prompt_template.encode('utf-8')).hexdigest()
            record = AICache(
                cache_key=cache_key,
                provider=self.provider,
                prompt_hash=prompt_hash,
                response_json=response_text
            )
            session.add(record)
            session.commit()
        finally:
            session.close()

        return response_text
