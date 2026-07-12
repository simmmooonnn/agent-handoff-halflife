import os

class _HF:
    def __init__(self, model):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.name = f"hf/{model.split('/')[-1]}"
        # R1.3 decoding-variance support: TEMP>0 switches to sampling; DSEED picks an
        # independent decoding stream. Both are folded into self.name, so CachedBackend
        # (which keys on name:item_id) isolates every (temp, dseed) run from the greedy
        # caches automatically. Default TEMP=0 keeps byte-identical greedy behavior.
        self.temp = float(os.environ.get("TEMP", "0") or 0)
        self.dseed = int(os.environ.get("DSEED", "0") or 0)
        if self.temp > 0:
            self.name += f"@t{self.temp:g}s{self.dseed}"
        # trust_remote_code gated behind TRC=1: OFF by default so natively-supported
        # families (Qwen2.5, Mistral, Phi-3-mini/medium) use the in-library modeling
        # code. Forcing it ON makes Phi-3-mini/medium load the repo's stale
        # modeling_phi3.py, which KeyErrors on the newer rope_scaling schema. Only
        # custom-arch models (Phi-3-small = Phi3Small) need TRC=1.
        trc = os.environ.get("TRC") == "1"
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trc)
        kw = dict(device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=trc)
        if os.environ.get("LOAD4BIT") == "1":
            kw = dict(device_map="auto", trust_remote_code=trc,
                      quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16))
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
    def generate(self, prompt, item_id, max_tokens):
        import torch
        msgs = [{"role": "user", "content": prompt}]
        # transformers >=5 returns a BatchEncoding (dict) here, not a bare tensor,
        # so request a dict and splat it into generate.
        inputs = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        if self.temp > 0:
            # deterministic-but-independent stream per (dseed, item): reruns hit the
            # same sample, different dseeds give genuinely different trajectories
            import hashlib
            torch.manual_seed(int(hashlib.sha1(
                f"{self.dseed}:{item_id}".encode()).hexdigest()[:8], 16))
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      do_sample=True, temperature=self.temp, top_p=0.95)
        else:
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

class _API:
    def __init__(self, provider, model):
        self.name = f"{provider}/{model}"; self.provider = provider; self.model = model
        # GEN_TEMP>0 enables sampling (diversity-coded handoffs need independent draws);
        # folded into name so temp>0 caches never collide with the greedy temp=0 caches.
        # NB: not "TEMP" -- on Windows that is the reserved temp-dir path env var.
        self.temp = float(os.environ.get("GEN_TEMP", "0") or 0)
        if self.temp > 0:
            self.name += f"@t{self.temp:g}"
        if provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY unset")
            from openai import OpenAI; self.client = OpenAI()
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"): raise RuntimeError("ANTHROPIC_API_KEY unset")
            import anthropic; self.client = anthropic.Anthropic()
    def generate(self, prompt, item_id, max_tokens):
        import time
        last = None
        for attempt in range(9):        # ~8 min of retry coverage survives transient DNS/network blips
            try:
                if self.provider == "openai":
                    kw = dict(model=self.model, temperature=self.temp,
                              messages=[{"role": "user", "content": prompt}])
                    if self.model.startswith(("gpt-5", "o3", "o4")):
                        # reasoning-model API: max_tokens is rejected; effort "none"
                        # disables hidden reasoning so the call matches the plain
                        # greedy-summarization regime used everywhere else.
                        kw.update(max_completion_tokens=max_tokens,
                                  reasoning_effort="none")
                    else:
                        kw.update(max_tokens=max_tokens)
                    r = self.client.chat.completions.create(**kw)
                    return r.choices[0].message.content or ""
                kw = dict(model=self.model, max_tokens=max_tokens,
                          messages=[{"role": "user", "content": prompt}])
                if self.model.startswith(("claude-fable", "claude-opus-4-8")):
                    # adaptive thinking cannot be disabled directly, but effort=low
                    # suppresses thinking blocks entirely (verified) -- matching the
                    # no-hidden-reasoning regime used for every other model.
                    kw["extra_body"] = {"output_config": {"effort": "low"}}
                if not getattr(self, "_no_temp", False):
                    kw["temperature"] = self.temp
                r = self.client.messages.create(**kw)
                # thinking-capable models emit [thinking, text, ...]; take text blocks
                return "".join(b.text for b in r.content
                               if getattr(b, "type", "") == "text")
            except Exception as e:
                # newest Anthropic models (fable-5, opus-4-8) reject the temperature
                # param outright ("`temperature` is deprecated for this model") --
                # drop it once and retry immediately; decoding is the provider default.
                if "temperature" in str(e) and "deprecated" in str(e) \
                        and not getattr(self, "_no_temp", False):
                    self._no_temp = True
                    continue
                last = e
                time.sleep(min(60, 2 * 2 ** attempt))
        raise last

def build_model_backend(provider, model):
    if provider == "hf": return _HF(model)
    if provider in ("openai", "anthropic"): return _API(provider, model)
    raise ValueError(provider)
