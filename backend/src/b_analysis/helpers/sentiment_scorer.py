from typing import Dict, Any

from src.utils.config import subreddit_to_model_map, default_model


class SentimentScorer:

    def __init__(self):
        # -- 1. subreddit → model mapping.
        self.subreddit_models = subreddit_to_model_map
        self.default_model = default_model

        # -- 2. pipeline cache.
        self.pipelines = {}
        self.device = self.select_device()

    # --- main scoring method ---

    def score(self, text: str, subreddit: str = None) -> Dict:

        # -- 1. select model based on subreddit.
        model_name = self.select_model(subreddit)
        
        # -- 2. load pipeline.
        pipeline = self.get_pipeline(model_name)

        # -- 3. if pipeline failed to load, try else.
        if pipeline is None:
            if model_name != self.default_model:
                pipeline = self.get_pipeline(self.default_model)
                if pipeline is None:
                    model_name = self.default_model
                
            if pipeline is None:
                raise Exception(f"failed to load pipeline for model : {model_name}")


        # -- 4. process text with pipeline.
        try:
            # these models are 512-token max; we keep it explicit + predictable.
            result = pipeline(text, truncation=True, max_length=512)
        except Exception:
            return {
                "score": 0.0,
                "category": "neutral",
                "model_used": model_name,
            }

        # -- 5. extract label and score from result.
        # transformers pipeline can return either:
        # - [ {'label': ..., 'score': ...} ]  (single top result)
        # - [ [ {'label': ..., 'score': ...} ] ] (when top_k is used)
        top = result[0]
        if isinstance(top, list) and top:
            top = top[0]

        label = str(top.get("label", "")).lower()
        conf = float(top.get("score", 0.0))

        # we need to normalize label formats bc we use diff models :
        # - finbert returns: positive/neutral/negative
        # - twitter-roberta often returns: label_0/label_1/label_2 (or LABEL_*)
        if label.startswith("label_"):
            # try model config mapping first (best).
            try:
                idx = int(label.split("_", 1)[1])
                mapped = pipeline.model.config.id2label.get(idx, "").lower()  # type: ignore[attr-defined]
            except Exception:
                mapped = ""

            # common sentiment head mapping if config isn't available.
            if not mapped:
                mapped = {0: "negative", 1: "neutral", 2: "positive"}.get(idx, "")

            label = mapped or label

        # update score based on our label.
        if "positive" in label:
            score = conf
        elif "negative" in label:
            score = -conf
        else:
            score = 0.0

        # -- 6. categorize from our score.
        if score > 0.5:
            category = "strong_bullish"
        elif score > 0.2:
            category = "moderate_bullish"
        elif score > -0.2:
            category = "neutral"
        elif score > -0.5:
            category = "moderate_bearish"
        else:
            category = "strong_bearish"

        return {
            "score": score,
            "category": category,
            "model_used": model_name,
        }

    # --- helper methods ---

    def select_model(self, subreddit: str = None) -> str:
        # select nlp model based on specific subreddit.
        if subreddit and subreddit.lower() in self.subreddit_models:
            return self.subreddit_models[subreddit.lower()]
        
        # if no subreddit is provided, use the default model.
        return self.default_model

    def get_pipeline(self, model_name: str) -> Any:
        
        # -- 1. check cache first.
        if model_name in self.pipelines:
            return self.pipelines[model_name]

        # -- 2. decide device (gpu if available, else cpu).
        # transformers pipeline uses: device=-1 for cpu, device=0 for first cuda gpu.
        device = self.device

        # -- 2. import transformers.
        try:
            from transformers import pipeline
        except Exception:
            self.pipelines[model_name] = None
            return None

        # -- 3. try to load the model.
        try:
            pipeline_obj = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
                top_k=1,
                device=device,
            )
            self.pipelines[model_name] = pipeline_obj
            return pipeline_obj
        except Exception:
            self.pipelines[model_name] = None
            return None

    def select_device(self) -> int:
        # keep it simple: use gpu if torch sees one, else cpu.
        try:
            import torch
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return -1