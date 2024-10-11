from gliner import GLiNER
import json

model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
config = model.config.to_dict()
print(json.dumps(config, indent=4))
