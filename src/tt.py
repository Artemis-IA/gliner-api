from gliner import GLiNER

# Charger le modèle GLiNER
model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")

# Extraire la configuration du modèle
entity_types = list(model.config.id2label.values())

# Afficher les types d'entités pris en charge
print("Liste des entités prises en charge par GLiNER :")
for entity in entity_types:
    print(entity)
