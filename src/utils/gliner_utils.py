import subprocess
import os
import json

def run_gliner_inference(file_path: str) -> list:
    # Exemple de commande pour exécuter GLiNER
    command = ["python", "gliner.py", "--infer", file_path]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"GLiNER inference failed: {result.stderr}")
    # Supposons que le résultat est au format JSON
    return json.loads(result.stdout)

def create_gliner_dataset(data: list, format: str = "json-ner") -> str:
    dataset_path = f"datasets/dataset.{format}"
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, 'w') as f:
        if format == "json-ner":
            json.dump(data, f, indent=4)
        elif format == "conllu":
            # Implémenter la conversion en CONLLU
            conllu_data = convert_to_conllu(data)
            f.write(conllu_data)
        else:
            raise ValueError("Format de dataset non supporté.")
    return dataset_path

def convert_to_conllu(data: list) -> str:
    # Implémenter la logique de conversion en CONLLU
    # Ceci est un exemple simplifié
    conllu_str = ""
    for item in data:
        token_id = item.get("token_id", 0)
        token = item.get("token", "")
        entity = item.get("entity", "O")
        conllu_str += f"{token_id}\t{token}\t_\t_\t_\t_\t_\t_\t_\t{entity}\n"
    return conllu_str

def train_gliner_model(dataset_path: str, epochs: int, batch_size: int) -> str:
    # Exemple de commande pour entraîner GLiNER
    command = ["python", "gliner.py", "--train", dataset_path, "--epochs", str(epochs), "--batch_size", str(batch_size)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"GLiNER training failed: {result.stderr}")
    # Supposons que le run_id est retourné
    return result.stdout.strip()
