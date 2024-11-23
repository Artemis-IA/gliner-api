import os

def concat_python_files(input_path, output_file, exclude_list=None):
    """
    Concatène le contenu de tous les fichiers Python d'un répertoire donné dans un fichier de sortie.
    
    :param input_path: Chemin du répertoire racine à parcourir
    :param output_file: Chemin du fichier de sortie
    :param exclude_list: Liste des sous-répertoires ou fichiers à exclure (chemins relatifs ou absolus)
    """
    if exclude_list is None:
        exclude_list = []

    # Ajouter `alembic` à la liste des exclusions
    exclude_list.append("alembic")

    # Normaliser les chemins d'exclusion pour comparaison
    exclude_list = [os.path.abspath(os.path.join(input_path, exclude)) for exclude in exclude_list]

    # Ouvrir le fichier de sortie en mode écriture
    with open(output_file, 'w') as fichier_final:
        # Parcourir le répertoire racine et ses sous-répertoires
        for root, dirs, files in os.walk(input_path):
            # Exclure les sous-répertoires spécifiés
            dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in exclude_list]
            
            for file in files:
                chemin_fichier = os.path.abspath(os.path.join(root, file))
                chemin_fichier_rel = os.path.relpath(chemin_fichier, input_path)
                
                # Vérifier si le fichier est dans la liste d'exclusion ou s'il n'est pas un fichier .py
                if chemin_fichier in exclude_list or not file.endswith('.py'):
                    continue

                # Ajouter le chemin relatif en commentaire au début du fichier
                fichier_final.write(f"\n# {chemin_fichier_rel}\n")
                fichier_final.write("#-----\n")  # Séparateur de début

                # Lire et écrire le contenu du fichier
                with open(chemin_fichier, 'r') as f:
                    fichier_final.write(f.read())
                
                # Ajouter le séparateur de fin
                fichier_final.write("\n#-----\n")
    
    print(f"Les fichiers Python ont été concaténés dans {output_file}.")

if __name__ == "__main__":
    # Variables modifiables par l'utilisateur
    path = input("Entrez le chemin du répertoire racine : ").strip()  # Répertoire à parcourir
    output_path = input("Entrez le chemin du fichier de sortie : ").strip()  # Fichier de sortie
    exclusions = input("Entrez les chemins des sous-répertoires ou fichiers à exclure (séparés par des virgules) : ").strip()

    # Traiter les exclusions (convertir en liste)
    exclude_list = [x.strip() for x in exclusions.split(",")] if exclusions else []

    # Appeler la fonction
    concat_python_files(path, output_path, exclude_list)
