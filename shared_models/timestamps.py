


import csv

import csv
import logging
import os
from google.cloud import storage

class TimestampsManager:
    def __init__(self, bucket_name):
        bucket_name = bucket_name
    
    def charger_timestamps_depuis_csv_s3(self, nom_fichier):
        """
        Load timestamps from the S3 bucket.

        Args:
            nom_fichier (str): Name of the file on S3.

        Returns:
            list: List of timestamps (as dictionaries).
        """
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)
        content = blob.download_as_text()

        lecteur_csv = csv.DictReader(content.splitlines())
        return [ligne for ligne in lecteur_csv]


    
    
    def charger_timestamps_depuis_fichier_local(self, nom_fichier):
        """
        Load timestamps from a local CSV file.

        Args:
            nom_fichier (str): File path to the local CSV file.

        Returns:
            list: List of timestamps (as dictionaries).
        """
        if not os.path.exists(nom_fichier):
            return []  # Retourne une liste vide si le fichier n'existe pas
        try:
            with open(nom_fichier, mode='r', encoding='utf-8') as file:
                lecteur_csv = csv.DictReader(file)
                return [ligne for ligne in lecteur_csv]
        except Exception as e:
            logging.error(f"Error loading timestamps from {nom_fichier}: {e}")
            return []

    def enregistrer_timestamps_dans_csv(self, timestamps, nom_fichier):
        """
        Enregistre les données des timestamps dans un fichier CSV sur le bucket S3.

        Args:
            timestamps (list): Liste des timestamps sous forme de dictionnaires.
            nom_fichier (str): Nom du fichier à enregistrer dans le bucket S3.
        """
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)

        # Définir les en-têtes du fichier CSV
        entetes = ['Id', 'X', 'X_t', 'A', 'A_t', 'B', 'B_t', 'C', 'C_t', 'D', 'Nom',
                'Direction', 'Harmonic', 'Prix_seuil', 'Prix_limit', 'True1', 'True2', 'True3', 'True4']

        # Construire les lignes du fichier CSV
        lignes = [entetes]
        for timestamp in timestamps:
            lignes.append([
                timestamp.get('Id', ''),
                timestamp.get('X', ''),
                timestamp.get('X_t', ''),
                timestamp.get('A', ''),
                timestamp.get('A_t', ''),
                timestamp.get('B', ''),
                timestamp.get('B_t', ''),
                timestamp.get('C', ''),
                timestamp.get('C_t', ''),
                timestamp.get('D', ''),
                timestamp.get('Nom', ''),
                timestamp.get('Direction', ''),
                timestamp.get('Harmonic', ''),
                timestamp.get('Prix_seuil', ''),
                timestamp.get('Prix_limit', ''),
                timestamp.get('True1', ''),
                timestamp.get('True2', ''),
                timestamp.get('True3', ''),
                timestamp.get('True4', '')
            ])

        # Générer le contenu CSV
        content = "\n".join([",".join(map(str, ligne)) for ligne in lignes])
        blob.upload_from_string(content)
