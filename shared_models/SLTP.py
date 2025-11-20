
import os
import csv
import logging
from typing import List, Optional, Dict, Any, Union
from google.cloud import storage

# region SLTP
class SLTP:
    """
    Gère la logique de Stop-Loss (SL) et Take-Profit (TP) pour un trade actif.

    Cette classe est responsable de la protection du capital (SL) et de la sécurisation des gains (TP).
    Elle encapsule les seuils de prix critiques et la logique de mise à jour dynamique (Trailing SL)
    pour adapter la sortie du trade à l'évolution du marché.
    """
    def __init__(self,
                 id: str,
                 nom: str,
                 direction: str,
                 harmonic: str,
                 entry: Union[float, str] = 0,
                 sl: Union[float, str] = 0,
                 tp1: Union[float, str] = 0,
                 tp2: Union[float, str] = 0,
                 tp3: Union[float, str] = 0,
                 tp4: Union[float, str] = 0,
                 tp5: Union[float, str] = 0,
                 prix_courant: Union[float, str] = 0,
                 comment: str = '',
                 link: str = '',
                 trailingSL: str = 'No'):
        """
        Initialise un gestionnaire SL/TP pour un trade spécifique.

        Args:
            id (str): Identifiant unique du trade.
            nom (str): Symbole de l'actif (ex: 'BTC').
            direction (str): 'LONG' ou 'SHORT'.
            harmonic (str): Nom du pattern harmonique associé.
            entry (Union[float, str]): Prix d'entrée.
            sl (Union[float, str]): Prix du Stop-Loss initial.
            tp1 (Union[float, str]): Premier niveau de prise de profit.
            tp2 (Union[float, str]): Deuxième niveau de prise de profit.
            tp3 (Union[float, str]): Troisième niveau de prise de profit.
            tp4 (Union[float, str]): Quatrième niveau de prise de profit.
            tp5 (Union[float, str]): Cinquième niveau de prise de profit.
            prix_courant (Union[float, str]): Dernier prix connu.
            comment (str): État ou commentaire actuel (ex: 'invalidated', 'tp1 hit').
            link (str): Lien vers le graphique ou l'analyse.
            trailingSL (str): Configuration du Trailing SL ('No', 'Each TP', 'Each 2 TP').
        """
        self.id = id
        self.nom = nom
        self.direction = direction or 'SHORT'
        self.harmonic = harmonic
        self.entry = float(entry)
        self.entry_prc = 0
        self.sl = float(sl)
        self.sl_prc = 0
        self.tp1 = float(tp1)
        self.tp1_prc = 0
        self.tp2 = float(tp2)
        self.tp2_prc = 0
        self.tp3 = float(tp3)
        self.tp3_prc = 0
        self.tp4 = float(tp4)
        self.tp4_prc = 0
        self.tp5 = float(tp5)
        self.tp5_prc = 0
        self.prix_courant = float(prix_courant)
        self.comment = comment
        self.link = link
        self.trailingSL = trailingSL

    

    def handle_sl(self) -> None:
        """
        Vérifie si le Stop-Loss a été atteint en fonction du prix courant.

        Si le prix franchit le niveau de SL dans la mauvaise direction, le statut (commentaire)
        est mis à jour à 'invalidated' pour signaler la clôture nécessaire de la position.
        """
        if self.direction == 'LONG':
            if self.prix_courant < float(self.sl):
                if self.comment != 'invalidated':
                    self.comment = 'invalidated'

        else:
            if self.prix_courant > float(self.sl):
                if self.comment != 'invalidated':

                    self.comment = 'invalidated'

    def mise_a_jour_prix(self, data: List[Dict[str, Any]]) -> None:
        """
        Met à jour le prix courant à partir d'un flux de données de marché.

        Cette méthode synchronise l'objet avec le marché réel et déclenche immédiatement
        la vérification du Stop-Loss.

        Args:
            data (List[Dict[str, Any]]): Liste de dictionnaires de tickers.
        """
        symbole = self.nom + "USDT"
        for item in data:
            if item['symbol'] == symbole:
                self.prix_courant = float(item['lastPrice'])
                self.handle_sl()
                break


    def calculate_sltp_percentage(self) -> None:
        """
        Calcule la distance en pourcentage entre le prix courant et tous les niveaux clés (Entry, SL, TPs).

        Ces pourcentages sont essentiels pour l'interface utilisateur, permettant au trader
        de visualiser rapidement combien de marge il reste avant un TP ou un SL.
        """
        self.entry_prc = format(((float(self.prix_courant) - float(self.entry)) /
                                 float(self.entry) * 100), ".2f") if self.entry else None
        self.sl_prc = format(((float(self.prix_courant) - float(self.sl)) /
                              float(self.sl) * 100), ".2f") if self.sl else None
        self.tp1_prc = format(((float(self.prix_courant) - float(self.tp1)) /
                               float(self.tp1) * 100), ".2f") if self.tp1 else None
        self.tp2_prc = format(((float(self.prix_courant) - float(self.tp2)) /
                               float(self.tp2) * 100), ".2f") if self.tp2 else None
        self.tp3_prc = format(((float(self.prix_courant) - float(self.tp3)) /
                               float(self.tp3) * 100), ".2f") if self.tp3 else None
        self.tp4_prc = format(((float(self.prix_courant) - float(self.tp4)) /
                               float(self.tp4) * 100), ".2f") if self.tp4 else None
        self.tp5_prc = format(((float(self.prix_courant) - float(self.tp5)) /
                               float(self.tp5) * 100), ".2f") if self.tp5 else None



    def adjust_stoploss(self) -> float:
        """
        Calcule un ajustement fin du Stop-Loss pour tenir compte du spread ou des frais.

        Ajoute ou soustrait un petit pourcentage (environ 0.17%) au SL théorique pour éviter
        d'être sorti par le bruit du marché ou le spread de l'exchange.

        Returns:
            float: Le prix du Stop-Loss ajusté.
        """
        # Adjust stop-loss by 2% for SELL positions
        if self.direction.lower() == 'sell' or self.direction.lower() == 'short':
            stop_loss_adjusted = self.sl * \
                1.0017  # Adjusting stop-loss upwards by 0.01%
        else:
            stop_loss_adjusted = self.sl * 0.9983



        return stop_loss_adjusted




# endregion




class SltpManager:
    """
    Gère la collection des objets SLTP et leur persistance.

    Cette classe offre des méthodes pour rechercher, mettre à jour et sauvegarder
    les configurations de SL/TP, agissant comme une interface entre la logique métier
    et le stockage (Cloud/Fichier).
    """
    def __init__(self, bucket_name: str):
        """
        Initialise le gestionnaire.

        Args:
            bucket_name (str): Nom du bucket Google Cloud Storage.
        """
        self.bucket_name = bucket_name


    def find_sltp(self, nom: str, direction: str, harmonic: str, tf: str, sltps: List[SLTP]) -> Optional[SLTP]:
        """
        Cherche un objet SLTP spécifique dans une liste basée sur ses caractéristiques clés.

        Args:
            nom (str): Symbole.
            direction (str): 'LONG' ou 'SHORT'.
            harmonic (str): Nom du pattern.
            tf (str): Timeframe.
            sltps (List[SLTP]): Liste des objets à parcourir.

        Returns:
            Optional[SLTP]: L'objet trouvé ou None.
        """
        found_sltp = None
        for sltp in sltps:
            if (sltp.nom == nom and sltp.direction == direction and sltp.harmonic == harmonic and sltp.tf == tf):
                found_sltp = sltp
                break
        return found_sltp


    def find_sltp_id(self, id: str, sltps: List[SLTP]) -> Optional[SLTP]:
        """
        Cherche un objet SLTP par son identifiant unique.

        Args:
            id (str): L'ID du trade.
            sltps (List[SLTP]): Liste des objets à parcourir.

        Returns:
            Optional[SLTP]: L'objet trouvé ou None.
        """
        found_sltp = None
        for sltp in sltps:
            if (sltp.id == id):
                found_sltp = sltp
                break
        return found_sltp

    def update_sl_and_send_message(self, sltp: SLTP, tp_hit: str, new_sl: float, message: str) -> None:
        """
        Met à jour le Stop-Loss d'un trade suite à l'atteinte d'un Take-Profit (Trailing SL).

        Cette méthode implémente la stratégie de sécurisation des gains : quand un TP est touché,
        le SL est remonté pour verrouiller une partie des profits ou mettre le trade à "Break-Even".

        Args:
            sltp (SLTP): L'objet SLTP à modifier.
            tp_hit (str): Le nom du TP atteint (ex: 'TP1').
            new_sl (float): Le nouveau prix du Stop-Loss.
            message (str): Message de log ou de notification (référence mutable, mais non retournée ici).
        """
        if sltp.trailingSL == 'No':
            if sltp.comment != tp_hit:
                message = "{} - sl kept the same .".format(tp_hit)
                sltp.comment = tp_hit
        else:
            sltp.sl = new_sl
            if sltp.comment != tp_hit:
                message = "{} - sl modified to {}.".format(tp_hit, new_sl)
                sltp.comment = tp_hit

    def check_tp_levels(self, sltp: SLTP, price: float, tp_levels: List[float], tp_messages: List[str]) -> bool:
        """
        Vérifie si un ou plusieurs niveaux de Take-Profit ont été franchis.

        Si un TP est franchi, cette méthode déclenche la logique de mise à jour du SL (Trailing).

        Args:
            sltp (SLTP): L'objet SLTP.
            price (float): Prix courant.
            tp_levels (List[float]): Liste des prix des TP.
            tp_messages (List[str]): Liste des labels correspondants (ex: ['TP1', 'TP2']).

        Returns:
            bool: True si un TP a été franchi, False sinon.
        """
        for i, tp in enumerate(tp_levels):
            if (price > tp if sltp.direction == 'LONG' else price < tp):
                if sltp.trailingSL == 'Each TP':
                    self.update_sl_and_send_message(
                        sltp, tp_messages[i], tp_levels[i - 1] if i > 0 else sltp.entry, tp_messages[i])
                elif sltp.trailingSL == 'Each 2 TP' and i % 2 == 1:
                    self.update_sl_and_send_message(
                        sltp, tp_messages[i], tp_levels[i - 2] if i > 1 else sltp.entry, tp_messages[i])
                return True
        return False

    def handle_tp_and_sl(self, sltp: SLTP, price: float, tp_levels: List[float], tp_messages: List[str]) -> None:
        """
        Orchestre la vérification complète des conditions de sortie (SL et TP).

        Vérifie d'abord si le SL est touché (priorité à la protection). Sinon, vérifie les TP.

        Args:
            sltp (SLTP): L'objet trade.
            price (float): Prix actuel.
            tp_levels (List[float]): Niveaux de TP.
            tp_messages (List[str]): Labels des TP.
        """
        if sltp.direction == 'LONG':
            if price < float(sltp.sl):
                if sltp.sl in tp_levels:
                    sltp.comment = 'closed'
                    message = "Closed "
                elif sltp.comment != 'invalidated':
                    sltp.direction = 'SHORT'
                    sltp.comment = 'invalidated'
            elif self.check_tp_levels(sltp, price, tp_levels[::-1], tp_messages[::-1]):
                return
        else:
            if price > float(sltp.sl):
                if sltp.comment != 'invalidated':
                    sltp.direction = 'LONG'
                    sltp.comment = 'invalidated'
            elif self.check_tp_levels(sltp, price, tp_levels, tp_messages):
                return

    def charge_sltps_depuis_cloud(self, nom_fichier: str) -> List[SLTP]:
        """
        Charge les configurations SLTP depuis le Cloud Storage.

        Args:
            nom_fichier (str): Nom du fichier CSV.

        Returns:
            List[SLTP]: Liste d'objets SLTP chargés.
        """
        sltps = []
        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)
        content = blob.download_as_text()

        lecteur_csv = csv.DictReader(content.splitlines())
        for ligne in lecteur_csv:
            indice = SLTP(
                id=ligne['Id'],
                nom=ligne['Nom'],
                direction=ligne['Direction'] if ligne['Direction'] else 'SHORT',
                harmonic=ligne['Harmonic'],
                entry=float(ligne['Entry']) if ligne['Entry'] else 0,
                sl=float(ligne['Sl']) if ligne['Sl'] != 'None' else 0,
                tp1=float(ligne['Tp1']) if ligne['Tp1'] != 'None' else 0,
                tp2=float(ligne['Tp2']) if ligne['Tp2'] != 'None' else 0,
                tp3=float(ligne['Tp3']) if ligne['Tp3'] != 'None' else 0,
                tp4=float(ligne['Tp4']) if ligne['Tp4'] != 'None' else 0,
                tp5=float(ligne['Tp5']) if ligne['Tp5'] != 'None' else 0,
                comment=ligne['Comment'] if ligne['Comment'] else '',
                prix_courant=ligne['Prix_courant'] if ligne['Prix_courant'] else 0,
                link=ligne['Link'] if ligne['Link'] else '',
                trailingSL=ligne.get('TrailingSL', 'No'))
            sltps.append(indice)
        return sltps


    def enregistrer_sltps_dans_csv(self, sltps: List[SLTP], nom_fichier: str) -> None:
        """
        Sauvegarde les configurations SLTP actuelles dans le Cloud.

        Args:
            sltps (List[SLTP]): Liste des objets à sauvegarder.
            nom_fichier (str): Nom du fichier de destination.
        """

        client = storage.Client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)

        entetes = ['Id', 'Nom', 'Direction', 'Harmonic', 'Entry', 'Sl', 'Tp1',
                'Tp2', 'Tp3', 'Tp4', 'Tp5', 'Comment', 'Prix_courant', 'Link', 'trailingSL']
        lignes = [entetes]

        for indice in sltps:

            lignes.append([
                indice.id,
                indice.nom,
                indice.direction,
                indice.harmonic,
                f"{indice.entry:.6f}" if isinstance(
                    indice.entry, (float, int)) else indice.entry,
                f"{indice.sl:.6f}" if isinstance(
                    indice.sl, (float, int)) else indice.sl,
                f"{indice.tp1:.6f}" if isinstance(
                    indice.tp1, (float, int)) else indice.tp1,
                f"{indice.tp2:.6f}" if isinstance(
                    indice.tp2, (float, int)) else indice.tp2,
                f"{indice.tp3:.6f}" if isinstance(
                    indice.tp3, (float, int)) else indice.tp3,
                f"{indice.tp4:.6f}" if isinstance(
                    indice.tp4, (float, int)) else indice.tp4,
                f"{indice.tp5:.6f}" if isinstance(
                    indice.tp5, (float, int)) else indice.tp5,
                indice.comment,
                f"{indice.prix_courant:.6f}" if isinstance(
                    indice.prix_courant, (float, int)) else indice.prix_courant,
                indice.link,
                f"{indice.trailingSL:.6f}" if isinstance(
                    indice.trailingSL, (float, int)) else indice.trailingSL,
            ])

        content = "\n".join([",".join(map(str, ligne)) for ligne in lignes])
        try:
            blob.upload_from_string(content)
        except Exception as e:
            logging.error(f"Failed to upload the file {nom_fichier}: {e}")
            
    
    @staticmethod
    def charge_sltps_depuis_fichier_local(nom_fichier: str) -> List[SLTP]:
        """
        Charge les données SLTP depuis un fichier CSV local (Dev/Test).

        Args:
            nom_fichier (str): Chemin du fichier local.

        Returns:
            List[SLTP]: Liste des objets SLTP.
        """
        sltps = []
        if not os.path.exists(nom_fichier):
            return []  # Retourne une liste vide si le fichier n'existe pas
        try:
            with open(nom_fichier, mode='r', encoding='utf-8') as file:
                lecteur_csv = csv.DictReader(file)
                for ligne in lecteur_csv:
                    sltp = SLTP(
                    id=ligne['Id'],
                    nom=ligne['Nom'],
                    direction=ligne['Direction'] if ligne['Direction'] else 'SHORT',
                    harmonic=ligne['Harmonic'],
                    entry=float(ligne['Entry']) if ligne['Entry'] else 0,
                    sl=float(ligne['Sl']) if ligne['Sl'] != 'None' else 0,
                    tp1=float(ligne['Tp1']) if ligne['Tp1'] != 'None' else 0,
                    tp2=float(ligne['Tp2']) if ligne['Tp2'] != 'None' else 0,
                    tp3=float(ligne['Tp3']) if ligne['Tp3'] != 'None' else 0,
                    tp4=float(ligne['Tp4']) if ligne['Tp4'] != 'None' else 0,
                    tp5=float(ligne['Tp5']) if ligne['Tp5'] != 'None' else 0,
                    comment=ligne['Comment'] if ligne['Comment'] else '',
                    prix_courant=ligne['Prix_courant'] if ligne['Prix_courant'] else 0,
                    link=ligne['Link'] if ligne['Link'] else '',
                    trailingSL=ligne.get('TrailingSL', 'No'))
                    sltps.append(sltp)
        except Exception as e:
            logging.error(f"Error loading SLTPs from {nom_fichier}: {e}")
        return sltps
