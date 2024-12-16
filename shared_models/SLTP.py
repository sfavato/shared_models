
import os

import csv
import logging
from google.cloud import storage

# region SLTP
class SLTP:
    def __init__(self, id, nom, direction, harmonic, entry=0, sl=0, tp1=0, tp2=0, tp3=0, tp4=0, tp5=0, prix_courant=0, comment='', link='', trailingSL='No'):
        self.id = id
        self.nom = nom
        self.direction = direction or 'SHORT'
        self.harmonic = harmonic
        self.entry = entry
        self.entry_prc = 0
        self.sl = sl
        self.sl_prc = 0
        self.tp1 = tp1
        self.tp1_prc = 0
        self.tp2 = tp2
        self.tp2_prc = 0
        self.tp3 = tp3
        self.tp3_prc = 0
        self.tp4 = tp4
        self.tp4_prc = 0
        self.tp5 = tp5
        self.tp5_prc = 0
        self.prix_courant = prix_courant
        self.comment = comment
        self.link = link
        self.trailingSL = trailingSL

    

    def handle_sl(self):
        if self.direction == 'LONG':
            if self.prix_courant < float(self.sl):
                if self.comment != 'invalidated':
                    self.comment = 'invalidated'

        else:
            if self.prix_courant > float(self.sl):
                if self.comment != 'invalidated':

                    self.comment = 'invalidated'

    def mise_a_jour_prix(self, data):
        symbole = self.nom + "USDT"
        for item in data:
            if item['symbol'] == symbole:
                self.prix_courant = float(item['lastPrice'])
                self.handle_sl()
                break


    def calculate_sltp_percentage(self):
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



    def adjust_stoploss(self):
        # Adjust stop-loss by 2% for SELL positions
        if self.direction.lower() == 'sell' or self.direction.lower() == 'short':
            stop_loss_adjusted = self.sl * \
                1.0017  # Adjusting stop-loss upwards by 0.01%
        else:
            stop_loss_adjusted = self.sl * 0.9983



        return stop_loss_adjusted




# endregion




class SltpManager:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name


    def find_sltp(self, nom, direction, harmonic, tf, sltps):
        found_sltp = None
        for sltp in sltps:
            if (sltp.nom == nom and sltp.direction == direction and sltp.harmonic == harmonic and sltp.tf == tf):
                found_sltp = sltp
                break
        return found_sltp


    def find_sltp_id(self, id, sltps):
        found_sltp = None
        for sltp in sltps:
            if (sltp.id == id):
                found_sltp = sltp
                break
        return found_sltp

    def update_sl_and_send_message(sltp, tp_hit, new_sl, message):
        if sltp.trailingSL == 'No':
            if sltp.comment != tp_hit:
                message = "{} - sl kept the same .".format(tp_hit)
                sltp.comment = tp_hit
        else:
            sltp.sl = new_sl
            if sltp.comment != tp_hit:
                message = "{} - sl modified to {}.".format(tp_hit, new_sl)
                sltp.comment = tp_hit

    def check_tp_levels(self, sltp, price, tp_levels, tp_messages):
        for i, tp in enumerate(tp_levels):
            if (price > tp if sltp.direction == 'LONG' else price < tp):
                if sltp.trailingSL == 'Each TP':
                    sltp.update_sl_and_send_message(
                        tp_messages[i], tp_levels[i - 1] if i > 0 else sltp.entry, tp_messages[i])
                elif sltp.trailingSL == 'Each 2 TP' and i % 2 == 1:
                    sltp.update_sl_and_send_message(
                        tp_messages[i], tp_levels[i - 2] if i > 1 else sltp.entry, tp_messages[i])
                return True
        return False

    def handle_tp_and_sl(self, sltp, price, tp_levels, tp_messages):
        if sltp.direction == 'LONG':
            if price < float(sltp.sl):
                if sltp.sl in tp_levels:
                    sltp.comment = 'closed'
                    message = "Closed "
                elif sltp.comment != 'invalidated':
                    sltp.direction = 'SHORT'
                    sltp.comment = 'invalidated'
            elif sltp.check_tp_levels(price, tp_levels[::-1], tp_messages[::-1]):
                return
        else:
            if price > float(sltp.sl):
                if sltp.comment != 'invalidated':
                    sltp.direction = 'LONG'
                    sltp.comment = 'invalidated'
            elif sltp.check_tp_levels(price, tp_levels, tp_messages):
                return

    def charge_sltps_depuis_cloud(self, nom_fichier):
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


    def enregistrer_sltps_dans_csv(self, sltps, nom_fichier):

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
            
    
    def charge_sltps_depuis_fichier_local(nom_fichier):
        """
        Load SLTP data from a local CSV file into a list of SLTP objects.

        Args:
            nom_fichier (str): The file path of the CSV file.

        Returns:
            list: A list of SLTP objects.
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
