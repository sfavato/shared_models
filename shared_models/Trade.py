import csv
import logging
import os
from google.cloud import storage

bucket_project="adept-coda-420809"

# region indices
class Trade:
    def __init__(self, id, owner, nom, direction, harmonic, prix_seuil, prix_limit, 
                 true1=None, true2=None, true1_triggered=False, true2_triggered=False, prix_courant=0, 
                 comment='', status='', link='', pourcentage=0, X=0, A=0, B=0, C=0, D=0, old_status='', 
                 has_4h_close_above=False, has_1_retest=False, has_sl=False, 
                 has_tp1=False, has_tp2=False, has_tp3=False, has_tp4=False, has_tp5=False, 
                 trailingSL='No', wantRetest='No', touched_entry=False, true3=0, true4=0, tf='4h', 
                 PreEntrySlightTouch=False, PostEntrySlightTouch=False, PreLimitSlightTouch=False, 
                 PostLimitSlightTouch=False, PreTrue1SlightTouch=False, PostTrue1SlightTouch=False, 
                 PreTrue2SlightTouch=False, PostTrue2SlightTouch=False, confidence=3, invalidation=False, 
                 has_3d_close_above=False, open_3min=0, close_3min=0, 
                 high_3min=0, low_3min=0, open_4h=0, close_4h=0, high_4h=0, low_4h=0):
        self.id = id
        self.owner = owner
        self.nom = nom
        self.direction = direction or 'SHORT'
        self.harmonic = harmonic
        self.prix_seuil = prix_seuil
        self.prix_limit = prix_limit
        self.status = status
        self.true1 = true1
        self.true2 = true2
        self.true3 = true3
        self.true4 = true4
        self.prix_courant = prix_courant
        self.true1_triggered = true1_triggered
        self.true2_triggered = true2_triggered
        self.comment = comment
        self.pourcentage = pourcentage
        self.link = link
        self.X = X
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.old_status = old_status
        self.has_4h_close_above = has_4h_close_above
        self.has_1_retest = has_1_retest
        self.has_sl = has_sl
        self.has_tp1 = has_tp1
        self.has_tp2 = has_tp2
        self.has_tp3 = has_tp3
        self.has_tp4 = has_tp4
        self.has_tp5 = has_tp5
        self.trailingSL = trailingSL
        self.wantRetest = wantRetest
        self.touched_entry = touched_entry
        self.tf = tf
        self.PreEntrySlightTouch = PreEntrySlightTouch
        self.PostEntrySlightTouch = PostEntrySlightTouch
        self.PreLimitSlightTouch = PreLimitSlightTouch
        self.PostLimitSlightTouch = PostLimitSlightTouch
        self.PreTrue1SlightTouch = PreTrue1SlightTouch
        self.PostTrue1SlightTouch = PostTrue1SlightTouch
        self.PreTrue2SlightTouch = PreTrue2SlightTouch
        self.PostTrue2SlightTouch = PostTrue2SlightTouch
        self.confidence = confidence
        self.Invalidation = invalidation
        self.has_3d_close_above = has_3d_close_above
        
        

    def mise_a_jour_prix(self, data):
        symbol = self.nom + 'USDT'
        

        for item in data:
            if item['symbol'] == symbol:
                self.prix_courant = float(item['lastPrice'])
                self.pourcentage = self.pourcentage_difference()

                break
    
    
    def pourcentage_difference(self):
        if self.prix_courant == 0:
            return 99999
        return format(float(abs((self.prix_courant - self.prix_seuil) / self.prix_seuil) * 100),  ".2f") 


    # Function to check if the 4-hour close above or below the entry value
    def check_close_above_entry(self, entry):
        close = float(self.close_4h)
        entry = float(entry)
        if self.direction == "SHORT":
            return close < entry
        else:
            return close > entry










# region import/export
class TradeManager:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name


    def charger_trades_depuis_cloud(self, nom_fichier):
        indices = []
        client = storage.Client(bucket_project)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)
        content = blob.download_as_text()

        lecteur_csv = csv.DictReader(content.splitlines())
        for ligne in lecteur_csv:
            indice = Trade(
                id=ligne['Id'],
                owner=ligne['Owner'],
                nom=ligne['Nom'],
                direction=ligne['Direction'] if ligne['Direction'] else 'SHORT',
                harmonic=ligne['Harmonic'],
                prix_seuil=float(ligne['Prix_seuil']
                                ) if ligne['Prix_seuil'] else 0,
                prix_limit=float(ligne['Prix_limit']
                                ) if ligne['Prix_limit'] else 0,
                true1=float(ligne['True1']) if ligne['True1'] else None,
                true2=float(ligne['True2']) if ligne['True2'] else None,
                true1_triggered=ligne.get('True1_triggered', 'False') == 'True',
                true2_triggered=ligne.get('True2_triggered', 'False') == 'True',
                comment=ligne['Comment'] if ligne['Comment'] else '',
                status=ligne['Status'] if ligne['Status'] else 'N/A',
                prix_courant=float(ligne['Prix_courant']
                                ) if ligne['Prix_courant'] else 0,
                link=ligne['Link'] if ligne['Link'] else '',
                pourcentage=ligne['Pourcentage'] if ligne['Pourcentage'] else 0,
                X=ligne['X'] if ligne['X'] else 0,
                A=ligne['A'] if ligne['A'] else 0,
                B=ligne['B'] if ligne['B'] else 0,
                C=ligne['C'] if ligne['C'] else 0,
                D=ligne['D'] if ligne['D'] else 0,
                old_status=ligne['OldStatus'] if ligne['OldStatus'] else ligne['Status'],
                has_4h_close_above=ligne.get('Has4H', 'False') == 'True',
                has_1_retest=ligne.get('Has1H', 'False') == 'True',
                has_sl=ligne.get('HasSL', 'False') == 'True',
                has_tp1=ligne.get('HasTP1', 'False') == 'True',
                has_tp2=ligne.get('HasTP2', 'False') == 'True',
                has_tp3=ligne.get('HasTP3', 'False') == 'True',
                has_tp4=ligne.get('HasTP4', 'False') == 'True',
                has_tp5=ligne.get('HasTP5', 'False') == 'True',
                trailingSL=ligne.get('TrailingSL', 'No'),
                wantRetest=ligne.get('WantRetest', 'No'),
                touched_entry=ligne.get('Touched_entry', 'False') == 'True',
                true3=ligne.get('True3', 0),
                true4=ligne.get('True4', 0),
                tf=ligne.get('Tf', '4H'),
                PreEntrySlightTouch=ligne.get('PreEntrySlightTouch') == 'True',
                PostEntrySlightTouch=ligne.get('PostEntrySlightTouch') == 'True',
                PreLimitSlightTouch=ligne.get('PreLimitSlightTouch') == 'True',
                PostLimitSlightTouch=ligne.get('PostLimitSlightTouch') == 'True',
                PreTrue1SlightTouch=ligne.get('PreTrue1SlightTouch') == 'True',
                PostTrue1SlightTouch=ligne.get('PostTrue1SlightTouch') == 'True',
                PreTrue2SlightTouch=ligne.get('PreTrue2SlightTouch') == 'True',
                PostTrue2SlightTouch=ligne.get('PostTrue2SlightTouch') == 'True',
                confidence=ligne.get('Confidence', 3),
                invalidation=ligne.get('Invalidation') == 'True',
                has_3d_close_above=ligne.get('Has3dClosedAbove') == 'True',

            )
            indices.append(indice)
        return indices


    def enregistrer_trades_dans_csv(self, indices, nom_fichier):
        client = storage.Client(bucket_project)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)

        entetes = ['Id', 'Owner', 'Nom', 'Direction', 'Harmonic', 'Prix_seuil', 'Prix_limit',
                'True1', 'True2', 'True1_triggered', 'True2_triggered', 'Comment', 'Prix_courant', 'Status', 'Link', 'Pourcentage', 'X', 'A', 'B', 'C', 'D', 'OldStatus', 'Has4H', 'Has1H', 'HasSL', 'HasTP1', 'HasTP2', 'HasTP3', 'HasTP4', 'HasTP5', 'TrailingSL', 'WantRetest', 'Touched_entry', 'True3', 'True4', 'Tf',
                'PreEntrySlightTouch', 'PostEntrySlightTouch', 'PreLimitSlightTouch', 'PostLimitSlightTouch', 'PreTrue1SlightTouch', 'PostTrue1SlightTouch', 'PreTrue2SlightTouch', 'PostTrue2SlightTouch', 'Confidence', 'Invalidation', 'Has3dClosedAbove']
        lignes = [entetes]
        for indice in indices:
            lignes.append([
                indice.id,
                indice.owner,
                indice.nom,
                indice.direction,
                indice.harmonic,
                indice.prix_seuil,
                indice.prix_limit,
                indice.true1 if indice.true1 is not None else '',
                indice.true2 if indice.true2 is not None else '',
                indice.true1_triggered if indice.true1_triggered is not None else 'False',
                indice.true2_triggered if indice.true2_triggered is not None else 'False',
                indice.comment if indice.comment is not None else '',
                indice.prix_courant,
                indice.status,
                indice.link,
                indice.pourcentage,
                indice.X,
                indice.A,
                indice.B,
                indice.C,
                indice.D,
                indice.old_status,
                indice.has_4h_close_above,
                indice.has_1_retest,
                indice.has_sl,
                indice.has_tp1,
                indice.has_tp2,
                indice.has_tp3,
                indice.has_tp4,
                indice.has_tp5,
                indice.trailingSL,
                indice.wantRetest,
                indice.touched_entry,
                indice.true3,
                indice.true4,
                indice.tf,
                indice.PreEntrySlightTouch,
                indice.PostEntrySlightTouch,
                indice.PreLimitSlightTouch,
                indice.PostLimitSlightTouch,
                indice.PreTrue1SlightTouch,
                indice.PostTrue1SlightTouch,
                indice.PreTrue2SlightTouch,
                indice.PostTrue2SlightTouch,
                indice.confidence,
                indice.Invalidation,
                indice.has_3d_close_above,

            ])

        content = "\n".join([",".join(map(str, ligne)) for ligne in lignes])
        blob.upload_from_string(content)
        
        

    def charger_trades_depuis_fichier_local(self,nom_fichier):
        """
        Load trades from a local CSV file into a list of Trade objects.

        Args:
            nom_fichier (str): The file path of the CSV file.

        Returns:
            list: A list of Trade objects.
        """
        trades = []
        if not os.path.exists(nom_fichier):
            return []  # Retourne une liste vide si le fichier n'existe pas
        try:
            with open(nom_fichier, mode='r', encoding='utf-8') as file:
                lecteur_csv = csv.DictReader(file)
                for ligne in lecteur_csv:
                    trade = Trade(
                        id=ligne['Id'],
                        owner=ligne['Owner'],
                        nom=ligne['Nom'],
                        direction=ligne['Direction'] if ligne['Direction'] else 'SHORT',
                        harmonic=ligne['Harmonic'],
                        prix_seuil=float(ligne['Prix_seuil']
                                        ) if ligne['Prix_seuil'] else 0,
                        prix_limit=float(ligne['Prix_limit']
                                        ) if ligne['Prix_limit'] else 0,
                        true1=float(ligne['True1']) if ligne['True1'] else None,
                        true2=float(ligne['True2']) if ligne['True2'] else None,
                        true1_triggered=ligne.get('True1_triggered', 'False') == 'True',
                        true2_triggered=ligne.get('True2_triggered', 'False') == 'True',
                        comment=ligne['Comment'] if ligne['Comment'] else '',
                        status=ligne['Status'] if ligne['Status'] else 'N/A',
                        prix_courant=float(ligne['Prix_courant']
                                        ) if ligne['Prix_courant'] else 0,
                        link=ligne['Link'] if ligne['Link'] else '',
                        pourcentage=ligne['Pourcentage'] if ligne['Pourcentage'] else 0,
                        X=ligne['X'] if ligne['X'] else 0,
                        A=ligne['A'] if ligne['A'] else 0,
                        B=ligne['B'] if ligne['B'] else 0,
                        C=ligne['C'] if ligne['C'] else 0,
                        D=ligne['D'] if ligne['D'] else 0,
                        old_status=ligne['OldStatus'] if ligne['OldStatus'] else ligne['Status'],
                        has_4h_close_above=ligne.get('Has4H', 'False') == 'True',
                        has_1_retest=ligne.get('Has1H', 'False') == 'True',
                        has_sl=ligne.get('HasSL', 'False') == 'True',
                        has_tp1=ligne.get('HasTP1', 'False') == 'True',
                        has_tp2=ligne.get('HasTP2', 'False') == 'True',
                        has_tp3=ligne.get('HasTP3', 'False') == 'True',
                        has_tp4=ligne.get('HasTP4', 'False') == 'True',
                        has_tp5=ligne.get('HasTP5', 'False') == 'True',
                        trailingSL=ligne.get('TrailingSL', 'No'),
                        wantRetest=ligne.get('WantRetest', 'No'),
                        touched_entry=ligne.get('Touched_entry', 'False') == 'True',
                        true3=ligne.get('True3', 0),
                        true4=ligne.get('True4', 0),
                        tf=ligne.get('Tf', '4H'),
                        PreEntrySlightTouch=ligne.get('PreEntrySlightTouch') == 'False',
                        PostEntrySlightTouch=ligne.get('PostEntrySlightTouch') == 'False',
                        PreLimitSlightTouch=ligne.get('PreLimitSlightTouch') == 'False',
                        PostLimitSlightTouch=ligne.get('PostLimitSlightTouch') == 'False',
                        PreTrue1SlightTouch=ligne.get('PreTrue1SlightTouch') == 'False',
                        PostTrue1SlightTouch=ligne.get('PostTrue1SlightTouch') == 'False',
                        PreTrue2SlightTouch=ligne.get('PreTrue2SlightTouch') == 'False',
                        PostTrue2SlightTouch=ligne.get('PostTrue2SlightTouch') == 'False',
                        confidence=ligne.get('Confidence',3)
                    )
                    trades.append(trade)
        except Exception as e:
            logging.error(f"Error loading trades from {nom_fichier}: {e}")
        return trades



# endregion
