import csv
import logging
import os
from typing import List, Optional, Dict, Any, Union
from google.cloud import storage

bucket_project = "adept-coda-420809"

# region indices
class Trade:
    """
    Représente un ordre de trading unique et encapsule tout son cycle de vie et ses métadonnées.

    Cette classe centralise l'état d'une opportunité de trading (entrée, cibles, stop-loss)
    ainsi que son suivi en temps réel (prix courant, déclenchements). Elle permet au système
    de prendre des décisions cohérentes (fermeture, ajustement) en se basant sur une source
    de vérité unique pour chaque position.
    """
    def __init__(self,
                 id: str,
                 owner: str,
                 nom: str,
                 direction: str,
                 harmonic: str,
                 prix_seuil: float,
                 prix_limit: float,
                 true1: Optional[float] = None,
                 true2: Optional[float] = None,
                 true1_triggered: bool = False,
                 true2_triggered: bool = False,
                 prix_courant: float = 0.0,
                 comment: str = '',
                 status: str = '',
                 link: str = '',
                 pourcentage: Union[float, str] = 0,
                 X: float = 0,
                 A: float = 0,
                 B: float = 0,
                 C: float = 0,
                 D: float = 0,
                 old_status: str = '',
                 has_4h_close_above: bool = False,
                 has_1_retest: bool = False,
                 has_sl: bool = False,
                 has_tp1: bool = False,
                 has_tp2: bool = False,
                 has_tp3: bool = False,
                 has_tp4: bool = False,
                 has_tp5: bool = False,
                 trailingSL: str = 'No',
                 wantRetest: str = 'No',
                 touched_entry: bool = False,
                 true3: float = 0,
                 true4: float = 0,
                 tf: str = '4h',
                 PreEntrySlightTouch: bool = False,
                 PostEntrySlightTouch: bool = False,
                 PreLimitSlightTouch: bool = False,
                 PostLimitSlightTouch: bool = False,
                 PreTrue1SlightTouch: bool = False,
                 PostTrue1SlightTouch: bool = False,
                 PreTrue2SlightTouch: bool = False,
                 PostTrue2SlightTouch: bool = False,
                 confidence: int = 3,
                 invalidation: bool = False,
                 has_3d_close_above: bool = False,
                 open_3min: float = 0,
                 close_3min: float = 0,
                 high_3min: float = 0,
                 low_3min: float = 0,
                 open_4h: float = 0,
                 close_4h: float = 0,
                 high_4h: float = 0,
                 low_4h: float = 0,
                 purity_score: float = 0.0,
                 confluence_score: float = 0.0,
                 historical_win_rate: float = 0.0):
        """
        Initialise une nouvelle instance de Trade avec tous ses paramètres de configuration et d'état.

        Les paramètres sont nombreux car l'objet doit persister l'intégralité du contexte technique
        (points harmoniques X, A, B, C, D) et opérationnel (flags de déclenchement) pour survivre
        aux redémarrages du système.
        """
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
        self.purity_score = purity_score
        self.confluence_score = confluence_score
        self.historical_win_rate = historical_win_rate
        
        

    def mise_a_jour_prix(self, data: List[Dict[str, Any]]) -> None:
        """
        Met à jour le prix courant du trade en analysant le flux de données marché.

        Cette méthode itère sur les données reçues pour trouver le symbole correspondant à ce trade.
        La mise à jour est critique pour déclencher les calculs de pourcentage et vérifier les
        conditions d'entrée ou de sortie lors des cycles suivants.

        Args:
            data (List[Dict[str, Any]]): Une liste de dictionnaires contenant les dernières données de marché (symbol, lastPrice, etc.).
        """
        symbol = self.nom + 'USDT'
        

        for item in data:
            if item['symbol'] == symbol:
                self.prix_courant = float(item['lastPrice'])
                self.pourcentage = self.pourcentage_difference()

                break
    
    
    def pourcentage_difference(self) -> str:
        """
        Calcule l'écart en pourcentage entre le prix courant et le prix seuil (entrée).

        Ce calcul est utilisé pour évaluer la proximité du prix par rapport au point d'entrée,
        permettant de filtrer ou de prioriser les notifications d'opportunités imminentes.
        Retourne une chaîne formatée pour l'affichage direct dans les interfaces ou logs.

        Returns:
            str: Le pourcentage de différence formaté à deux décimales, ou une valeur élevée si le prix est nul.
        """
        if self.prix_courant == 0:
            return "99999.00" # Retourner une string pour la cohérence du type de retour attendu
        return format(float(abs((self.prix_courant - self.prix_seuil) / self.prix_seuil) * 100),  ".2f") 


    # Function to check if the 4-hour close above or below the entry value
    def check_close_above_entry(self, entry: Union[float, str]) -> bool:
        """
        Vérifie si la clôture de la bougie 4H a franchi le niveau d'entrée dans la direction opposée.

        Cette vérification est essentielle pour valider la cassure d'un niveau (breakout) ou
        détecter une invalidation potentielle du setup selon la direction du trade (SHORT ou LONG).

        Args:
            entry (Union[float, str]): Le prix d'entrée à comparer.

        Returns:
            bool: True si la clôture confirme le franchissement, False sinon.
        """
        close = float(self.close_4h)
        entry_val = float(entry)
        if self.direction == "SHORT":
            return close < entry_val
        else:
            return close > entry_val


# region import/export
class TradeManager:
    """
    Gère la persistance et la récupération des objets Trade depuis le stockage Cloud ou local.

    Cette classe abstraie la complexité des interactions I/O (CSV, Google Cloud Storage) pour
    fournir aux services de haut niveau des méthodes simples de chargement et de sauvegarde
    des portefeuilles de trades.
    """
    def __init__(self, bucket_name: str):
        """
        Initialise le gestionnaire avec le nom du bucket cible.

        Args:
            bucket_name (str): Le nom du bucket Google Cloud Storage utilisé pour la persistance.
        """
        self.bucket_name = bucket_name


    def charger_trades_depuis_cloud(self, nom_fichier: str) -> List[Trade]:
        """
        Récupère et instancie une liste d'objets Trade depuis un fichier CSV stocké sur le Cloud.

        Cette méthode est utilisée au démarrage des services ou lors des cycles de rafraîchissement
        pour restaurer l'état complet du système à partir de la source de vérité persistante.

        Args:
            nom_fichier (str): Le nom du fichier CSV dans le bucket.

        Returns:
            List[Trade]: Une liste d'instances de la classe Trade.
        """
        indices = []
        client = storage.Client(project=bucket_project)
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
                X=float(ligne['X']) if ligne['X'] else 0,
                A=float(ligne['A']) if ligne['A'] else 0,
                B=float(ligne['B']) if ligne['B'] else 0,
                C=float(ligne['C']) if ligne['C'] else 0,
                D=float(ligne['D']) if ligne['D'] else 0,
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
                true3=float(ligne.get('True3', 0)),
                true4=float(ligne.get('True4', 0)),
                tf=ligne.get('Tf', '4H'),
                PreEntrySlightTouch=ligne.get('PreEntrySlightTouch') == 'True',
                PostEntrySlightTouch=ligne.get('PostEntrySlightTouch') == 'True',
                PreLimitSlightTouch=ligne.get('PreLimitSlightTouch') == 'True',
                PostLimitSlightTouch=ligne.get('PostLimitSlightTouch') == 'True',
                PreTrue1SlightTouch=ligne.get('PreTrue1SlightTouch') == 'True',
                PostTrue1SlightTouch=ligne.get('PostTrue1SlightTouch') == 'True',
                PreTrue2SlightTouch=ligne.get('PreTrue2SlightTouch') == 'True',
                PostTrue2SlightTouch=ligne.get('PostTrue2SlightTouch') == 'True',
                confidence=int(ligne.get('Confidence', 3)),
                invalidation=ligne.get('Invalidation') == 'True',
                has_3d_close_above=ligne.get('Has3dClosedAbove') == 'True',
                purity_score=float(ligne.get('purity_score', 0.0)),
                confluence_score=float(ligne.get('confluence_score', 0.0)),
                historical_win_rate=float(ligne.get('historical_win_rate', 0.0)),
            )
            indices.append(indice)
        return indices


    def enregistrer_trades_dans_csv(self, indices: List[Trade], nom_fichier: str) -> None:
        """
        Sauvegarde l'état actuel d'une liste de trades dans un fichier CSV sur le Cloud.

        Cette méthode assure la persistance des données après chaque cycle de mise à jour,
        garantissant que les modifications (statuts, prix, triggers) sont sécurisées et
        partagées entre les microservices.

        Args:
            indices (List[Trade]): La liste des objets Trade à sauvegarder.
            nom_fichier (str): Le nom du fichier cible dans le bucket.
        """
        client = storage.Client(project=bucket_project)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(nom_fichier)

        entetes = ['Id', 'Owner', 'Nom', 'Direction', 'Harmonic', 'Prix_seuil', 'Prix_limit',
                'True1', 'True2', 'True1_triggered', 'True2_triggered', 'Comment', 'Prix_courant', 'Status', 'Link', 'Pourcentage', 'X', 'A', 'B', 'C', 'D', 'OldStatus', 'Has4H', 'Has1H', 'HasSL', 'HasTP1', 'HasTP2', 'HasTP3', 'HasTP4', 'HasTP5', 'TrailingSL', 'WantRetest', 'Touched_entry', 'True3', 'True4', 'Tf',
                'PreEntrySlightTouch', 'PostEntrySlightTouch', 'PreLimitSlightTouch', 'PostLimitSlightTouch', 'PreTrue1SlightTouch', 'PostTrue1SlightTouch', 'PreTrue2SlightTouch', 'PostTrue2SlightTouch', 'Confidence', 'Invalidation', 'Has3dClosedAbove',
                'purity_score', 'confluence_score', 'historical_win_rate']
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
                indice.purity_score,
                indice.confluence_score,
                indice.historical_win_rate,
            ])

        content = "\n".join([",".join(map(str, ligne)) for ligne in lignes])
        blob.upload_from_string(content)
        
        

    def charger_trades_depuis_fichier_local(self, nom_fichier: str) -> List[Trade]:
        """
        Charge les trades depuis un fichier CSV local.

        Cette méthode est utile pour le développement local, les tests unitaires ou
        lorsque l'accès au Cloud n'est pas disponible, permettant de simuler le comportement
        du système avec des données statiques.

        Args:
            nom_fichier (str): Le chemin vers le fichier CSV local.

        Returns:
            List[Trade]: Une liste d'objets Trade.
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
                        X=float(ligne['X']) if ligne['X'] else 0,
                        A=float(ligne['A']) if ligne['A'] else 0,
                        B=float(ligne['B']) if ligne['B'] else 0,
                        C=float(ligne['C']) if ligne['C'] else 0,
                        D=float(ligne['D']) if ligne['D'] else 0,
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
                        true3=float(ligne.get('True3', 0)),
                        true4=float(ligne.get('True4', 0)),
                        tf=ligne.get('Tf', '4H'),
                        PreEntrySlightTouch=ligne.get('PreEntrySlightTouch') == 'False',
                        PostEntrySlightTouch=ligne.get('PostEntrySlightTouch') == 'False',
                        PreLimitSlightTouch=ligne.get('PreLimitSlightTouch') == 'False',
                        PostLimitSlightTouch=ligne.get('PostLimitSlightTouch') == 'False',
                        PreTrue1SlightTouch=ligne.get('PreTrue1SlightTouch') == 'False',
                        PostTrue1SlightTouch=ligne.get('PostTrue1SlightTouch') == 'False',
                        PreTrue2SlightTouch=ligne.get('PreTrue2SlightTouch') == 'False',
                        PostTrue2SlightTouch=ligne.get('PostTrue2SlightTouch') == 'False',
                        confidence=int(ligne.get('Confidence',3)),
                        purity_score=float(ligne.get('purity_score', 0.0)),
                        confluence_score=float(ligne.get('confluence_score', 0.0)),
                        historical_win_rate=float(ligne.get('historical_win_rate', 0.0)),
                    )
                    trades.append(trade)
        except Exception as e:
            logging.error(f"Error loading trades from {nom_fichier}: {e}")
        return trades



# endregion
