"""
FORGE v0.1 — Transformer une vidéo de poste de travail en données.
=================================================================
Ce que fait ce script :
  1. Ouvre une vidéo (fichier .mp4)
  2. Détecte les personnes et objets image par image (YOLO)
  3. Mesure le temps passé par les personnes dans une "zone" définie
  4. Compte les traversées de l'écran (allers-retours)
  5. Sort un rapport texte + un fichier JSON de données

AVANT DE LANCER (une seule fois) :
  pip install ultralytics opencv-python

POUR LANCER :
  python forge_v01.py ma_video.mp4

Astuce pour récupérer une vidéo de test : cherche sur YouTube
"workshop timelapse" ou "garage camera" et télécharge-la en mp4.
"""

import sys
import json
import time
from collections import defaultdict

import cv2  # OpenCV : lire la vidéo, dessiner dessus
from ultralytics import YOLO  # Le modèle de détection


# ---------------------------------------------------------------
# RÉGLAGES — c'est ICI que tu modifieras le plus au début
# ---------------------------------------------------------------

# La "zone de travail" à surveiller, en % de l'image.
# (x_min, y_min, x_max, y_max) → ici : le quart en bas à gauche.
# EXERCICE 1 : change ces valeurs pour cibler le poste de travail
# visible dans TA vidéo.
ZONE = (0.5, 0.3, 1.0, 0.8)
ZONE2 = (0.0, 0.4, 0.35, 1.0)   # zone 2 : côté gauche (etabli/ponts)


# On n'analyse qu'une image sur N (la vidéo fait ~25-30 img/sec,
# analyser chaque image serait lent et inutile).
# EXERCICE 2 : essaie 5, 15, 30 et observe la différence de vitesse.
UNE_IMAGE_SUR = 10

# Les objets qui nous intéressent (noms YOLO en anglais).
# EXERCICE 3 : ajoute "car" si tu analyses un garage,
# "bottle", "cup"... Liste complète : cherche "COCO classes".
CLASSES_SUIVIES = {"person"}


# ---------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------

def centre_dans_zone(boite, largeur, hauteur):
    """Vrai si le centre d'une boîte de détection est dans la ZONE."""
    x1, y1, x2, y2 = boite
    cx = (x1 + x2) / 2 / largeur   # centre x, en % de l'image
    cy = (y1 + y2) / 2 / hauteur   # centre y, en % de l'image
    zx1, zy1, zx2, zy2 = ZONE
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def formater_duree(secondes):
    """834 secondes → '13 min 54 s' (plus lisible pour le rapport)."""
    m, s = divmod(int(secondes), 60)
    return f"{m} min {s:02d} s"


# ---------------------------------------------------------------
# PROGRAMME PRINCIPAL
# ---------------------------------------------------------------

def analyser(chemin_video):
    print(f"🔧 Forge v0.1 — analyse de : {chemin_video}")

    # 1. Charger le modèle. "yolov8n" = version nano : rapide,
    #    parfaite pour débuter. (Se télécharge tout seul au 1er lancement.)
    modele = YOLO("yolov8n.pt")

    # 2. Ouvrir la vidéo
    video = cv2.VideoCapture(chemin_video)
    if not video.isOpened():
        print("❌ Impossible d'ouvrir la vidéo. Vérifie le chemin.")
        return

    ips = video.get(cv2.CAP_PROP_FPS) or 25          # images/seconde
    largeur = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duree_par_analyse = UNE_IMAGE_SUR / ips  # temps réel entre 2 analyses

    # 3. Les compteurs — le cœur des "données" qu'on extrait
    temps_en_zone = 0.0                    # secondes de présence en zone
    presences = defaultdict(float)          # temps de présence par objet
    nb_max_personnes = 0
    allers_retours = 0
    derniere_zone = None
    zone_par_id = {}
    trajets_par_id = defaultdict(int)                    # record de monde simultané
    dans_zone_avant = False
    entrees_en_zone = 0                     # nb de fois où qqn ENTRE en zone
    num_image = 0
    debut = time.time()

    # 4. Boucle principale : on lit la vidéo image par image
    while True:
        ok, image = video.read()
        if not ok:
            break  # fin de la vidéo
        num_image += 1
        if num_image % UNE_IMAGE_SUR != 0:
            continue  # on saute cette image (voir UNE_IMAGE_SUR)

        # LA ligne magique : YOLO détecte tout ce qu'il voit
        resultats = modele.track(image, verbose=False, persist=True, conf=0.4)[0]

        personnes_visibles = 0
        qqn_en_zone = False

        for boite in resultats.boxes:
            nom = modele.names[int(boite.cls)]       # ex : "person"
            if nom not in CLASSES_SUIVIES:
                continue
            presences[nom] += duree_par_analyse
            if nom == "person":
                personnes_visibles += 1
                coords = boite.xyxy[0].tolist()      # [x1, y1, x2, y2]
                if centre_dans_zone(coords, largeur, hauteur):
                    qqn_en_zone = True
                    zone_actuelle = "DROITE"
                else:
                    cx = (coords[0] + coords[2]) / 2 / largeur
                    cy = (coords[1] + coords[3]) / 2 / hauteur
                    if ZONE2[0] <= cx <= ZONE2[2] and ZONE2[1] <= cy <= ZONE2[3]:
                        zone_actuelle = "GAUCHE"
                    else:
                        zone_actuelle = None
                pid = int(boite.id) if boite.id is not None else -1
                if zone_actuelle is not None and pid != -1:
                    if pid in zone_par_id and zone_par_id[pid] != zone_actuelle:
                        trajets_par_id[pid] += 1
                        allers_retours += 1
                    zone_par_id[pid] = zone_actuelle
                    
                        
                    
                    

        # Mise à jour des stats de zone
        if qqn_en_zone:
            temps_en_zone += duree_par_analyse
            if not dans_zone_avant:
                entrees_en_zone += 1   # quelqu'un vient d'ENTRER
        dans_zone_avant = qqn_en_zone
        nb_max_personnes = max(nb_max_personnes, personnes_visibles)
        # --- MISSION 2 : affichage en direct ---
        image_annotee = resultats.plot()
        h, w = image_annotee.shape[:2]
        cv2.rectangle(image_annotee,
                      (int(ZONE[0]*w), int(ZONE[1]*h)),
                      (int(ZONE[2]*w), int(ZONE[3]*h)),
                      (255, 0, 0), 2)
        cv2.imshow("FORGE - vision en direct", image_annotee)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Petit affichage de progression toutes les ~300 images
        if num_image % (UNE_IMAGE_SUR * 30) == 0:
            print(f"  ... image {num_image} analysée")

    video.release()
    cv2.destroyAllWindows()
    duree_video = num_image / ips

    # 5. LE RAPPORT — la partie que ton futur "client" lira
    rapport = f"""
=====================================
   RAPPORT FORGE v0.1
=====================================
Vidéo analysée   : {chemin_video}
Durée            : {formater_duree(duree_video)}
Temps d'analyse  : {int(time.time() - debut)} s

--- ZONE DE TRAVAIL {ZONE} ---
Temps occupé     : {formater_duree(temps_en_zone)}
Taux d'occupation: {100 * temps_en_zone / max(duree_video, 1):.0f} %
Entrées en zone  : {entrees_en_zone} fois
Allers-retours G<->D : {allers_retours} fois
Personnes suivies : {len(zone_par_id)}

--- PRÉSENCES DÉTECTÉES ---"""
    for nom, secs in sorted(presences.items()):
        rapport += f"\n{nom:<12} : {formater_duree(secs)} à l'écran"
    rapport += f"\nMax simultané    : {nb_max_personnes} personne(s)\n"
    for pid, nb in sorted(trajets_par_id.items()):
        rapport += f"\nPersonne #{pid:<3}: {nb} changement(s) de zone"

    print(rapport)

    # 6. Les données brutes en JSON (le vrai "produit" de Forge :
    #    des données réutilisables, pas juste un texte)
    donnees = {
        "video": chemin_video,
        "duree_video_s": round(duree_video, 1),
        "zone": ZONE,
        "temps_en_zone_s": round(temps_en_zone, 1),
        "taux_occupation_pct": round(100 * temps_en_zone / max(duree_video, 1), 1),
        "entrees_en_zone": entrees_en_zone,
        "presences_s": {k: round(v, 1) for k, v in presences.items()},
        "max_personnes_simultanees": nb_max_personnes,
    }
    with open("forge_rapport.json", "w", encoding="utf-8") as f:
        json.dump(donnees, f, indent=2, ensure_ascii=False)
    print("💾 Données sauvegardées dans forge_rapport.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python forge_v01.py ma_video.mp4")
    else:
        analyser(sys.argv[1])


# ---------------------------------------------------------------
# TES 5 PROCHAINES MISSIONS (dans l'ordre, une par session) :
# ---------------------------------------------------------------
# 1. Le faire tourner sur une vraie vidéo YouTube d'atelier
# 2. Ajuster ZONE pour qu'elle corresponde au poste de travail
# 3. Ajouter une 2e zone (ex : l'armoire à outils) et compter
#    les allers-retours entre les deux  ← LA stat qui vaut de l'or
# 4. Afficher la vidéo avec les rectangles dessinés dessus
#    (indice : cv2.rectangle + cv2.imshow) → tes premières démos
# 5. Générer le rapport en une page HTML jolie au lieu du texte
# ---------------------------------------------------------------
