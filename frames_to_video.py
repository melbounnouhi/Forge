import cv2, glob, sys, os

dossier = sys.argv[1]
sortie = sys.argv[2]
fps = 30

images = sorted(glob.glob(os.path.join(dossier, "*.jpg")))
h, w = cv2.imread(images[0]).shape[:2]
video = cv2.VideoWriter(sortie, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for chemin in images:
    video.write(cv2.imread(chemin))
video.release()
print(f"OK : {len(images)} images -> {sortie}")